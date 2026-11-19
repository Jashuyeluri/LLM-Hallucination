import time

import llm_client
import settings
from claim_extraction import extract_claims
from consistency import detect_conflicts
from certificate import build_certificate
from correction import correct_response, needs_changes
from ensemble import verify_claims
from scoring import compute_faithfulness_score
from rag_retriever import retrieve_context
from sentence_utils import split_sentences
from sentence_utils import jaccard

SIMILARITY_FLOOR = 0.6


def _is_known(claim, known_claims):
    return any(jaccard(claim, known) >= SIMILARITY_FLOOR for known in known_claims)


def _new_unsupported(results, known_claims):
    return [
        item["claim"]
        for item in results
        if item["label"] == "unsupported" and not _is_known(item["claim"], known_claims)
    ]


def _verify_with_retrieval(source_text, claims, model, options):
    retrieval = retrieve_context(source_text, claims, top_k=settings.rag_top_k(), max_chars=settings.rag_context_chars())
    context = retrieval["text"] or source_text
    verification = verify_claims(context, claims, model=model, **options)
    # Convert evidence offsets from the retrieved context back to original-source offsets.
    originals = retrieval["sentences"] if retrieval["text"] else split_sentences(source_text)
    for result in verification["results"]:
        mapped = []
        for evidence in result.get("evidence", []):
            idx = evidence.get("id")
            if isinstance(idx, int) and 0 <= idx < len(originals):
                mapped.append(dict(originals[idx]))
        result["evidence"] = mapped
    retrieval["context_chars"] = len(context)
    retrieval["source_chars"] = len(source_text)
    retrieval["context_reduction_pct"] = round((1 - len(context) / max(1, len(source_text))) * 100, 2)
    return verification, retrieval


def _evaluate(source_text, text, model, options):
    claims = extract_claims(text, model=model)
    verification, _ = _verify_with_retrieval(source_text, claims, model, options)
    detect_conflicts(claims, verification["results"], model=model)
    return verification["results"], compute_faithfulness_score(verification["results"])


def _snapshot(iteration, text, score, new_unsupported):
    return {
        "iteration": iteration,
        "text": text,
        "score": score,
        "contradicted": score["contradicted"],
        "unsupported": score["unsupported"],
        "needs_review": score["needs_review"],
        "new_unsupported": len(new_unsupported),
    }


def _still_flawed(results, new_unsupported, policy):
    return needs_changes(results, policy, new_unsupported)


def _correction_loop(source_text, llm_response, results, score, model, options, policy, limit):
    iterations = [_snapshot(0, llm_response, score, [])]
    known = [item["claim"] for item in results if item["label"] != "contradicted"]
    flawed = _still_flawed(results, [], policy)
    if not flawed:
        return None, None, iterations, True, "no_errors"
    if limit <= 0:
        return None, None, iterations, False, "correction_disabled"

    current_text, current_results, new_unsupported = llm_response, results, []
    corrected_text, corrected_score = None, None
    for iteration in range(1, limit + 1):
        candidate = correct_response(
            source_text, current_text, current_results, model=model,
            unsupported_policy=policy, force_remove=new_unsupported,
        )
        if candidate.strip() == current_text.strip():
            return corrected_text, corrected_score, iterations, False, "no_change"
        current_results, corrected_score = _evaluate(source_text, candidate, model, options)
        new_unsupported = _new_unsupported(current_results, known)
        current_text = corrected_text = candidate
        iterations.append(_snapshot(iteration, candidate, corrected_score, new_unsupported))
        if not _still_flawed(current_results, new_unsupported, policy):
            return corrected_text, corrected_score, iterations, True, "converged"
    return corrected_text, corrected_score, iterations, False, "max_iterations"


def run_pipeline(
    source_text,
    llm_response,
    model=None,
    auto_correct=True,
    mode=None,
    threshold=None,
    max_iterations=None,
    unsupported_policy="keep",
    use_calibration=True,
    require_evidence=True,
    source_manifest=None,
    use_guard=None,
):
    started = time.perf_counter()
    limit = settings.max_iterations() if max_iterations is None else max_iterations
    if not auto_correct:
        limit = 0
    options = {
        "mode": mode,
        "threshold": threshold,
        "use_calibration": use_calibration,
        "require_evidence": require_evidence,
        "use_guard": use_guard,
    }

    with llm_client.track_calls() as counter:
        claims = extract_claims(llm_response, model=model)
        verification, rag_metrics = _verify_with_retrieval(source_text, claims, model, options)
        results = verification["results"]
        score = compute_faithfulness_score(results)
        consistency = detect_conflicts(claims, results, model=model)
        corrected, corrected_score, iterations, converged, stop_reason = _correction_loop(
            source_text, llm_response, results, score, model, options, unsupported_policy, limit
        )

    source_manifest = source_manifest or [{
        "id": 1, "name": "Provided source", "kind": "pasted_text",
        "start": 0, "end": len(source_text), "characters": len(source_text),
    }]
    for result in results:
        for evidence in result.get("evidence", []):
            start = evidence.get("start", -1)
            match = next((item for item in source_manifest
                          if item.get("start", 0) <= start < item.get("end", 0)), None)
            if match:
                evidence["source_id"] = match["id"]
                evidence["source_name"] = match["name"]

    total_claims = max(1, len(results))
    unresolved_ratio = (score["unsupported"] + score["needs_review"]) / total_claims
    abstention_recommended = unresolved_ratio >= 0.5 or score["supported"] == 0
    phase_summary = {
        "phase_1_ingestion": {"status": "complete", "source_count": len(source_manifest)},
        "phase_2_retrieval": {"status": "complete", "retrieval_mode": rag_metrics["retrieval_mode"],
                              "chunks_retrieved": rag_metrics["chunks_retrieved"]},
        "phase_3_claim_extraction": {"status": "complete", "claims_extracted": len(claims)},
        "phase_4_verification": {"status": "complete", "verifier_mode": verification["metrics"]["mode_used"],
                                 "guard_downgrades": verification["metrics"]["guard_downgrades"]},
        "phase_5_correction": {"status": "complete" if corrected is not None or stop_reason == "no_errors" else "not_converged",
                               "iterations": len(iterations) - 1, "converged": converged},
        "phase_6_evaluation": {"status": "metrics_generated", "faithfulness_score": score["faithfulness_score"],
                               "abstention_recommended": abstention_recommended},
    }

    final_text = corrected if corrected is not None else llm_response
    certificate = build_certificate(
        source_text, llm_response, results,
        config={
            "verifier_mode": verification["metrics"]["mode_used"],
            "threshold": verification["metrics"]["threshold"],
            "calibrated": verification["metrics"]["calibrated"],
            "guard_enabled": verification["metrics"]["guard_enabled"],
            "require_evidence": require_evidence,
            "model": model,
        },
        corrected_response=corrected,
    )
    return {
        "claims": claims,
        "results": results,
        "score": score,
        "corrected_response": corrected,
        "corrected_score": corrected_score,
        "iterations": iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "consistency_available": consistency["available"],
        "consistency_conflicts": consistency["conflicts"],
        "metrics": {
            **verification["metrics"],
            "rag": rag_metrics,
            "llm_calls_total": counter["calls"],
            "latency_ms_total": round((time.perf_counter() - started) * 1000, 1),
        },
        "certificate": certificate,
        "final_text": final_text,
        "source_text": source_text,
        "source_manifest": source_manifest,
        "phase_summary": phase_summary,
        "abstention_recommended": abstention_recommended,
        "rag": rag_metrics,
    }
