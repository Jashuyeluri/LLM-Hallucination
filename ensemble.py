import time

import calibration
import settings
from entailment_check import verify_with_llm
from grounding_guard import apply_guard
from nli_verifier import is_available, verify_with_nli
from result_schema import build_result, set_confidence
from sentence_utils import split_sentences


def _merge_evidence(first, second):
    merged = []
    seen = set()
    for item in list(first) + list(second):
        if item["id"] not in seen:
            seen.add(item["id"])
            merged.append(item)
    return merged


def combine(llm_result, nli_result, allow_llm_support=False):
    evidence = _merge_evidence(llm_result["evidence"], nli_result["evidence"])
    shared = {
        "verifier": "llm+nli",
        "llm_label": llm_result["label"],
        "nli_label": nli_result["label"],
    }
    if llm_result["label"] == nli_result["label"]:
        confidence = (llm_result["confidence"] + nli_result["confidence"]) / 2
        return build_result(llm_result["claim"], llm_result["label"], confidence, evidence, agreement=True, **shared)
    if (
        allow_llm_support
        and llm_result["label"] == "supported"
        and nli_result["label"] == "unsupported"
        and llm_result["confidence"] >= 0.80
        and bool(llm_result.get("evidence"))
    ):
        return build_result(
            llm_result["claim"],
            "supported",
            llm_result["confidence"],
            llm_result["evidence"],
            agreement=False,
            reason="strong_llm_support_nli_uncertain",
            **shared,
        )

    confidence = min(llm_result["confidence"], nli_result["confidence"]) * 0.5
    return build_result(llm_result["claim"], "needs_review", confidence, evidence, agreement=False, **shared)


def _nli_only(result):
    result["llm_label"] = None
    result["nli_label"] = result["label"]
    result["agreement"] = None
    return result


def _accepts(result, threshold):
    return result["label"] in ("supported", "contradicted") and result["confidence"] >= threshold


def _apply_calibration(results):
    calibrator = calibration.get_calibrator()
    if calibrator is None:
        return False
    floor = settings.review_below()
    for result in results:
        if result["label"] == "needs_review":
            continue
        result["raw_confidence"] = result["confidence"]
        set_confidence(result, calibrator.predict(result["confidence"]))
        if result["confidence"] < floor:
            result["original_label"] = result["label"]
            result["label"] = "needs_review"
    return True


def _run_llm_only(source_text, sentences, claims, model, require_evidence):
    results = verify_with_llm(source_text, sentences, claims, model=model, require_evidence=require_evidence)
    return results, 0, len(claims), 1


def _run_full(source_text, sentences, claims, model, require_evidence, predictor):
    llm_results = verify_with_llm(source_text, sentences, claims, model=model, require_evidence=require_evidence)
    nli_results = verify_with_nli(sentences, claims, predictor=predictor)
    results = [combine(llm, nli) for llm, nli in zip(llm_results, nli_results)]
    return results, 0, len(claims), 1


def _run_cascade(source_text, sentences, claims, model, require_evidence, predictor, threshold):
    nli_results = verify_with_nli(sentences, claims, predictor=predictor)
    escalate = [index for index, result in enumerate(nli_results) if not _accepts(result, threshold)]
    results = [_nli_only(result) for result in nli_results]
    llm_requests = 0
    if escalate:
        llm_results = verify_with_llm(
            source_text, sentences, [claims[index] for index in escalate],
            model=model, require_evidence=require_evidence,
        )
        for index, llm_result in zip(escalate, llm_results):
            results[index] = combine(llm_result, nli_results[index], allow_llm_support=True)
        llm_requests = 1
    return results, len(claims) - len(escalate), len(escalate), llm_requests


def verify_claims(
    source_text,
    claims,
    model=None,
    mode=None,
    threshold=None,
    nli_predictor=None,
    use_calibration=True,
    require_evidence=True,
    use_guard=None,
):
    started = time.perf_counter()
    requested = mode or settings.verifier_mode()
    nli_ready = nli_predictor is not None or is_available()
    effective = requested if nli_ready else "llm"
    limit = settings.cascade_threshold() if threshold is None else threshold
    sentences = split_sentences(source_text)

    if not claims:
        results, by_nli, escalated, requests = [], 0, 0, 0
    elif effective == "llm":
        results, by_nli, escalated, requests = _run_llm_only(source_text, sentences, claims, model, require_evidence)
    elif effective == "full":
        results, by_nli, escalated, requests = _run_full(
            source_text, sentences, claims, model, require_evidence, nli_predictor
        )
    else:
        results, by_nli, escalated, requests = _run_cascade(
            source_text, sentences, claims, model, require_evidence, nli_predictor, limit
        )

    calibrated = _apply_calibration(results) if use_calibration else False
    guard_on = settings.guard_enabled() if use_guard is None else use_guard
    guard_downgrades = apply_guard(results) if guard_on else 0

    return {
        "results": results,
        "metrics": {
            "mode_requested": requested,
            "mode_used": effective,
            "nli_available": nli_ready,
            "threshold": limit,
            "claims_total": len(claims),
            "claims_resolved_by_nli": by_nli,
            "claims_escalated": escalated,
            "llm_calls_made": requests,
            "calibrated": calibrated,
            "guard_enabled": guard_on,
            "guard_downgrades": guard_downgrades,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    }
