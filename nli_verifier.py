import importlib.util

import settings
from result_schema import build_result
from sentence_utils import top_overlap_sentences

DECISION_FLOOR = 0.5
_state = {"model": None}


def is_available():
    if not settings.enable_nli():
        return False
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def _normalise(outputs):
    scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    for item in outputs:
        label = str(item["label"]).lower()
        if "entail" in label:
            scores["entailment"] = float(item["score"])
        elif "contra" in label:
            scores["contradiction"] = float(item["score"])
        elif "neutral" in label:
            scores["neutral"] = float(item["score"])
    return scores


def _load_model():
    if _state["model"] is None:
        from transformers import pipeline
        _state["model"] = pipeline("text-classification", model=settings.nli_model(), device=-1)
    return _state["model"]


def predict_pairs(pairs):
    if not pairs:
        return []
    classifier = _load_model()
    inputs = [{"text": premise, "text_pair": hypothesis} for premise, hypothesis in pairs]
    outputs = classifier(inputs, top_k=None, truncation=True)
    if outputs and isinstance(outputs[0], dict):
        outputs = [outputs]
    return [_normalise(item) for item in outputs]


def _premises_for(claim, sentences, top_k):
    chosen = top_overlap_sentences(claim, sentences, top_k)
    premises = [(item["text"], [item]) for item in chosen]
    if len(chosen) > 1:
        ordered = sorted(chosen, key=lambda item: item["id"])
        premises.append((" ".join(item["text"] for item in ordered), ordered))
    return premises


def _decide(entries):
    if not entries:
        return "unsupported", 0.0, []
    best_entail = max(entries, key=lambda entry: entry[0]["entailment"])
    best_contra = max(entries, key=lambda entry: entry[0]["contradiction"])
    entail = best_entail[0]["entailment"]
    contra = best_contra[0]["contradiction"]
    if entail >= DECISION_FLOOR and entail >= contra:
        return "supported", entail, best_entail[1]
    if contra >= DECISION_FLOOR:
        return "contradicted", contra, best_contra[1]
    return "unsupported", 1.0 - max(entail, contra), []


def verify_with_nli(sentences, claims, predictor=None, top_k=3):
    if not claims:
        return []
    predict = predictor or predict_pairs
    plans = [_premises_for(claim, sentences, top_k) for claim in claims]
    pairs = [(text, claim) for claim, premises in zip(claims, plans) for text, _ in premises]
    scores = iter(predict(pairs) if pairs else [])

    results = []
    for claim, premises in zip(claims, plans):
        entries = [(next(scores), evidence) for _, evidence in premises]
        label, confidence, evidence = _decide(entries)
        results.append(build_result(claim, label, confidence, evidence, verifier="nli"))
    return results
