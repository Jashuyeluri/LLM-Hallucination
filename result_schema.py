def _scores(label, confidence):
    if label == "supported":
        return confidence, 0.0, 1.0 - confidence
    if label == "contradicted":
        return 0.0, confidence, 1.0 - confidence
    return 0.0, 0.0, confidence


def build_result(claim, label, confidence, evidence=None, **extra):
    confidence = max(0.0, min(float(confidence), 1.0))
    entailment, contradiction, neutral = _scores(label, confidence)
    result = {
        "claim": claim,
        "label": label,
        "confidence": round(confidence, 4),
        "entailment_score": round(entailment, 4),
        "contradiction_score": round(contradiction, 4),
        "neutral_score": round(neutral, 4),
        "evidence": list(evidence or []),
    }
    result.update(extra)
    return result


def set_confidence(result, confidence):
    confidence = max(0.0, min(float(confidence), 1.0))
    entailment, contradiction, neutral = _scores(result["label"], confidence)
    result["confidence"] = round(confidence, 4)
    result["entailment_score"] = round(entailment, 4)
    result["contradiction_score"] = round(contradiction, 4)
    result["neutral_score"] = round(neutral, 4)
    return result
