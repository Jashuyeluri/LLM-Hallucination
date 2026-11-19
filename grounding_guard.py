import re

import settings
from result_schema import set_confidence
from sentence_utils import content_tokens

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9']+")
_CAPITALIZED = re.compile(r"[A-Z][A-Za-z'-]+")
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor|without|cannot)\b|n't\b", re.IGNORECASE)


def _numbers(text):
    found = {match.replace(",", "") for match in _NUMBER.findall(text)}
    tokens = re.findall(r"[a-z]+", text.lower())
    index = 0
    while index < len(tokens):
        word = tokens[index]
        if word not in _NUMBER_WORDS:
            index += 1
            continue
        value = _NUMBER_WORDS[word]
        if value >= 20 and index + 1 < len(tokens) and tokens[index + 1] in _NUMBER_WORDS:
            unit = _NUMBER_WORDS[tokens[index + 1]]
            if 1 <= unit <= 9:
                value += unit
                index += 1
        found.add(str(value))
        index += 1
    return found


def _entities(claim):
    """Extract named entities, excluding a leading multi-word subject."""
    words = claim.split()

    if not words:
        return set()

    start = 1

    if (
        len(words) >= 2
        and _CAPITALIZED.fullmatch(words[0].strip(".,;:!?()\"'"))
        and _CAPITALIZED.fullmatch(words[1].strip(".,;:!?()\"'"))
    ):
        start = 2

    found = set()

    for word in words[start:]:
        cleaned = word.strip(".,;:!?()\"'")

        if cleaned.lower().endswith("'s"):
            cleaned = cleaned[:-2]

        if _CAPITALIZED.fullmatch(cleaned):
            found.add(cleaned.lower())

    return found



def audit(result):
    if result["label"] not in ("supported", "contradicted"):
        return []
    evidence = result.get("evidence") or []
    if not evidence:
        return []
    claim = result["claim"]
    text = " ".join(item["text"] for item in evidence)
    flags = []
    claim_tokens = set(content_tokens(claim))
    if claim_tokens:
        overlap = len(claim_tokens & set(content_tokens(text))) / len(claim_tokens)
        if overlap < settings.guard_min_overlap():
            flags.append("weak_citation")
    if result["label"] == "supported":
        if _numbers(claim) - _numbers(text):
            flags.append("number_not_in_evidence")
        if _entities(claim) - set(_WORD.findall(text.lower())):
            flags.append("entity_not_in_evidence")
        if bool(_NEGATION.search(claim)) != bool(_NEGATION.search(text)):
            flags.append("negation_mismatch")
    return flags


def apply_guard(results):
    downgraded = 0
    for result in results:
        flags = audit(result)
        if not flags:
            continue
        result["guard_flags"] = flags
        result["original_label"] = result.get("original_label", result["label"])
        result["label"] = "needs_review"
        set_confidence(result, min(result["confidence"], 0.5))
        downgraded += 1
    return downgraded
