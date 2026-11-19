import math
import re

_BOUNDARY = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]["\'”’)\]])\s+|\n+')
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "fig"}
_TRAILING_WORD = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "he", "she", "they", "them", "his", "her", "their", "has",
    "have", "had", "do", "does", "did", "not", "no", "so", "than", "then", "there", "also",
}


def _ends_with_abbreviation(segment):
    match = _TRAILING_WORD.search(segment.rstrip())
    if not match:
        return False
    word = match.group(1).lower()
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def _append_span(spans, text, start, end):
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    offset = start + (len(raw) - len(raw.lstrip()))
    spans.append({"text": stripped, "start": offset, "end": offset + len(stripped)})


def split_sentences(text):
    spans = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        segment = text[cursor:match.start()]
        if "\n" not in match.group() and _ends_with_abbreviation(segment):
            continue
        _append_span(spans, text, cursor, match.start())
        cursor = match.end()
    _append_span(spans, text, cursor, len(text))
    return [{"id": index, **span} for index, span in enumerate(spans)]


def content_tokens(text):
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def _weight(token):
    return 2.0 if any(char.isdigit() for char in token) else 1.0


def top_overlap_sentences(claim, sentences, top_k=3):
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return []
    scored = []
    for sentence in sentences:
        shared = claim_tokens & set(content_tokens(sentence["text"]))
        if not shared:
            continue
        score = sum(_weight(token) for token in shared) / math.sqrt(len(claim_tokens))
        scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [sentence for _, sentence in scored[:top_k]]


def jaccard(first, second):
    left, right = set(content_tokens(first)), set(content_tokens(second))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
