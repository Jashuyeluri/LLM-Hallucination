"""Hybrid RAG retrieval for source-grounded claim verification.

The source remains the authority. Retrieval narrows long documents to relevant
chunks; it never generates facts. BM25 is dependency-free, while semantic
retrieval and FAISS are optional accelerators.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache

from sentence_utils import content_tokens, split_sentences


def _chunks(text: str, max_chars: int = 1200, overlap_sentences: int = 1):
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks, current = [], []
    current_len = 0
    for sentence in sentences:
        addition = len(sentence["text"]) + 1
        if current and current_len + addition > max_chars:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(x["text"]) + 1 for x in current)
        current.append(sentence)
        current_len += addition
    if current:
        chunks.append(current)
    return [
        {
            "id": i,
            "sentences": group,
            "text": " ".join(x["text"] for x in group),
            "start": group[0]["start"],
            "end": group[-1]["end"],
        }
        for i, group in enumerate(chunks)
    ]


def _bm25_scores(query: str, chunks):
    qterms = set(content_tokens(query))
    if not qterms or not chunks:
        return [0.0] * len(chunks)
    tokenized = [content_tokens(item["text"]) for item in chunks]
    lengths = [len(tokens) for tokens in tokenized]
    avg_len = sum(lengths) / max(1, len(lengths))
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))
    scores = []
    k1, b = 1.5, 0.75
    for tokens, length in zip(tokenized, lengths):
        frequencies = Counter(tokens)
        score = 0.0
        for term in qterms:
            tf = frequencies[term]
            if not tf:
                continue
            idf = math.log(1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / max(avg_len, 1)))
        scores.append(score)
    return scores


@lru_cache(maxsize=1)
def _embedding_model():
    if os.getenv("RAG_ENABLE_SEMANTIC", "true").lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    except Exception:
        return None


def _build_semantic_index(chunks):
    model = _embedding_model()
    if model is None or not chunks:
        return None
    try:
        import numpy as np
        vectors = np.asarray(model.encode([x["text"] for x in chunks], normalize_embeddings=True), dtype="float32")
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return (model, index, "faiss")
        except ImportError:
            return (model, vectors, "numpy")
    except Exception:
        return None


def _semantic_ranking(query: str, semantic_index, count: int):
    if semantic_index is None:
        return None, None
    try:
        import numpy as np
        model, index, backend = semantic_index
        query_vector = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
        if backend == "faiss":
            _, indices = index.search(query_vector, count)
            return indices[0].tolist(), "hybrid_bm25_faiss"
        similarities = (index @ query_vector[0]).tolist()
        return sorted(range(count), key=lambda i: similarities[i], reverse=True), "hybrid_bm25_numpy"
    except Exception:
        return None, None


def retrieve_context(source_text: str, queries, top_k: int = 4, max_chars: int = 6000):
    """Return relevant source sentences and provenance for one or more queries."""
    chunks = _chunks(source_text)
    if not chunks:
        return {"text": "", "sentences": [], "chunks_retrieved": 0, "retrieval_mode": "empty"}
    selected = {}
    retrieval_mode = "bm25_fallback"
    semantic_index = _build_semantic_index(chunks)
    for query in queries or [source_text[:500]]:
        lexical = _bm25_scores(query, chunks)
        semantic_order, semantic_mode = _semantic_ranking(query, semantic_index, len(chunks))
        if semantic_order is not None:
            retrieval_mode = semantic_mode
            lex_order = sorted(range(len(chunks)), key=lambda i: lexical[i], reverse=True)
            fused = {i: 0.0 for i in range(len(chunks))}
            for rank, idx in enumerate(lex_order):
                fused[idx] += 1.0 / (60 + rank + 1)
            for rank, idx in enumerate(semantic_order):
                fused[idx] += 1.0 / (60 + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)
        else:
            ranked = sorted(range(len(chunks)), key=lambda i: lexical[i], reverse=True)
        for idx in ranked[:top_k]:
            if lexical[idx] > 0 or semantic_order is not None:
                selected[idx] = chunks[idx]
    if not selected:
        # Keep a small deterministic context so the verifier can label claims unsupported.
        selected[0] = chunks[0]
    sentences_by_span = {}
    for chunk in selected.values():
        for sentence in chunk["sentences"]:
            sentences_by_span[(sentence["start"], sentence["end"])] = sentence
    ordered = sorted(sentences_by_span.values(), key=lambda x: x["start"])
    kept, used = [], 0
    for sentence in ordered:
        cost = len(sentence["text"]) + 1
        if kept and used + cost > max_chars:
            break
        kept.append(sentence)
        used += cost
    context = "\n".join(item["text"] for item in kept)
    return {"text": context, "sentences": kept, "chunks_retrieved": len(selected), "retrieval_mode": retrieval_mode}
