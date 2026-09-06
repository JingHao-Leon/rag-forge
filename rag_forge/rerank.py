"""Rerankers: MMR diversity reranking (offline) and a pluggable LLM reranker interface."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dense import HashEmbedder


def mmr_rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 5,
               embedder: object | None = None, lambda_: float = 0.7) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance over (doc_id, text) candidates.

    Balances query relevance against redundancy among the selected results —
    the classic fix for "five chunks that all say the same thing".
    """
    emb = embedder or HashEmbedder()
    q = emb.embed(query)
    texts = [c[1] for c in candidates]
    vecs = np.stack([emb.embed(t) for t in texts]) if texts else np.zeros((0, emb.dim), dtype=np.float32)
    rel = (vecs @ q).tolist()
    selected: list[int] = []
    scores: list[float] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_i, best_s = None, float("-inf")
        for i in remaining:
            div = 1.0 if not selected else max(float(np.dot(vecs[i], vecs[j])) for j in selected)
            s = lambda_ * rel[i] - (1 - lambda_) * div
            if s > best_s:
                best_i, best_s = i, s
        selected.append(best_i)
        scores.append(best_s)
        remaining.remove(best_i)
    return [(candidates[i][0], scores[n]) for n, i in enumerate(selected)]


def make_llm_reranker(rerank_fn: Callable[[str, list[str]], list[int]]) -> Callable:
    """Adapt any LLM that returns a permutation of indices into a rerank function.

    rerank_fn(query, [texts]) -> [indices in preference order].
    Keeps the library LLM-agnostic: bring your own OpenAI/DeepSeek/local call.
    """

    def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 5):
        order = rerank_fn(query, [c[1] for c in candidates])
        seen = set()
        out = []
        for i in order:
            if 0 <= i < len(candidates) and i not in seen:
                out.append((candidates[i][0], 1.0 / (len(out) + 1)))
                seen.add(i)
            if len(out) >= top_k:
                break
        for i, (doc_id, _) in enumerate(candidates):  # keep unranked tail
            if i not in seen and len(out) < top_k:
                out.append((doc_id, 0.0))
        return out

    return rerank
