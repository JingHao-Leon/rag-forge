"""Retrieval metrics: Recall@k, MRR, nDCG@k on labeled query sets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueryCase:
    query: str
    relevant: set[str]  # doc ids considered relevant


def load_cases(path: str | Path) -> list[QueryCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(QueryCase(query=obj["query"], relevant=set(obj["relevant"])))
    return cases


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked_lists: list[list[str]], relevants: list[set[str]]) -> float:
    """Mean Reciprocal Rank over a set of queries."""
    assert len(ranked_lists) == len(relevants)
    rr = []
    for ranked, rel in zip(ranked_lists, relevants):
        r = 0.0
        for i, d in enumerate(ranked, start=1):
            if d in rel:
                r = 1.0 / i
                break
        rr.append(r)
    return sum(rr) / len(rr) if rr else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG."""
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(search_fn, cases: list[QueryCase], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """search_fn(query, top_k) -> list[doc_id]. Returns averaged metrics."""
    out: dict = {"n_queries": len(cases)}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(search_fn(c.query, k), c.relevant, k) for c in cases) / len(cases)
        out[f"ndcg@{k}"] = sum(ndcg_at_k(search_fn(c.query, k * 2), c.relevant, k) for c in cases) / len(cases)
    ranked = [search_fn(c.query, 10) for c in cases]
    out["mrr@10"] = mrr(ranked, [c.relevant for c in cases])
    return out
