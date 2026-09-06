"""Rank fusion: RRF and weighted normalized-score fusion."""

from __future__ import annotations

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           top_k: int | None = None) -> list[tuple[str, float]]:
    """Fuse several ranked id lists with Reciprocal Rank Fusion.

    RRF is remarkably robust because it ignores raw scores (incomparable
    across retrievers) and uses only ranks: score(d) = Σ 1/(k + rank_i(d)).
    """
    agg: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            agg[doc_id] = agg.get(doc_id, 0.0) + 1.0 / (k + rank)
    out = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return out[:top_k] if top_k else out


def weighted_fusion(score_maps: list[dict[str, float]], weights: list[float],
                    top_k: int | None = None) -> list[tuple[str, float]]:
    """Fuse raw-score maps after min-max normalizing each map to [0, 1]."""
    assert len(score_maps) == len(weights)
    normed: list[dict[str, float]] = []
    for m in score_maps:
        if not m:
            normed.append({})
            continue
        lo, hi = min(m.values()), max(m.values())
        if hi == lo:  # single candidate (or all-equal): the max maps to 1.0
            normed.append({d: 1.0 for d in m})
        else:
            normed.append({d: (s - lo) / (hi - lo) for d, s in m.items()})
    agg: dict[str, float] = {}
    for m, w in zip(normed, weights):
        for d, s in m.items():
            agg[d] = agg.get(d, 0.0) + w * s
    out = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return out[:top_k] if top_k else out
