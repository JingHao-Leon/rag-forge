"""BM25 (Okapi) sparse index — the half of hybrid retrieval that catches exact terms."""

from __future__ import annotations

import math
from collections import Counter

from .tokenize import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._N = 0
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        tf = Counter(toks)
        self.doc_ids.append(doc_id)
        self._tf.append(tf)
        self._len.append(len(toks))
        for t in tf:
            self._df[t] += 1
        self._N += 1
        self._avgdl = sum(self._len) / self._N

    def build(self, corpus: dict[str, str]) -> "BM25Index":
        for doc_id, text in corpus.items():
            self.add(doc_id, text)
        return self

    def idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        # Robertson's 2-epsilon formulation, always non-negative
        return math.log((self._N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        tf, dl = self._tf[doc_idx], self._len[doc_idx]
        s = 0.0
        for t in set(tokenize(query)):
            if t not in tf:
                continue
            idf = self.idf(t)
            f = tf[t]
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return s

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = [(self.doc_ids[i], self.score(query, i)) for i in range(self._N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
