"""Dense retrieval with a pluggable embedder.

The default ``HashEmbedder`` needs zero downloads: hashed character n-grams
with sublinear TF and L2 normalization — a real (if shallow) semantic signal
that keeps the whole library offline-runnable. Swap in
``SentenceTransformerEmbedder`` (lazy import) for production-grade embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .tokenize import tokenize

_WORD = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def _ngrams(text: str, n: int = 3) -> list[str]:
    toks = tokenize(text)
    grams: list[str] = []
    for t in toks:
        if len(t) < n:
            grams.append(t)
        else:
            grams.extend(t[i:i + n] for i in range(len(t) - n + 1))
    return grams


class HashEmbedder:
    """Hashing-trick embedder: dim-dimensional, sublinear-TF, L2-normalized."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, gram: str) -> int:
        # feature hashing (not security): blake2b is fast and well-distributed
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tf = Counter(self._bucket(g) for g in _ngrams(text))
        for b, f in tf.items():
            v[b] = 1.0 + math.log(f)  # sublinear TF
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Optional production embedder — used only if sentence-transformers is installed."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy, heavy dep
        self._m = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        v = self._m.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)


class DenseIndex:
    def __init__(self, embedder: object | None = None):
        self.embedder = embedder or HashEmbedder()
        self.doc_ids: list[str] = []
        self._vecs: np.ndarray | None = None

    def build(self, corpus: dict[str, str]) -> "DenseIndex":
        self.doc_ids = list(corpus)
        self._vecs = self.embedder.embed_batch([corpus[d] for d in self.doc_ids])
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        assert self._vecs is not None, "call build() first"
        q = self.embedder.embed(query)[None, :]
        # embeddings are L2-normalized → cosine similarity is a single matmul
        sims = (self._vecs @ q.T).squeeze(1)
        order = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
