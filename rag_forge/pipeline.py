"""The hybrid retrieval pipeline: chunk → index (BM25 + dense) → fuse → rerank."""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25Index
from .chunking import Chunk, recursive_chunks
from .dense import DenseIndex
from .fusion import reciprocal_rank_fusion, weighted_fusion


@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass
class HybridPipeline:
    chunk_size: int = 300
    chunk_overlap: int = 40
    rrf_k: int = 60
    candidates_per_retriever: int = 20
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    rerank_fn: object | None = None  # optional callable(query, candidates) -> [(doc_id, score)]
    embedder: object | None = None   # pluggable dense embedder (default: offline hash)
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _bm25: BM25Index | None = field(default=None, repr=False)
    _dense: DenseIndex | None = field(default=None, repr=False)

    def build(self, corpus: dict[str, str]) -> "HybridPipeline":
        self._chunks = []
        for doc_id, text in corpus.items():
            self._chunks.extend(recursive_chunks(doc_id, text, self.chunk_size, self.chunk_overlap))
        corpus_chunks = {f"{c.doc_id}::{c.start}": c.text for c in self._chunks}
        self._bm25 = BM25Index().build(corpus_chunks)
        self._dense = DenseIndex(self.embedder).build(corpus_chunks)
        return self

    def _parse(self, chunk_key: str) -> str:
        return chunk_key.split("::", 1)[0]

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[RetrievalResult]:
        """mode: bm25 | dense | hybrid (RRF). Rerank/MMR applies to all modes."""
        k = self.candidates_per_retriever
        if mode in ("bm25", "hybrid"):
            bm25_hits = self._bm25.search(query, top_k=k)
        if mode in ("dense", "hybrid"):
            dense_hits = self._dense.search(query, top_k=k)

        if mode == "bm25":
            fused = [(self._parse(d), s) for d, s in bm25_hits]
            ranks = {"bm25": {self._parse(d): r for r, (d, _) in enumerate(bm25_hits, 1)}}
        elif mode == "dense":
            fused = [(self._parse(d), s) for d, s in dense_hits]
            ranks = {"dense": {self._parse(d): r for r, (d, _) in enumerate(dense_hits, 1)}}
        else:
            rrf = reciprocal_rank_fusion([[d for d, _ in bm25_hits], [d for d, _ in dense_hits]],
                                         k=self.rrf_k)
            # chunk-level RRF → collapse to doc level by max score
            best: dict[str, float] = {}
            for chunk_key, s in rrf:
                doc = self._parse(chunk_key)
                best[doc] = max(best.get(doc, 0.0), s)
            fused = sorted(best.items(), key=lambda x: x[1], reverse=True)
            ranks = {"bm25": {self._parse(d): r for r, (d, _) in enumerate(bm25_hits, 1)},
                     "dense": {self._parse(d): r for r, (d, _) in enumerate(dense_hits, 1)}}

        if self.rerank_fn is not None and fused:
            chunk_text = {c.doc_id: c.text for c in self._chunks}
            fused = self.rerank_fn(query, [(d, chunk_text.get(d, "")) for d, _ in fused])[:top_k]

        results = [RetrievalResult(doc_id=d, score=s,
                                   bm25_rank=ranks.get("bm25", {}).get(d),
                                   dense_rank=ranks.get("dense", {}).get(d))
                   for d, s in fused[:top_k]]
        return results
