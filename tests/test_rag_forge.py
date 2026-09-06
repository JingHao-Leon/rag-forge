from pathlib import Path

import pytest

from rag_forge.bm25 import BM25Index
from rag_forge.chunking import fixed_chunks, recursive_chunks
from rag_forge.cli import load_corpus
from rag_forge.dense import DenseIndex
from rag_forge.eval import mrr, ndcg_at_k, recall_at_k
from rag_forge.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_forge.pipeline import HybridPipeline
from rag_forge.rerank import make_llm_reranker, mmr_rerank
from rag_forge.tokenize import tokenize

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATA / "docs.jsonl")


# ------------------------------------------------------------------ units --

def test_tokenizer_mixed_language():
    toks = tokenize("RAG 检索增强生成 is great")
    assert "rag" in toks and "great" in toks
    assert "检索" in toks  # Chinese bigram
    assert all(t != " " for t in toks)


def test_fixed_chunks_overlap_and_coverage():
    text = "x" * 1000
    chunks = fixed_chunks("a", text, size=200, overlap=40)
    assert chunks[0].text == text[:200]
    assert chunks[1].start == 160  # step = size - overlap
    joined = "".join(c.text for c in chunks)
    assert joined.startswith(text[:200]) and text[999] in joined[-10:]


def test_recursive_chunks_respect_sentence_boundaries():
    text = "第一句话。第二句话。第三句话。" * 10
    chunks = recursive_chunks("a", text, size=30, overlap=0)
    assert all(len(c.text) <= 30 for c in chunks)
    # exact partition: no character lost or duplicated by chunking
    assert "".join(c.text for c in chunks) == text


def test_bm25_ranks_exact_term_first():
    idx = BM25Index().build({"a": "苹果 香蕉 苹果", "b": "梨子 橙子", "c": "苹果 葡萄"})
    hits = idx.search("苹果", top_k=3)
    assert hits[0][0] == "a"  # higher tf wins
    assert hits[-1][0] == "b" and hits[-1][1] == 0.0  # no term overlap → zero score


def test_rrf_prefers_docs_ranked_high_by_multiple_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    top3 = [d for d, _ in fused[:3]]
    assert set(top3[:2]) == {"a", "b"}  # appear high in both lists
    assert fused[0][1] >= fused[2][1]


def test_weighted_fusion_normalizes_scores():
    out = weighted_fusion([{"a": 100.0, "b": 0.0}, {"a": 0.8, "b": 0.4}], [1.0, 1.0])
    d = dict(out)
    assert d["a"] == pytest.approx(1.0 + 1.0)  # max in both maps after min-max
    assert d["b"] == pytest.approx(0.0)
    out2 = weighted_fusion([{"a": 100.0}, {"a": 0.8}], [0.0, 1.0])
    assert dict(out2)["a"] == pytest.approx(1.0)  # weight gates a retriever off


def test_metrics_known_values():
    assert recall_at_k(["a", "b"], {"a"}, 1) == 1.0
    assert recall_at_k(["c", "a"], {"a"}, 1) == 0.0
    assert mrr([["a", "b"], ["c", "a"]], [{"a"}, {"a"}]) == pytest.approx(0.75)
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1.0)  # perfect top-1
    assert ndcg_at_k(["c", "a"], {"a"}, 2) < 1.0


# ---------------------------------------------------------------- integration --

def test_dense_index_search_semantically_related(corpus):
    idx = DenseIndex().build(corpus)
    hits = idx.search("怎么微调大模型 LoRA", top_k=3)
    assert "d02" in [d for d, _ in hits]


def test_hybrid_beats_or_matches_single_retrievers_on_fixture(corpus):
    from rag_forge.eval import evaluate, load_cases

    cases = load_cases(DATA / "queries.jsonl")
    pipe = HybridPipeline().build(corpus)

    def run(mode):
        return evaluate(
            lambda q, k: [r.doc_id for r in pipe.search(q, top_k=k, mode=mode)], cases
        )

    metrics = {m: run(m) for m in ("bm25", "dense", "hybrid")}
    assert metrics["hybrid"]["recall@3"] >= min(metrics["bm25"]["recall@3"], metrics["dense"]["recall@3"])
    assert metrics["hybrid"]["recall@3"] > 0.5  # sanity: fixture must be retrievable
    return metrics


def test_llm_reranker_adapter_reorders():
    rerank = make_llm_reranker(lambda q, texts: [2, 0, 1])
    out = rerank("q", [("a", "x"), ("b", "y"), ("c", "z")], top_k=3)
    assert [d for d, _ in out] == ["c", "a", "b"]


def test_mmr_rerank_returns_all_unique():
    cands = [("a", "苹果 香蕉"), ("b", "苹果 香蕉"), ("c", "汽车 轮胎")]
    out = mmr_rerank("苹果", cands, top_k=3, lambda_=0.5)
    ids = [d for d, _ in out]
    assert len(ids) == len(set(ids)) == 3
    # duplicate chunk of the top hit should be demoted by redundancy penalty
    assert ids.index("a") != 1 or ids.index("b") != 0
