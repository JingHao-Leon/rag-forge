"""CLI: build a hybrid index over a JSONL corpus and run queries / evaluation.

Usage:
    python -m rag_forge.cli search --corpus data/docs.jsonl --query "怎么减少幻觉"
    python -m rag_forge.cli eval --corpus data/docs.jsonl --queries data/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_corpus(path: str | Path) -> dict[str, str]:
    corpus = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            corpus[obj["id"]] = obj["title"] + "\n" + obj["text"]
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag-forge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="run a hybrid search")
    s.add_argument("--corpus", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")

    e = sub.add_parser("eval", help="evaluate retrieval quality on labeled queries")
    e.add_argument("--corpus", required=True)
    e.add_argument("--queries", required=True)
    e.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")

    args = ap.parse_args()

    from .eval import evaluate, load_cases
    from .pipeline import HybridPipeline

    corpus = load_corpus(args.corpus)
    pipe = HybridPipeline().build(corpus)

    if args.cmd == "search":
        for r in pipe.search(args.query, top_k=args.top_k, mode=args.mode):
            title = corpus.get(r.doc_id, "").split("\n", 1)[0]
            print(f"{r.score:.4f}  {r.doc_id}  bm25#{r.bm25_rank} dense#{r.dense_rank}  {title}")
    else:
        cases = load_cases(args.queries)
        metrics = evaluate(lambda q, k: [r.doc_id for r in pipe.search(q, top_k=k, mode=args.mode)], cases)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
