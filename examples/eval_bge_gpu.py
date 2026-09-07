"""Evaluate rag-forge with a real BGE embedding model on a GPU.

Compares the offline hash embedder against BAAI/bge-small-zh-v1.5 on the
built-in labeled fixture (20 Chinese AI docs × 10 queries), in dense and
hybrid modes — quantifying what a production embedder buys over hashing.

    uv run python examples/eval_bge_gpu.py          # needs CUDA (3090 measured)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_forge.dense import DenseIndex, SentenceTransformerEmbedder
from rag_forge.eval import evaluate, load_cases
from rag_forge.pipeline import HybridPipeline

DATA = Path(__file__).resolve().parent.parent / "data"


def load_corpus() -> dict[str, str]:
    corpus = {}
    for line in (DATA / "docs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            corpus[obj["id"]] = obj["title"] + "\n" + obj["text"]
    return corpus


def run_mode(pipe: HybridPipeline, mode: str, cases) -> dict:
    return evaluate(lambda q, k: [r.doc_id for r in pipe.search(q, top_k=k, mode=mode)],
                    cases)


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "this eval targets a CUDA GPU"
    cases = load_cases(DATA / "queries.jsonl")
    corpus = load_corpus()
    out = {"embedder": "BAAI/bge-small-zh-v1.5", "gpu": torch.cuda.get_device_name(0)}

    t0 = time.time()
    pipe = HybridPipeline(embedder=SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
    pipe.build(corpus)
    out["build_seconds"] = round(time.time() - t0, 1)

    for mode in ("dense", "hybrid"):
        t0 = time.time()
        metrics = run_mode(pipe, mode, cases)
        metrics["seconds"] = round(time.time() - t0, 2)
        out[f"bge_{mode}"] = metrics
        print(f"bge {mode}: recall@1={metrics['recall@1']:.2f} "
              f"recall@3={metrics['recall@3']:.2f} mrr@10={metrics['mrr@10']:.3f}")

    dest = Path("results/bge_eval.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dest}")


if __name__ == "__main__":
    main()
