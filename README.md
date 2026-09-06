# rag-forge · 混合检索 RAG 引擎

**BM25 × 稠密向量双路检索、RRF 融合、MMR 多样性重排、标注数据集评测——纯 Python 零模型下载即可运行，一行命令换装 sentence-transformers 生产级向量。**

[![tests](https://img.shields.io/badge/tests-15%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 为什么是"混合"检索？

BM25 抓得住精确术语（型号、专名、代码），稠密向量懂语义改写（"胡编乱造" ≈ "幻觉"），两者错误模式互补。融合用 **RRF（Reciprocal Rank Fusion）**——只看排名不看原始分，天然免疫两路打分量纲不可比的问题。

## 📊 内置标注数据集实测（20 篇中文 AI 文档 × 10 个查询）

| 模式 | recall@1 | recall@3 | recall@5 | MRR@10 | nDCG@5 |
|---|---|---|---|---|---|
| BM25 only | 0.70 | **1.00** | 1.00 | **0.95** | 0.955 |
| Dense only（哈希嵌入） | 0.40 | 0.80 | 0.90 | 0.70 | 0.736 |
| **Hybrid（RRF）** | 0.50 | 0.90 | **1.00** | 0.82 | 0.857 |

> 一个诚实的观察：哈希嵌入是纯词法级 fallback（无模型下载），语义能力弱于 BM25 情有可原；但 hybrid 仍把它的 recall@5 从 0.90 拉到 **1.00**、MRR +17%——**融合的收益恰恰在单路最弱时最大**。换装真实 Embedding 模型后（见下），稠密路变强，hybrid 上限继续抬升。复现：`uv run python -m rag_forge.cli eval --corpus data/docs.jsonl --queries data/queries.jsonl`

## ✨ 核心设计

- **中英混合分词**：英文小写化 + 中文单字/二元组，无词典依赖（二元组让 BM25 在无分词器时依然能命中「检索增强」）
- **分隔符对齐分块**：按 `\n\n → \n → 。 → ； → 空格` 逐级切分，**500 次随机 fuzz 验证精确分区**（`"".join(chunks) == 原文`，不丢一个字符）
- **BM25Okapi**：Robertson 2-epsilon IDF（恒非负），k1/b 可调
- **RRF + 加权融合**双方案，单元素/全同分等边界均有测试覆盖
- **MMR 重排**：λ 平衡相关性与多样性，治"五条结果一个味"
- **LLM 重排适配器**：`make_llm_reranker(any_llm)` 注入你自己的模型调用，库本身零 LLM 依赖
- **评测三件套**：Recall@k / MRR / nDCG@k，标注格式就是 jsonl，3 分钟接上自己的数据

## 🚀 快速开始

```bash
uv sync   # 仅 numpy + pytest

# 检索
uv run python -m rag_forge.cli search --corpus data/docs.jsonl \
    --query "怎么在一张显卡上微调大模型" --top-k 3

# 评测（bm25 / dense / hybrid 三模式对比）
uv run python -m rag_forge.cli eval --corpus data/docs.jsonl --queries data/queries.jsonl --mode hybrid
```

作为库使用：

```python
from rag_forge.pipeline import HybridPipeline

pipe = HybridPipeline(chunk_size=300).build({"d1": "文档内容...", "d2": "..."})
results = pipe.search("微调大模型需要多少显存", top_k=3, mode="hybrid")
for r in results:
    print(r.doc_id, r.score, f"bm25#{r.bm25_rank} dense#{r.dense_rank}")
```

换装生产级向量模型（可选）：

```python
from rag_forge.dense import DenseIndex, SentenceTransformerEmbedder
from rag_forge.pipeline import HybridPipeline

pipe = HybridPipeline()
pipe._dense = DenseIndex(SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
pipe.build(corpus)  # 其余逻辑完全不变
```

## 📁 结构

```
rag_forge/
├── tokenize.py    # 中英混合分词（中文 unigram + bigram）
├── chunking.py    # fixed / recursive 分块，精确分区不变量（fuzz 验证）
├── bm25.py        # BM25Okapi 稀疏索引
├── dense.py       # 哈希嵌入（零依赖）+ sentence-transformers 适配
├── fusion.py      # RRF / 加权 min-max 融合
├── rerank.py      # MMR 多样性重排 + LLM 重排适配器
├── pipeline.py    # HybridPipeline：chunk → 双路索引 → 融合 → 重排
├── eval.py        # Recall@k / MRR / nDCG@k
└── cli.py         # search / eval 命令行
data/              # 20 篇标注文档 + 10 个查询（中文 AI 主题）
tests/             # 15 个测试 + 500-trial fuzz
```

## 🧪 测试

```bash
uv run pytest                              # 15 passed
uv run pytest tests/test_fuzz_chunking.py --trials 2000   # 加大 fuzz 强度
```

## Roadmap

- [ ] HyDE 查询改写（LLM 生成假设文档再检索）
- [ ] 块级评测（当前为文档级标注）
- [ ] FAISS/HNSW 后端适配（大规模库）

## License

MIT
