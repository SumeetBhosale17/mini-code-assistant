# 01 — The Pipeline

Two phases that never overlap:

```
OFFLINE  (run once, when the repo changes)
  repo files ──▶ chunker ──▶ embedder ──▶ FAISS index
                (AST split)  (text→vec)   (store + search)

ONLINE  (every question)
  question ──▶ embedder ──▶ retrieve top-k ──▶ LLM ──▶ answer
              (SAME model!) (cosine sim)      (reasoning)
```

Two invariants hold across both phases:
1. Query and chunks are embedded with the **same model** (different models = different vector spaces).
2. Vectors are **L2-normalized**, so FAISS inner-product search equals cosine similarity.

**Per-stage pages:** [02](./02-chunking.md) · [03](./03-embeddings.md) · [04](./04-vector-store.md) ·
[05](./05-retrieval.md) · [06](./06-llm.md). Full Q&A in [`concepts.md`](./concepts.md).
