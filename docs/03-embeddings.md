# 03 — Embeddings (Stage 2)

Map each chunk to a dense vector where semantically similar code lands nearby.

**Module:** [`../src/embedder.py`](../src/embedder.py) · **Tests:** [`../tests/unit/test_embedder.py`](../tests/unit/test_embedder.py)

## Key idea
`sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) runs locally on CPU — no API, no cost.
Vectors are **L2-normalized** (`normalize_embeddings=True`) so inner product equals cosine
(**invariant #2**). `embed_chunks` (indexing) and `embed_query` (querying) both funnel through
`embed_texts`, which structurally enforces the **same model** for both (**invariant #1**).

## Full deep-dive — [`concepts.md`](./concepts.md)
- **Concept:** Section 3 (Q5–Q7)
- **Implementation:** Section 10 (Q22–Q27)
