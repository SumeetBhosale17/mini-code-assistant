# 04 — Vector Store (Stage 3)

Store the vectors in a FAISS index built for fast nearest-neighbor search, and persist it to disk.

**Module:** [`../src/index_store.py`](../src/index_store.py) · **Tests:** [`../tests/unit/test_index_store.py`](../tests/unit/test_index_store.py)

## Key idea
`IndexFlatIP` does exact inner-product search (= cosine on normalized vectors). Two files persist
to `.rag_index/`: `vectors.faiss` (vectors only) and `metadata.json` (the `CodeChunk`s). They're
joined by **row order** — `chunks[i]` describes vector row `i` — the invariant of this stage.
Persisting splits the OFFLINE (build) phase from the ONLINE (load) phase, so questions never
re-embed the repo.

## Full deep-dive — [`concepts.md`](./concepts.md)
- **Concept:** Section 4 (Q8–Q9)
- **Implementation:** Section 11 (Q28–Q33)
