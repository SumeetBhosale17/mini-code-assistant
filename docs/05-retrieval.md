# 05 — Retrieval (Stage 4)

Embed the question with the *same* model, search the index, return the top-k most similar chunks.

**Module:** [`../src/retriever.py`](../src/retriever.py) · **Tests:** [`../tests/unit/test_retriever.py`](../tests/unit/test_retriever.py)

## Key idea
`retrieve` embeds the query, calls `index.search(vec, k)` (returns scores + row-ids), and maps
ids back to `chunks` via the stage-3 row-order join — skipping FAISS's `-1` padding when `k`
exceeds the index size. It returns `RetrievalResult(chunk, score)`, keeping the score for
ranking, thresholds, and citations. It's a pure function: the caller loads the index once and
asks many questions.

## Full deep-dive — [`concepts.md`](./concepts.md)
- **Concept:** Section 5 (Q10)
- **Implementation:** Section 12 (Q34–Q39)
