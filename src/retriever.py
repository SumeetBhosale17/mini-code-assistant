"""Stage 4 — embed a query and return the top-k most similar chunks (cosine)."""

from dataclasses import dataclass

import faiss

from src import config
from src.chunker import CodeChunk
from src.embedder import embed_query


@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


def retrieve(
    question: str,
    index: faiss.Index,
    chunks: list[CodeChunk],
    k: int = config.TOP_K,
    min_score: float = config.SIMILARITY_THRESHOLD,
) -> list[RetrievalResult]:
    """Embed the question (same model), search the index, map row-ids back to chunks."""
    query_vec = embed_query(question)  # shape (dim,)
    scores, ids = index.search(query_vec.reshape(1, -1), k)  # each shape (1, k)
    results = []
    for idx, score in zip(ids[0], scores[0], strict=True):
        if idx < 0:  # FAISS pads with -1 when fewer than k vectors exists
            continue
        if score < min_score:  # below relevance bar -> treat as no match
            continue
        results.append(RetrievalResult(chunks[idx], float(score)))

    return results
