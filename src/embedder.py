"""Stage 2 — encode text chunks into L2-normalized embedding vectors (local, free)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config
from src.chunker import CodeChunk


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return embeddings.astype("float32")


def embed_chunks(chunks: list[CodeChunk]) -> np.ndarray:
    if not chunks:
        return np.empty((0, config.EMBEDDING_DIM), dtype="float32")
    return embed_texts([c.content for c in chunks])


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
