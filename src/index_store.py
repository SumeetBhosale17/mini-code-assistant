"""Stage 3 — build, save, and load the FAISS index plus chunk metadata."""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src import config
from src.chunker import CodeChunk


def _paths(index_dir: str) -> tuple[Path, Path]:
    """The two files of a persisted index = derived once so save/load always agree."""
    base = Path(index_dir)
    return base / config.INDEX_FILE, base / config.METADATA_FILE


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an exact cosine-search index from L2-normalized vectors."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index,
    chunks: list[CodeChunk],
    index_dir: str = config.INDEX_DIR,
) -> None:
    """Persist index + chunk metadata to disk, in matching row order (the join key)."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index_path, meta_path = _paths(index_dir)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


def load_index(
    index_dir: str = config.INDEX_DIR,
) -> tuple[faiss.Index, list[CodeChunk]]:
    """Load index + chunks back - the first step of every ONLINE (query) session."""
    index_path, meta_path = _paths(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"No index at {index_path}. Build it first (--index).")
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        chunks = [CodeChunk(**d) for d in json.load(f)]
    return index, chunks
