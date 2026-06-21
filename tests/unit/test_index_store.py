import numpy as np

from src.chunker import CodeChunk
from src.index_store import build_index, load_index, save_index


def _fake(n, dim=4):
    vecs = np.random.rand(n, dim).astype("float32")
    chunks = [CodeChunk(f"code{i}", "f.py", i, "function", f"fn{i}") for i in range(n)]
    return vecs, chunks


def test_build_index_dims():
    vecs, _ = _fake(5)
    index = build_index(vecs)
    assert index.ntotal == 5 and index.d == 4


def test_save_load_roundtrip(tmp_path):
    vecs, chunks = _fake(3)
    save_index(build_index(vecs), chunks, str(tmp_path))
    index, loaded = load_index(str(tmp_path))
    assert index.ntotal == 3
    assert [c.name for c in loaded] == ["fn0", "fn1", "fn2"]  # order preserved
