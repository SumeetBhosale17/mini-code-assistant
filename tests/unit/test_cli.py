from pathlib import Path

from src.cli import run_index


def test_run_index_creates_files(tmp_path):
    (tmp_path / "m.py").write_text("def f(): return 1")
    run_index(str(tmp_path))
    # save_index defaults to config.INDEX_DIR (.rag_index) — assert files exist there,
    # or pass an index_dir if you parametrize. Simplest: check .rag_index was written.
    assert (Path(".rag_index") / "vectors.faiss").exists()
