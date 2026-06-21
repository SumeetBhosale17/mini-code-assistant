from src.cli import run_index


def test_run_index_creates_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f(): return 1")

    index_dir = tmp_path / "idx"  # isolated — must NOT touch the real .rag_index/
    run_index(str(repo), str(index_dir))

    assert (index_dir / "vectors.faiss").exists()
    assert (index_dir / "metadata.json").exists()
