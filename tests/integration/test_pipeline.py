"""End-to-end: chunk -> embed -> index -> save -> load -> retrieve (no LLM call)."""

from src.chunker import chunk_repo
from src.embedder import embed_chunks
from src.index_store import build_index, load_index, save_index
from src.llm import answer
from src.retriever import retrieve


def _build_repo(tmp_path):
    (tmp_path / "db.py").write_text(
        "def connect_to_database(host):\n"
        '    """Open a connection to the Postgres database."""\n'
        "    return f'connected to {host}'\n"
    )
    (tmp_path / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    index_dir = str(tmp_path / ".rag_index")
    chunks = chunk_repo(str(tmp_path))
    save_index(build_index(embed_chunks(chunks)), chunks, index_dir)
    return index_dir


def test_pipeline_finds_relevant_code(tmp_path):
    index_dir = _build_repo(tmp_path)
    index, chunks = load_index(index_dir)
    results = retrieve("how do I open a database connection?", index, chunks)
    assert results and results[0].chunk.name == "connect_to_database"


def test_threshold_filters_and_answer_short_circuits(tmp_path):
    index_dir = _build_repo(tmp_path)
    index, chunks = load_index(index_dir)
    results = retrieve(
        "anything", index, chunks, min_score=0.99
    )  # force below-threshold
    assert results == []
    assert answer("anything", results) == "No relevant code found."  # no API call
