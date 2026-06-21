from src.chunker import CodeChunk
from src.embedder import embed_chunks
from src.index_store import build_index
from src.retriever import retrieve


def _chunks():
    return [
        CodeChunk("def add(a, b): return a + b", "math.py", 1, "function", "add"),
        CodeChunk(
            "def connect_to_database(): ...",
            "db.py",
            1,
            "function",
            "connect_to_database",
        ),
    ]


def test_returns_most_relevant_first():
    chunks = _chunks()
    index = build_index(embed_chunks(chunks))
    results = retrieve("how do I open a database connection?", index, chunks, k=2)
    assert results[0].chunk.name == "connect_to_database"


def test_handles_k_larger_than_index():
    chunks = _chunks()[:1]  # only one chunk
    index = build_index(embed_chunks(chunks))
    # min_score=0.0 isolates the padding logic from the threshold
    results = retrieve("anything", index, chunks, k=5, min_score=0.0)
    assert len(results) == 1  # the -1 padding must be skipped
