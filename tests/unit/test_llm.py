from src.chunker import CodeChunk
from src.llm import build_prompt
from src.retriever import RetrievalResult


def _results():
    chunk = CodeChunk("def connect_db(): ...", "db.py", 42, "function", "connect_db")
    return [RetrievalResult(chunk, 0.81)]


def test_prompt_includes_question_and_context():
    prompt = build_prompt("how do I connect?", _results())
    assert "how do I connect?" in prompt
    assert "def connect_db" in prompt
    assert "db.py" in prompt and "42" in prompt


def test_empty_results_message():
    from src.llm import answer

    assert answer("anything", []) == "No relevant code found."
