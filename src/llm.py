"""Stage 5 — assemble a grounded prompt and call the LLM (Google Gemini, free tier)."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import config
from src.retriever import RetrievalResult

SYSTEM_INSTRUCTION = (
    "You are a code assistant. Answer the question using ONLY the provided "
    "code context. If the answer is not in the context, say you don't know - "
    "do not guess. Cite the file path and line number for any code you reference."
)


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Create the Gemini client once (reads GEMINI_API_KEY from env / .env)."""
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY not set. Put it in .env or export it before asking."
        )
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Format retrieved chunks + question into one grounded prompt (pure -> testable).

    Each result becomes a labeled block so the model can cite it:
        # <file_path>:<start_line> (<chunk_type> <name>)
        <content>
    """
    blocks = [
        f"# {r.chunk.file_path}:{r.chunk.start_line} "
        f"({r.chunk.chunk_type} {r.chunk.name})\n{r.chunk.content}"
        for r in results
    ]
    context = "\n\n".join(blocks)
    return f"Code context:\n\n{context}\n\nQuestion: {question}"


def answer(question: str, results: list[RetrievalResult]) -> str:
    """Call Gemini with the grounded prompt; return the answer text."""
    if not results:
        return "No relevant code found in the index."

    response = _get_client().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text or "The model returned an empty response."
