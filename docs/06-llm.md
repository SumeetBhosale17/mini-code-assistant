# 06 — LLM Reasoning (Stage 5)

Assemble a prompt with the retrieved chunks + the question, and ask the LLM to answer grounded
only in that code.

**Module:** [`../src/llm.py`](../src/llm.py) · **Tests:** [`../tests/unit/test_llm.py`](../tests/unit/test_llm.py)

## Key idea
`build_prompt` (pure) turns each `RetrievalResult` into a labeled `# file:line (type name)` block
plus the question; `answer` sends it to **Gemini** (`gemini-2.5-flash`, free tier) with a
`SYSTEM_INSTRUCTION` that says: use only the provided code, admit ignorance, cite `file:line`.
That grounding is the main anti-hallucination lever. Empty results short-circuit (no API call).
The API key loads from `.env` (python-dotenv); it never lives in source.

## Full deep-dive — [`concepts.md`](./concepts.md)
- **Concept:** Section 6 (Q11–Q12)
- **Implementation:** Section 13 (Q40–Q45)
