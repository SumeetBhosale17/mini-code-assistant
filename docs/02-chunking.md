# 02 — Chunking (Stage 1)

Split source files into semantically coherent units — one function or class per chunk — using
Python's `ast` module, not fixed-size slicing.

**Module:** [`../src/chunker.py`](../src/chunker.py) · **Tests:** [`../tests/unit/test_chunker.py`](../tests/unit/test_chunker.py)

## Key idea
`ast.parse` locates each top-level function/class by line number; the chunk *content* is sliced
from the **raw source** (so comments/formatting survive — `ast.unparse` would discard them).
Decorators are included; a class is one chunk; unparseable files are skipped. The `CodeChunk`
record carries `content`, `file_path`, `start_line`, `chunk_type`, `name` — the citation metadata
every later stage relies on.

## Full deep-dive — [`concepts.md`](./concepts.md)
- **Concept:** Section 2 (Q3–Q4)
- **Implementation:** Section 9 (Q16–Q21)
