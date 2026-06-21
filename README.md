# Mini Code Assistant (RAG)

[![CI](https://github.com/SumeetBhosale17/mini-code-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/SumeetBhosale17/mini-code-assistant/actions/workflows/ci.yml)

A minimal Retrieval-Augmented Generation assistant that answers questions about a codebase.
Point it at a repo, ask "where is the database connection initialized?", and it retrieves the
relevant code and explains it — instead of hallucinating an answer.

Built as a learning project to understand the RAG pipeline end to end. Each stage has a short
deep-dive page in [`docs/`](./docs); the full why/how/when/what + Q&A lives in
[`docs/concepts.md`](./docs/concepts.md).

## The core problem

You cannot just paste an entire codebase into an LLM prompt:

- **Hard limit** — LLMs have a fixed context window (~128k–200k tokens). A real codebase is
  millions of tokens. It does not fit.
- **Soft limit** — even when content fits, model attention degrades over very long inputs, and
  you pay per token on every query.

RAG's whole job is to select the relevant ~3–5 chunks and send only those.

→ [`docs/00-core-problem.md`](./docs/00-core-problem.md)

## The pipeline

```
OFFLINE  (run once, when the repo changes)
  repo files ──▶ chunker ──▶ embedder ──▶ FAISS index
                (AST split)  (text→vec)   (store + search)

ONLINE  (every question)
  question ──▶ embedder ──▶ retrieve top-k ──▶ LLM ──▶ answer
              (same model!) (cosine sim)      (reasoning)
```

Two phases. The offline phase runs once — you never re-embed the whole repo per question.
The online phase runs every time someone asks.

→ [`docs/01-pipeline.md`](./docs/01-pipeline.md)

## Stages

**1. Chunking** — split files into semantically coherent units (one function or class per chunk)
using Python's `ast` module, not fixed-size slicing. → [`docs/02-chunking.md`](./docs/02-chunking.md)

**2. Embedding** — map each chunk to a dense vector where semantically similar code lands nearby.
Normalize so cosine similarity reduces to a dot product. → [`docs/03-embeddings.md`](./docs/03-embeddings.md)

**3. Vector store** — store vectors in a FAISS index built for fast nearest-neighbor search,
and persist it to disk. → [`docs/04-vector-store.md`](./docs/04-vector-store.md)

**4. Retrieval** — embed the question with the *same* model, search the index, return the
top-k most similar chunks. → [`docs/05-retrieval.md`](./docs/05-retrieval.md)

**5. LLM reasoning** — assemble a prompt with the retrieved chunks + the question, and ask the
LLM to answer grounded only in that code. → [`docs/06-llm.md`](./docs/06-llm.md)

## How to run

```bash
pip install -r requirements.txt

# Set your free Gemini API key (get one — no credit card — at https://aistudio.google.com/apikey)
# PowerShell:  $env:GEMINI_API_KEY="your_key_here"
# bash:        export GEMINI_API_KEY=your_key_here

# Index a repository (run once)
python -m src.cli --index ./path/to/your/repo

# Interactive session — ask many questions (loads the model + index once)
python -m src.cli

# Or a one-shot question
python -m src.cli --ask "where is the database connection initialized?"
```

## Demo

```text
$ python -m src.cli --ask "How are code files split into chunks?"

Code files are split into chunks by reading the file's text and parsing it into an
Abstract Syntax Tree (AST):
  1. chunk_file reads the file into a string.                     (src/chunker.py:54)
  2. chunk_source parses it into an AST; a SyntaxError yields []. (src/chunker.py:21)
  3. It walks the top-level nodes, keeping function/class defs.   (src/chunker.py:37)
  4. For each, it slices the *raw source* by line number —
     decorators included, comments preserved.                    (src/chunker.py:39)
  5. Each becomes a CodeChunk(content, file_path, line, type, name).

Sources:
    src/chunker.py:54  (0.63)
    src/chunker.py:21  (0.51)
    src/chunker.py:75  (0.51)
```

Bare `python -m src.cli` opens an interactive session that loads the model + index once,
then answers many questions in a loop.

## Evaluation

A tiny retrieval eval checks that each question surfaces the right module in the top-k —
no LLM call, so it's fast and deterministic:

```bash
python -m evals.run_eval
```

Current result: **Recall@5 = 6/6** — every question retrieves its expected file. Add or edit
cases in [`evals/cases.json`](./evals/cases.json).

## Repo structure

```
src/        one module per pipeline stage
docs/       per-stage deep dives + concepts.md
tests/      unit (per-module) + integration (end-to-end) tests
evals/      tiny retrieval eval (recall@k)
.github/    CI: ruff, pyright, pytest
.rag_index/ generated index (gitignored)
```
