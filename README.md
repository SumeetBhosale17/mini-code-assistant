# Mini Code Assistant (RAG)

A minimal Retrieval-Augmented Generation assistant that answers questions about a codebase.
Point it at a repo, ask "where is the database connection initialized?", and it retrieves the
relevant code and explains it — instead of hallucinating an answer.

Built as a learning project to understand the RAG pipeline end to end. Deep dives for each
stage live in [`docs/`](./docs) and are written as each stage is built.

## The core problem

You cannot just paste an entire codebase into an LLM prompt:

- **Hard limit** — LLMs have a fixed context window (~128k–200k tokens). A real codebase is
  millions of tokens. It does not fit.
- **Soft limit** — even when content fits, model attention degrades over very long inputs, and
  you pay per token on every query.

RAG's whole job is to select the relevant ~3–5 chunks and send only those.

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

# Ask questions
python -m src.cli --ask "where is the database connection initialized?"
```

## Repo structure

```
src/        one module per pipeline stage
docs/       why/how/when/what deep dive per stage
tests/      unit tests (start with the chunker)
.rag_index/ generated index (gitignored)
```
