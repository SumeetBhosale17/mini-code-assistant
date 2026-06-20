# CLAUDE.md — Mini RAG Code Assistant

## What this project is

A minimal (~300 line) Retrieval-Augmented Generation (RAG) code assistant. It indexes
a code repository, then answers natural-language questions about that code by retrieving
the most relevant chunks and asking an LLM to reason over them. This is a learning project,
not production software.

## How I want you to work with me — READ THIS FIRST

I am building this to **understand** it, not to ship it fast. When helping me:

- **Build ONE stage at a time.** Do not generate the whole pipeline in one shot, even if asked.
- **Explain before you code.** Before writing a module, walk me through it in this order:
  **why** the stage exists, **how** it works, **when** each design choice applies, **what** the
  code will do. (why / how / when / what — this is how I learn.)
- **Make me write it.** Prefer guiding me to the implementation over handing me finished code.
  Give me the structure, the function signatures, the reasoning — and let me fill the body.
  Only write a complete module if I explicitly ask, or if I'm clearly stuck after trying.
- **Be direct and honest.** If my idea is wrong, over-engineered, or scope-creeping, say so plainly.
  Do not validate weak reasoning.
- **Comments explain WHY, not WHAT.** `# normalize so cosine == dot product` is useful;
  `# loop over chunks` is noise.
- **Keep modules small and independently testable.**

## Architecture (the pipeline)

```
OFFLINE  (run once, when the repo changes)
  repo files ──▶ chunker ──▶ embedder ──▶ FAISS index
                (AST split)  (text→vec)   (store + search)

ONLINE  (every question)
  question ──▶ embedder ──▶ retrieve top-k ──▶ LLM ──▶ answer
              (SAME model!) (cosine sim)      (reasoning)
```

The two non-obvious invariants:
1. Query and chunks MUST be embedded with the same model — different models = different
   vector spaces = meaningless similarity scores.
2. Vectors are L2-normalized at index time so FAISS inner-product search equals cosine similarity.

## Build plan — stages map 1:1 to modules and docs

| Stage | Module | Doc | Responsibility |
|-------|--------|-----|----------------|
| 1 | `src/chunker.py` | `docs/02-chunking.md` | Parse files with `ast`, extract function/class chunks |
| 2 | `src/embedder.py` | `docs/03-embeddings.md` | Encode chunks to vectors, normalize |
| 3 | `src/index_store.py` | `docs/04-vector-store.md` | Build/save/load FAISS index + metadata |
| 4 | `src/retriever.py` | `docs/05-retrieval.md` | Embed query, top-k cosine search |
| 5 | `src/llm.py` | `docs/06-llm.md` | Assemble prompt with context, call the LLM (Gemini) |
| 6 | `src/cli.py` | — | Wire `--index` and `--ask` together |

Build them in order. Do not start stage N+1 until stage N runs and I understand it.

## Tech stack

- Python 3.10+
- `sentence-transformers` — embedding model `all-MiniLM-L6-v2` (384-dim, ~90MB, fast)
- `faiss-cpu` — vector index (`IndexFlatIP` for this scale; exact search is fine)
- `google-genai` — LLM, model string `gemini-2.0-flash` (Google Gemini free tier; no budget)
- API key from env var `GEMINI_API_KEY` — never hardcode secrets. Free key (no card): https://aistudio.google.com/apikey

## Conventions

- Type hints on every function.
- `@dataclass` for the `CodeChunk` record (content, file_path, start_line, chunk_type, name).
- Persist index to `.rag_index/` (gitignored): `vectors.faiss` + `metadata.json`.
- Each `docs/0X-….md` is written as we finish its stage — it's the learning record,
  in why/how/when/what form, not API documentation.

## First run

If `src/` is empty, scaffold the directory structure above (empty module files with a
one-line docstring each, plus `requirements.txt` and `.gitignore`). Then stop and wait —
do not implement anything until I say which stage to start.
