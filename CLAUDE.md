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
- **Keep `CLAUDE.md` current.** After every completed stage or new decision, update this
  file so it always reflects the as-built project.

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

- Python 3.13+ (pinned via `requires-python` in `pyproject.toml`)
- `sentence-transformers` — embedding model `all-MiniLM-L6-v2` (384-dim, ~90MB, fast)
- `faiss-cpu` — vector index (`IndexFlatIP` for this scale; exact search is fine)
- `google-genai` — LLM, model string `gemini-2.0-flash` (Google Gemini free tier; no budget)
- API key from env var `GEMINI_API_KEY` — never hardcode secrets. Free key (no card): https://aistudio.google.com/apikey

## Conventions

- Type hints on every function.
- `@dataclass` for the `CodeChunk` record (content, file_path, start_line, chunk_type, name).
- Persist index to `.rag_index/` (gitignored): `vectors.faiss` + `metadata.json`.
- Learning deep-dives live in `docs/concepts.md` (one section per stage, in why/how/when/what
  form, plus running Q&A) — not separate `docs/0X-….md` files. Append the stage's Q&A there
  when the stage is done.

## Progress & decisions (as built — keep current)

**Stage status:** ✅ Stage 1 (chunker) + ✅ Stage 2 (embedder) + ✅ Stage 3 (index store) complete.
- Stage 1: `src/chunker.py`, `src/config.py`, `tests/unit/test_chunker.py`, deep-dive §9.
- Stage 2: `src/embedder.py` (sentence-transformers `all-MiniLM-L6-v2`, L2-normalized,
  cached model), `tests/unit/test_embedder.py`, deep-dive §10. `config.EMBEDDING_MODEL`
  + `EMBEDDING_DIM` now live — both invariants implemented here.
- Stage 3: `src/index_store.py` (FAISS `IndexFlatIP`, persisted to `.rag_index/` as
  `vectors.faiss` + `metadata.json`, joined by row order via a shared `_paths` helper),
  `tests/unit/test_index_store.py`, deep-dive §11. `config.INDEX_DIR/INDEX_FILE/METADATA_FILE`
  now live.

**Next: stage 4 (retriever — `index.search` + top-k).**

**Decisions made along the way:**
- **Deps:** `requirements.txt` (runtime) + `requirements-dev.txt` (dev). `uv` is *not* the
  dependency manager — it's only a task-runner invoked by the pyright pre-commit hook
  (`uv run pyright`), so `uv.lock` is an incidental artifact and is gitignored.
- **Tooling:** `ruff` (lint) + `ruff format` + `pyright` (types) + `pytest`, wired via
  `pre-commit`. Tool config lives in `pyproject.toml` (config only — no build-system/packaging).
- **Tests:** `tests/unit/` now; `tests/integration/` later for the full pipeline. Add
  `importmode = "importlib"` to the pytest config when integration tests arrive.
- **Git:** branch per stage, **merge commits (no squash)**, one PR per stage. Always
  `git checkout main && git pull` before branching for the next stage.
- **Docs:** learning Q&A doc renamed `interview-qa.md` → `docs/concepts.md` (public repo).
  ⚠️ `README.md` still links to non-existent `docs/0X-….md` files — reconcile later (point
  them at `concepts.md` or create the files).

## First run

If `src/` is empty, scaffold the directory structure above (empty module files with a
one-line docstring each, plus `requirements.txt` and `.gitignore`). Then stop and wait —
do not implement anything until I say which stage to start.
