# 00 — The Core Problem

Why you can't just paste an entire codebase into an LLM prompt:

- **Hard limit** — LLMs have a fixed context window (~128k–200k tokens). A real codebase is
  millions of tokens; it doesn't fit.
- **Soft limit** — even when it fits, attention degrades over very long inputs, and you pay per
  token on every query.

RAG's whole job is to select the relevant ~3–5 chunks and send only those.

**Full deep-dive:** [`concepts.md`](./concepts.md) → Section 1 (Q1–Q2).
