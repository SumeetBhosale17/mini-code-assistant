# RAG Code Assistant — Concepts & Deep Dives

> Living document. Each concept includes: the question, the answer, and the *justification* (why it matters).

---

## Section 1: The Core Problem

### Q1. Why can't you just paste an entire codebase into an LLM prompt?

**Answer:**
Two reasons — hard and soft.

**Hard limit:** LLMs have a context window (maximum tokens they can process). GPT-4 is ~128k tokens, Claude is ~200k. A real codebase (e.g. Django, React) contains millions of tokens. It literally won't fit.

**Soft limit:** Even if a codebase fit, LLM attention quality degrades over very long contexts. The model struggles to extract the needle from a 100k-token haystack. And you pay per input token — sending 100k tokens per question is expensive.

**Justification:** This is *why RAG exists*. RAG's entire purpose is to select the relevant ~3-5% of code and send only that.

---

### Q2. What is RAG? Define it precisely.

**Answer:**
RAG = Retrieval-Augmented Generation.

It's a pattern where, instead of relying on an LLM's parametric memory (what it learned during training), you:
1. **Retrieve** relevant documents/chunks at query time from an external store
2. **Augment** the LLM prompt with those retrieved chunks
3. **Generate** an answer grounded in that retrieved context

**Justification:** Without retrieval, an LLM answering code questions is guessing from training data — it might hallucinate function names, APIs, or logic that doesn't exist in your repo. With retrieval, it reasons over actual code.

---

## Section 2: Chunking

### Q3. What is a chunk, and why do we chunk at all?

**Answer:**
A chunk is a small, semantically coherent unit of code — typically one function, one class, or one logical block.

We chunk because:
- Embedding quality degrades on very long text. A 5,000-line file produces a vague, averaged-out vector that doesn't represent any specific functionality well.
- We need fine-grained retrieval. A user asking "how is the DB connection closed?" needs the `close_connection()` function, not the entire 500-line `database.py` file.
- Token efficiency: we retrieve 3–5 chunks (~500 tokens total) instead of the whole file (~10,000 tokens).

**Justification:** Chunk granularity is the biggest tuning knob in RAG quality. Too coarse = irrelevant code retrieved. Too fine = chunks lose context (a 3-line chunk has no useful embedding).

---

### Q4. What is the difference between fixed-size chunking and semantic/AST chunking?

**Answer:**

| Strategy | How | Problem |
|---|---|---|
| Fixed-size | Split every N tokens/lines | Cuts functions in half. A chunk might contain only the signature without the body — the embedding is meaningless. |
| Sliding window | Fixed size + overlap | Better — overlapping chunks reduce cut-off issues. But still arbitrary. |
| AST-based | Parse the code, extract complete functions/classes | Chunks are semantically coherent. The embedding captures "what this code does" accurately. |

**For a code assistant, AST-based is strongly preferred.** Python's `ast` module gives exact start/end line numbers for every function and class.

**Justification:** You can think of chunking as lossy compression. AST chunking preserves the most semantically important unit (the function) — fixed-size chunking is blind to semantics.

---

## Section 3: Embeddings

### Q5. What is an embedding, mathematically?

**Answer:**
An embedding is a function $f: \text{text} \rightarrow \mathbb{R}^d$ — it maps a piece of text to a dense vector in a $d$-dimensional space (typically $d = 384$ or $d = 1536$).

The model is trained so that **semantically similar text maps to nearby vectors**.

Example:
- `"connect to the database"` → $\mathbf{v}_1 \in \mathbb{R}^{384}$
- `"establish DB connection"` → $\mathbf{v}_2 \in \mathbb{R}^{384}$
- `"bake chocolate cake"` → $\mathbf{v}_3 \in \mathbb{R}^{384}$

$\cos(\mathbf{v}_1, \mathbf{v}_2) \approx 0.92$ (similar)
$\cos(\mathbf{v}_1, \mathbf{v}_3) \approx 0.11$ (unrelated)

**Justification:** The entire retrieval system rests on this property. If the embeddings don't capture semantic similarity, the whole pipeline fails — even with a perfect LLM.

---

### Q6. Why cosine similarity instead of Euclidean distance?

**Answer:**
Cosine similarity measures the **angle** between two vectors, ignoring their magnitude.

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

We prefer it because text embeddings vary in magnitude based on length — a long docstring and a short one-liner produce vectors of very different norms. Euclidean distance would penalize length differences even when the content is semantically identical.

**After L2 normalization** (making every vector unit-length), cosine similarity = inner product. This is why we normalize vectors before storing in FAISS.

**Justification:** This is the same reason your NLA course covers condition number — the *scale* of a vector shouldn't dictate its relationship to another. Normalizing removes a confounding factor.

---

### Q7. Why must you use the same embedding model for indexing and querying?

**Answer:**
Each embedding model defines its own vector space. The position of `"database connection"` in model A's 384-dim space has no relationship to its position in model B's 768-dim space.

If you index with model A and query with model B:
- Dimensions may not even match (384 vs 768)
- Even if dimensions match, the geometry is completely different
- Cosine similarities are random noise, not semantic scores

**Analogy:** It's like encoding a message in Morse code and trying to decode it as ASCII. Same signal, wrong codebook = garbage.

**Justification:** This is a common production bug. Teams upgrade embedding models and forget to re-index their entire corpus. The result is subtly degraded retrieval that's hard to debug.

---

## Section 4: Vector Database

### Q8. What is a vector database? Why not use a regular SQL/NoSQL DB?

**Answer:**
A vector database is optimized for **nearest-neighbor search in high-dimensional space**.

Regular databases (Postgres, MongoDB) excel at exact lookup: `WHERE id = 42` or `WHERE name = 'Alice'`. They have no concept of "find the 5 rows most similar to this query vector."

You could compute cosine similarity in SQL:
```sql
SELECT id, (vector <=> query_vec) AS dist FROM chunks ORDER BY dist LIMIT 5;
```
pgvector does this. But naive implementation is O(n) — you compare the query against every row.

**FAISS, Pinecone, Weaviate** use Approximate Nearest Neighbor (ANN) algorithms (like HNSW or IVF) that reduce this to sub-linear time using index structures, at the cost of occasionally missing the very best match.

**Justification:** For our mini assistant (thousands of chunks), exact search is fast enough. For production systems (millions of docs), ANN is essential.

---

### Q9. What is FAISS and what does `IndexFlatIP` mean?

**Answer:**
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search on dense vectors.

`IndexFlatIP`:
- `Flat` = brute-force exact search (no approximation)
- `IP` = Inner Product similarity

On L2-normalized (unit) vectors: Inner Product = Cosine Similarity.

Other FAISS index types:
| Index | Speed | Accuracy | Memory |
|---|---|---|---|
| `IndexFlatIP` | O(n), slowest | Exact | High |
| `IndexIVFFlat` | O(n/n_cells) | ~Exact | Medium |
| `IndexHNSWFlat` | O(log n) | Very good | High |
| `IndexIVFPQ` | O(log n) | Good | Low |

For a mini assistant: `IndexFlatIP` is perfect.

---

## Section 5: Retrieval

### Q10. What is top-k retrieval? How do you choose k?

**Answer:**
Top-k retrieval = return the k chunks with the highest cosine similarity to the query vector.

Choosing k involves a tradeoff:
- **Too small (k=1-2):** Risk missing the key chunk. If the answer spans two functions, you might retrieve only one.
- **Too large (k=10-20):** Irrelevant chunks are included. The LLM prompt grows, costs increase, and noise can degrade answer quality.

**Typical starting point: k=5.** In practice, you tune based on observed retrieval quality.

**Advanced:** Use a **similarity threshold** instead of (or in addition to) fixed k. Only include chunks with similarity > 0.6. If nothing is above threshold, report "no relevant code found" rather than forcing an answer from poor context.

---

## Section 6: LLM Reasoning

### Q11. What is the LLM's actual job in the RAG pipeline?

**Answer:**
The LLM doesn't retrieve — retrieval is done. The LLM:

1. **Reads** the retrieved chunks (code it has never seen before)
2. **Understands** what the code does in context
3. **Reasons** across multiple chunks (e.g., "the DB is initialized in `db.py:42` and that object is passed into `auth.py:88`")
4. **Generates** a coherent natural-language explanation with citations

Without the LLM, you've done semantic search — useful, but the user still has to read raw code. The LLM turns relevant code into *explanation*.

**Justification:** This is why RAG is stronger than pure semantic search for Q&A. Retrieval finds the needle; LLM explains what the needle is.

---

### Q12. What is hallucination in this context, and how does RAG reduce it?

**Answer:**
Hallucination = the LLM generating plausible but factually incorrect content. In code assistants, this means inventing function names, APIs, or logic that don't exist in your codebase.

**Without RAG:** The LLM answers from parametric memory — what it learned during training. For your private codebase, it has no training data. It invents plausible-sounding code.

**With RAG:** The prompt contains actual code from your repo. The LLM is grounded — it should only cite and explain what's in the context. The system prompt explicitly says: *"do not guess if the answer is not in the provided code."*

**RAG doesn't eliminate hallucination** — the LLM can still misinterpret retrieved code, or if retrieval fails (wrong chunks), it may still guess. It significantly reduces hallucination by providing a factual anchor.

---

## Section 7: Failure Modes

### Q13. List the key failure modes in a RAG pipeline and how to detect/fix each.

| Failure Mode | Symptom | Cause | Fix |
|---|---|---|---|
| Retrieval returns wrong chunks | Answer is about unrelated code | Bad chunking, wrong embedding model, query too vague | Better chunking strategy, re-embed, rephrase query |
| Retrieval returns nothing useful | Low similarity scores (<0.4) | Query is out-of-domain, code not indexed | Add similarity threshold, expand index |
| LLM ignores retrieved context | Answers from general knowledge | System prompt too weak | Stronger instruction: "only use the provided code" |
| Chunk cut off mid-function | Incomplete logic in answer | MAX_CHUNK_CHARS too low or chunking at wrong boundaries | Increase limit or fix chunking strategy |
| Stale index | Changes to codebase not reflected | Index not rebuilt after code edits | Trigger re-index on file change (inotify/watchdog) |
| Mismatched embedding models | Garbage retrieval after model upgrade | Model changed, index not rebuilt | Always re-index when changing embedding model |

---

## Section 8: System Design Extensions

### Q14. How would you scale this to a 1M-file codebase?

**Answer:**
Several changes needed:

1. **ANN index**: Replace `IndexFlatIP` with `IndexHNSWFlat` or `IndexIVFPQ` in FAISS. Or use a managed vector DB (Pinecone, Weaviate, Qdrant).
2. **Incremental indexing**: Don't re-index the whole corpus on every code change. Track file modification timestamps, only re-embed changed files.
3. **Distributed embedding**: Run embedding on GPU cluster or use a hosted embedding API (OpenAI `text-embedding-3-small`, Cohere).
4. **Metadata filtering**: Before vector search, filter by language, directory, or file. Reduces search space dramatically.
5. **Reranking**: After top-k retrieval, use a cross-encoder model to re-score chunks more accurately. More expensive but higher precision.

---

### Q15. What is the difference between this RAG system and something like Claude Code or Cursor?

**Answer:**
Our mini assistant: retrieval + explanation. **Passive** — it reads and explains.

Claude Code / Cursor: **Active agents** with additional capabilities:
- **Tool use**: Can execute shell commands, run tests, read/write files
- **Multi-turn planning**: Decomposes a task into steps and executes them iteratively
- **Feedback loops**: Runs code, reads the error, revises the code, runs again
- **Context management**: Maintains conversation history + relevant code across multiple turns
- **Diff-based editing**: Proposes file edits as diffs, not just explanations

The RAG pipeline is the *retrieval backbone* of these systems. Claude Code uses similar semantic search internally but wraps it in an agentic loop that can act on results.

**This is the natural next step:** take our pipeline, add file-write tools, shell execution, and a loop that feeds LLM output back as new context.

---

## Section 9: Chunker — Implementation Deep Dive (stage 1)

> Builds on Section 2. Section 2 is *why* AST chunking; this is *how* `src/chunker.py` actually does it.

### Q16. Mechanically, how does the chunker turn a file into chunks?

**Answer:**
1. `tree = ast.parse(source)` produces a `Module` node whose `.body` is the list of top-level statements.
2. Iterate `tree.body`; keep nodes that are `ast.FunctionDef`, `ast.AsyncFunctionDef`, or `ast.ClassDef`.
3. Each such node exposes `.name`, `.lineno` (1-based start), `.end_lineno` (1-based end, Python ≥3.8), and `.decorator_list`.
4. Slice the chunk out of the *original text*: `source.splitlines()[start-1 : end]` (the `-1` converts a 1-based line number to a 0-based index).
5. Wrap it in a `CodeChunk(content, file_path, start_line, chunk_type, name)`.

**Justification:** The AST gives exact, reliable boundaries — no regex, no guessing where a function ends. `end_lineno` is the key: it turns "grab the whole function" into a slice, not a re-parse.

---

### Q17. Why slice the raw source instead of `ast.unparse(node)`?

**Answer:** `ast.unparse()` regenerates source *from the tree* — but the AST does not store comments or original formatting (comments aren't part of the abstract syntax). Unparsing would silently drop every comment and normalize whitespace/quotes. We slice the original text so comments, docstrings, blank lines, and exact style survive into the chunk.

**Justification:** The chunk is both (a) what we embed — and real comments carry semantic signal — and (b) what we show the user as a citation. We want the code *as written*, not a reconstruction. The AST is for *locating*, not *reproducing*.

---

### Q18. How are decorators handled, and why does it matter?

**Answer:** A decorated function's `node.lineno` points at the `def` line, **not** the `@decorator` above it. The decorators live in `node.decorator_list`, each with its own `lineno`. We start the chunk at `decorator_list[0].lineno` when decorators exist, else `node.lineno`.

**Justification:** `@app.route("/login")` or `@staticmethod` is essential context — it often tells you *what the function is* (a route handler, a fixture, a cached call). Cutting it off strips meaning that both the embedding and the LLM need.

---

### Q19. Why iterate `tree.body` (top-level only) instead of `ast.walk()` (every node)?

**Answer:** `ast.walk()` yields *every* node recursively — so a class's methods and any nested functions would each become their own chunk **in addition to** the class chunk that already contains them. That's duplicated, overlapping content. Iterating `tree.body` takes only top-level definitions, so a class becomes exactly one chunk (methods included).

**Justification:** Overlapping chunks waste index space and let near-duplicates crowd out the top-k. The tradeoff: a very large class is one big chunk — acceptable now; revisit with per-method splitting only if classes get unwieldy.

---

### Q20. What are the chunker's edge cases and deliberate blind spots?

**Answer:**
- **Unparseable file** (syntax error, Python 2): `ast.parse` raises `SyntaxError` → we catch it and return `[]`, skipping the file instead of crashing the whole index build.
- **Module-level code** (imports, top-level constants, script body): not captured — only functions/classes. So "where is `CONFIG` defined?" can miss.
- **Leading `#` comments above a def:** not captured. The AST tracks decorators but not comments, so a comment block directly above a function is dropped. (Docstrings *are* captured — they live *inside* the body, between `def` and `end_lineno`.)
- **Non-Python files:** `ast` is Python-only — out of scope (other languages need tree-sitter).

**Justification:** Retrieval can only surface what chunking captured. Knowing these blind spots tells you the failure boundary *before* you debug a "why didn't it find X" later.

---

### Q21. Why is `chunk_source(source, file_path)` a pure function (no file IO)?

**Answer:** It takes a string and returns chunks — it never touches the disk. File reading lives in `chunk_file`; directory walking in `iter_python_files`.

**Justification:** Purity makes the core trivially testable: feed it an inline source string and assert on the result — no temp files, fast, deterministic. The IO stays in thin wrappers around the pure core. This is exactly why the unit tests cover decorators, classes, and syntax errors in a few lines each.

---

## Section 10: Embedder — Implementation Deep Dive (stage 2)

> Builds on Section 3. Section 3 is *why* embeddings/cosine; this is *how* `src/embedder.py` does it.

### Q22. Mechanically, how does `all-MiniLM-L6-v2` turn a chunk into a 384-d vector?

**Answer:** The text is tokenized, run through a 6-layer transformer (the "L6"), which produces one contextual vector per token; those token vectors are then **mean-pooled** into a single 384-dim sentence vector. `sentence-transformers` wraps tokenize → transformer → pool behind one `.encode()` call.

**Justification:** You don't implement attention or pooling — but knowing the output is a *pooled average* explains why a very long chunk produces a blurry, washed-out vector (averaging dilutes specifics). That's the implementation-level reason stage 1's chunking matters so much.

---

### Q23. Why this model specifically?

**Answer:** 384-dim, ~90MB, 6 layers → small and fast on **CPU**, no GPU/API/cost, quality good enough for code Q&A at this scale. Bigger models (e.g. `all-mpnet-base-v2`, 768-dim) score higher but are slower and heavier. The choice lives in `config.EMBEDDING_MODEL` (single source of truth).

**Justification:** For a CPU-only learning project the speed/size win dominates. Swapping the model is a one-line config change — but you **must re-index** afterward (Q7), because the vector space changes.

---

### Q24. How do we make FAISS inner-product search equal cosine similarity, in code?

**Answer:** `model.encode(..., normalize_embeddings=True)` returns unit-length vectors (‖v‖ = 1). For unit vectors, dot product **is** cosine. So when stage 3 uses FAISS `IndexFlatIP` (inner product), it's computing cosine for free. That's invariant #2, implemented in one kwarg.

**Justification:** Normalizing once, at embed time, means stages 3–4 do zero extra math. (Cautionary tale: the misspelled `normalize_embedding` silently fell into `**kwargs` and raised — an unrecognized kwarg means *no normalization happened*.)

---

### Q25. Why load the model with a cached singleton (`lru_cache(maxsize=1)`)?

**Answer:** The model is ~90MB and takes a second or two to load. `lru_cache(maxsize=1)` is a one-line lazy singleton: the first call to `_get_model()` loads it, every later call returns the same cached instance.

**Justification:** This matters most in stage 4, where *every* query embeds text — you don't want to reload 90MB of weights per question.

---

### Q26. How does the code structurally enforce "same model for index and query" (invariant #1)?

**Answer:** Both `embed_chunks` (indexing, many texts) and `embed_query` (querying, one text) funnel through the single `embed_texts`, which uses the one cached model. There is no second code path that *could* pick a different model.

**Justification:** Q7 says *why* a model mismatch destroys retrieval; this is *how* you make the mistake structurally impossible — enforced by design, not by remembering.

---

### Q27. Why return `float32`, and why the `(0, dim)` empty case?

**Answer:** FAISS requires `float32` vectors, so `.astype("float32")` guarantees the dtype before stage 3. And an empty input must still return a 2-D array of shape `(0, dim)` (via `np.empty((0, EMBEDDING_DIM), ...)`) — not a 0-d or `(0, 0)` array — so `index.add()` doesn't choke on a wrong-shaped input.

**Justification:** Getting shape and dtype right *at the boundary* prevents confusing failures two stages downstream.

---

## Section 11: Vector Store — Implementation Deep Dive (stage 3)

> Builds on Section 4. Section 4 is *why* a vector DB / FAISS / `IndexFlatIP`; this is *how* `src/index_store.py` builds, persists, and reloads it.

### Q28. What two files make up a persisted index, and why two?

**Answer:** The FAISS index (`config.INDEX_FILE`, a binary holding only the vectors) and a JSON file (`config.METADATA_FILE`, holding the `CodeChunk` records). FAISS stores float vectors and integer ids — it knows nothing about your file paths or code text. So the human-meaningful data lives separately, as JSON.

**Justification:** Separation of concerns — FAISS does the math, JSON holds the meaning. They're joined at query time by **row position**.

---

### Q29. What is the ordering contract, and why is it the invariant of this stage?

**Answer:** `index.add()` assigns ids `0..N-1` in insertion order, and a search returns those ids. The metadata is saved in the **same order**, so `chunks[i]` describes vector row `i`. That row number is the only link between the two files. Reorder or filter one without doing the identical operation to the other, and every result is mislabeled.

**Justification:** This is *silent* corruption — no crash, just wrong code returned for a given vector. Treat the vectors and the chunk list as a single unit that always moves together.

---

### Q30. Why persist to disk at all — why not rebuild the index every run?

**Answer:** Embedding the whole repo is the expensive offline step. Persisting (`faiss.write_index` + JSON dump) means answering a question only **loads** the index — it never re-embeds. This is the concrete boundary between the OFFLINE pipeline (chunk → embed → index, run on repo change) and the ONLINE pipeline (load → search → answer, run per question).

**Justification:** Turns a multi-second embed into a sub-second load per session — and lets indexing and querying be separate commands (`--index` vs `--ask`).

---

### Q31. Why JSON for metadata instead of `pickle`?

**Answer:** JSON is human-readable (open `metadata.json` and inspect it), **safe** (`pickle` executes arbitrary code on load — a real security risk), and portable. It's slightly larger/slower, which is irrelevant at our scale. `asdict(chunk)` serializes a `CodeChunk` to a dict; `CodeChunk(**d)` rebuilds it — which works because `CodeChunk` is a flat dataclass of JSON-native types.

**Justification:** For small data and a learning project, readability + safety beat pickle's speed every time.

---

### Q32. How do `save_index` and `load_index` avoid path drift?

**Answer:** Both derive their paths from `config.INDEX_FILE` / `config.METADATA_FILE` through a single `_paths()` helper. The filenames are defined once, so save and load **cannot disagree**. (This exact drift — one function writing `vectors.faiss` while the other read `vector.faiss` — was a real bug caught in review.)

**Justification:** Single source of truth — the same principle as `config.EMBEDDING_MODEL` enforcing the same-model invariant in stage 2.

---

### Q33. Why `IndexFlatIP`, and how does `build_index` get the dimension?

**Answer:** `IndexFlatIP` is a flat (exact, brute-force) inner-product index; on L2-normalized vectors, inner product **is** cosine similarity (invariant #2). The dimension comes from `vectors.shape[1]` — derived from the data itself, and equal to `config.EMBEDDING_DIM`. At our scale, exact search is plenty fast; ANN indexes (IVF/HNSW) only pay off at millions of vectors.

**Justification:** The simplest correct choice — no recall loss, no tuning knobs.

---

## Section 12: Retriever — Implementation Deep Dive (stage 4)

> Builds on Section 5. Section 5 is *why* top-k retrieval and how to choose k; this is *how* `src/retriever.py` does it.

### Q34. What does `index.search(query, k)` return?

**Answer:** Two NumPy arrays, each shape `(n_queries, k)`: `scores` and `ids`. For a single query, `scores[0]` and `ids[0]` are length-`k`. `scores` are inner products = cosine (normalized vectors), sorted **descending** (best first). `ids` are row positions into the chunk list — or `-1` padding. Both arrays are always the same length, so `zip(ids[0], scores[0], strict=True)` is safe and self-documenting.

**Justification:** You need both — `ids` to find the chunk, `scores` to rank, threshold, and report confidence.

---

### Q35. Why reshape the query vector to `(1, dim)`?

**Answer:** `index.search` is batched — it expects a 2-D matrix of queries `(n_queries, dim)`. `embed_query` returns 1-D `(dim,)`, so `.reshape(1, -1)` makes it a batch of one; you then read row `[0]` of the results.

**Justification:** One API handles 1 or many queries; we just pass a batch of one.

---

### Q36. Why must you skip `ids < 0`?

**Answer:** When `k` exceeds the number of vectors in the index, FAISS fills the empty slots with `-1`. Doing `chunks[-1]` would silently return the **last** chunk — a wrong result with no error. So `if idx < 0: continue`.

**Justification:** A classic silent bug. The index isn't always ≥ `k` (small repos, filtered indexes), so this must be handled, not assumed away.

---

### Q37. How does retrieval use the stage-3 row-order join?

**Answer:** `search` returns row ids; `chunks[id]` fetches the `CodeChunk` stored at that row. This works only because `save_index` wrote `metadata.json` in the same order the vectors were added. Retrieval is where that contract pays off.

**Justification:** A bare id is meaningless; the join turns "row 7, score 0.81" into "`connect_to_database` in `db.py`".

---

### Q38. Why is `retrieve` a pure function that takes `index` and `chunks` rather than loading them?

**Answer:** The caller (the CLI) calls `load_index()` **once** per session, then asks many questions without reloading the index or re-reading metadata. Keeping `retrieve` free of IO also makes it deterministic and trivial to test (build a tiny index, call it).

**Justification:** Loading is a one-time session cost; retrieval is per-query — don't fold the two together.

---

### Q39. Why return `RetrievalResult(chunk, score)` instead of just chunks?

**Answer:** The score is needed downstream — to rank, to optionally drop weak matches (a `SIMILARITY_THRESHOLD` → "no relevant code found"), and to convey confidence in the final answer. Dropping it loses information you can't recover later.

**Justification:** Carry the score into the LLM stage; it's cheap to keep and useful to have.

---

## Section 13: LLM — Implementation Deep Dive (stage 5)

> Builds on Section 6. Section 6 is *why* the LLM reasons over retrieved code and how RAG curbs hallucination; this is *how* `src/llm.py` does it.

### Q40. How does the prompt ground the model — system instruction vs user content?

**Answer:** Two separate pieces. The **system instruction** (`SYSTEM_INSTRUCTION`, passed via `GenerateContentConfig`) carries the *rules*: use ONLY the provided code, say you don't know if it's absent, cite `file:line`. The **user content** (`build_prompt` output) carries the *data*: the retrieved chunks as context + the question. Rules separate from data is how the API is designed and keeps grounding persistent.

**Justification:** The system instruction is the anti-hallucination lever (§Q12) — it's what makes the model answer from your repo, not its training memory.

---

### Q41. How does `build_prompt` make citations possible?

**Answer:** Each `RetrievalResult` becomes a labeled block — `# <file_path>:<start_line> (<chunk_type> <name>)` then the code. Those labels are the `file_path`/`start_line` carried all the way from stage 1. The model just echoes them back as citations.

**Justification:** Citations aren't magic — they appear because you put `file:line` in the prompt. No labels → no citations.

---

### Q42. Why is `build_prompt` pure and separate from the API call?

**Answer:** It's `(question, results) → str` with no network or key, so you can unit-test the whole prompt shape offline. `answer` is the only impure part (the Gemini call).

**Justification:** Same pure-core / thin-IO-shell pattern as `chunk_source` and `embed_texts` — testability, with the external dependency isolated in one function.

---

### Q43. Why short-circuit on empty results, and guard `response.text`?

**Answer:** If retrieval returned nothing, `answer` returns a fixed message **without calling Gemini** — saving a (quota-limited) API call and refusing to answer ungrounded. And `response.text or "..."` guards the case where Gemini returns no text (e.g. a safety block), so the function never returns `None`.

**Justification:** Don't spend a call — or risk a hallucinated answer — when there's no context; and always hand callers a real string.

---

### Q44. Why cache the client, and how is the API key handled safely?

**Answer:** `_get_client` uses `lru_cache(maxsize=1)` → one client per process (like the embedding model), reused across questions. `load_dotenv()` pulls `GEMINI_API_KEY` from `.env` into `os.environ`; we validate it's present (clear error if not) and pass it explicitly. The secret lives only in `.env`/the environment — `config.py` holds the model *name*, never the key.

**Justification:** Secrets stay out of source control, and the client isn't rebuilt per query.

---

### Q45. What did the `429 ... limit: 0` teach about depending on a hosted LLM?

**Answer:** A `429 RESOURCE_EXHAUSTED` with `limit: 0` means the model has **no free-tier quota** on your key — not that you used it up, so retrying won't help. `gemini-2.0-flash` had `limit: 0`; switching `config.GEMINI_MODEL` to `gemini-2.5-flash` fixed it.

**Justification:** Provider quotas/rate-limits are an operational reality separate from your code. Because the LLM is isolated behind `llm.py` + one config line, swapping the model (or even the provider, e.g. Ollama) is a one-line change, not a refactor.

---

## Appendix A — Dev Tooling, Testing & Git Workflow (reference)

> Engineering-practice notes for this repo (not RAG concepts). Reflects the config actually in use.

### A1. The four dev tools

| Tool | Role | Run it |
|------|------|--------|
| **ruff** | Linter — unused imports, undefined names, import order, bug patterns (replaces flake8 + isort + pyupgrade). | `ruff check --fix .` |
| **ruff format** | Auto-formatter (Black-compatible) — stop hand-formatting. | `ruff format .` |
| **pyright** | Static type checker — verifies type hints, catches mismatches before runtime. | `pyright` |
| **pre-commit** | Runs the above automatically on `git commit`, so unclean code never lands. | `pre-commit run --all-files` |

None of these touch the RAG pipeline — they keep the code clean as it grows.

### A2. `pyproject.toml` (tool config; deps stay in `requirements*.txt`)
- `[tool.ruff]` / `[tool.ruff.lint]` — line length, target version, and lint rule families (`E/W` pycodestyle, `F` pyflakes, `I` isort, `UP` pyupgrade, `B` bugbear).
- `[tool.pyright]` — `include` the source dirs; `venvPath="."` + `venv=".venv"` point pyright at the venv so it resolves installed deps.
- `[tool.pytest.ini_options]` — `pythonpath=["."]` so `from src.chunker import ...` resolves from the repo root; `testpaths=["tests"]`.
- `[project]` must include `version` to be valid PEP 621 (e.g. `version = "0.1.0"`), or omit the table entirely if the file is config-only. No `[build-system]` is needed unless you actually package the project.

### A3. `requirements-dev.txt` (dev-only deps, separate from runtime)
Runtime deps and the tools *you* develop with are different concerns; a deploy installs only `requirements.txt`. The `-r` line pulls runtime in too, so one command sets up everything:
```
-r requirements.txt

pytest
ruff
pyright
pre-commit
```

### A4. `.pre-commit-config.yaml`
ruff + ruff-format from the official mirror; pyright as a `local` hook so it uses your venv's pyright (and therefore sees installed deps). `pass_filenames: false` makes pyright check the whole project, which is how it works best:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9            # placeholder — run `pre-commit autoupdate` to pin the real latest
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: pyright
        language: system
        types: [python]
        pass_filenames: false
```

### A5. First-time setup (Windows PowerShell)
```powershell
python -m venv .venv                          # 1. isolated env
.\.venv\Scripts\Activate.ps1                  # 2. activate (prompt shows (.venv))
# if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned, then re-activate
python -m pip install --upgrade pip           # 3. modern pip
pip install -r requirements-dev.txt           # 4. installs dev + (via -r) runtime
pre-commit install                            # 5. wire the git hook (once per clone)
pre-commit run --all-files                    # 6. optional: run hooks on everything now
```
First `pyright` run quietly downloads its Node runtime — give it a few seconds.

### A6. Test layout
```
tests/
  unit/
    test_chunker.py     # pure-function tests (chunk_source) — fast, no IO
  integration/          # later: full index → retrieve → answer
```
`testpaths=["tests"]` already discovers subfolders. With nested dirs, add this so pytest imports cleanly regardless of duplicate basenames:
```toml
[tool.pytest.ini_options]
importmode = "importlib"
```
Run: `pytest -v`.

### A7. Git workflow (branch per stage, merge commits — no squash)
```bash
git checkout -b feat/stage-1-chunker        # 1. start a stage (uncommitted work follows you)
# ... atomic commits at green checkpoints ...
git push -u origin feat/stage-1-chunker     # 2. first push sets upstream
gh pr create --fill                         # 3. open PR when the stage is DONE (tests green)
gh pr merge --merge --delete-branch         # 4. MERGE COMMIT (not --squash); removes the branch
git checkout main && git pull               # 5. sync local main
```
- **Create a branch** at the start of each stage. **Open the PR** when the stage is complete. **Merge** with `--merge` (a merge commit preserves every commit — no squash). **Delete the branch** immediately after merge.
- Commit message convention (Conventional Commits, imperative mood):
```
feat(chunker): add AST chunker + shared config (stage 1)
test(chunker): cover defs, classes, decorators, skip-dirs
chore: add ruff, pyright, pytest, pre-commit
docs: track CLAUDE.md and learning notes
```
- **Never commit:** `.venv/`, `.rag_index/`, `__pycache__/`, `.env` (all gitignored).

---

*Last updated: Session 7 — stage 5 LLM implementation deep-dive (Q40–Q45)*
