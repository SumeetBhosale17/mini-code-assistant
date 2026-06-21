"""General configuration - one source of truth shared across pipeline stages."""

# --- Stage 1: Chunking -------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".py",)

# Dirs we never index: vendored deps, VCS internals, caches, our own output.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        "venv",
        ".venv",
        ".git",
        "__pycache__",
        ".rag_index",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
    }
)

# --- Stage 2: Embedding ------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # must be identical at index & query time
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dim; keep in sync with the model

# --- Stage 3: vector store --------------------------------------------------
INDEX_DIR: str = ".rag_index"  # gitignored dir holding the row files below
INDEX_FILE: str = "vector.faiss"  # the FAISS index
METADATA_FILE: str = "metadata.json"  # chunk records, in the same row order

# --- later stages (commented until then) ------------------------------------
# GEMINI_MODEL: str = "gemini-2.0-flash"
