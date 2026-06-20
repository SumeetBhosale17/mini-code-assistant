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

# --- later stages (commented until then) ------------------------------------
# EMBEDDING_MODEL: str = "all-MiniLM-L6-v2" # Must match between index & query
# INDEX_DIR: str = ".rag_index"
# GEMINI_MODEL: str = "gemini-2.0-flash"
