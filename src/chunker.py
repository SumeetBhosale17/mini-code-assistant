"""Stage 1 — split source files into AST-based chunks (one function/class each)."""

import ast
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src import config


@dataclass
class CodeChunk:
    content: str  # verbatim source of this unit - what we embed and show the user
    file_path: str  # origin, for citations like src/db.py:42
    start_line: int  # 1-based startline, for citations
    chunk_type: str  # "function" | "class"
    name: str  # symbol name, for readability + debugging


def chunk_source(
    source: str,
    file_path: str,
) -> list[CodeChunk]:
    """Parse one file's text into function/class chunks.

    Pure (no IO). Boundaries come from the AST,
    but the *content* is sliced from the raw source
    so comments and formatting survive -
    ast.unparse() would discard them.
    """
    chunks: list[CodeChunk] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = (
                node.decorator_list[0].lineno if node.decorator_list else node.lineno
            )
            end = node.end_lineno
            content = "\n".join(lines[start - 1 : end])
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(CodeChunk(content, file_path, start, chunk_type, node.name))

    return chunks


def chunk_file(file_path: str) -> list[CodeChunk]:
    """Read a file from disk and hand its text to chunk_source"""
    source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_source(source, file_path)
    return chunks


def iter_python_files(
    repo_path: str,
    skip_dirs: frozenset[str] = config.SKIP_DIRS,
    extensions: tuple[str, ...] = config.SUPPORTED_EXTENSIONS,
) -> Iterator[str]:
    """Yield .py paths under repo_path, skipping vendored/generated dirs."""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith(extensions):  # tuple -> checks all extensions
                yield os.path.join(root, file)


def chunk_repo(repo_path: str) -> list[CodeChunk]:
    """Flatten every file's chunk into single list the embedder will consume."""
    all_chunks: list[CodeChunk] = []

    for file_path in iter_python_files(repo_path):
        all_chunks.extend(chunk_file(file_path))

    return all_chunks
