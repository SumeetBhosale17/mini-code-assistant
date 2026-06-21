"""Stage 6 — CLI entry point: --index, --ask, or an interactive session (default)."""

import argparse

import faiss

from src import config
from src.chunker import CodeChunk, chunk_repo
from src.embedder import embed_chunks
from src.index_store import build_index, load_index, save_index
from src.llm import answer
from src.retriever import retrieve


def run_index(repo_path: str, index_dir: str = config.INDEX_DIR) -> None:
    chunks = chunk_repo(repo_path)
    if not chunks:
        print("No Python chunks found.")
        return
    save_index(build_index(embed_chunks(chunks)), chunks, index_dir)
    print(f"Indexed {len(chunks)} chunks -> {index_dir}/")


def _answer_and_print(
    question: str,
    index: faiss.Index,
    chunks: list[CodeChunk],
) -> None:
    """Retrieve -> answer -> print. Shared by one-shot --ask and the chat loop."""
    results = retrieve(question, index, chunks)
    print(answer(question, results))
    if results:
        print("\nSources:")
        for r in results:
            print(f"    {r.chunk.file_path}:{r.chunk.start_line}  ({r.score:.2f})")


def run_ask(question: str) -> None:
    try:
        index, chunks = load_index()
    except FileNotFoundError as e:
        print(e)
        return
    _answer_and_print(question, index, chunks)


def run_chat() -> None:
    """Interactive session: load the index ONCE, then answer questions in a loop."""
    try:
        index, chunks = load_index()
    except FileNotFoundError as e:
        print(e)
        return
    print("Ask about the codebase. Type 'exit' (or Ctrl-D) to quit.\n")
    while True:
        try:
            question = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        _answer_and_print(question, index, chunks)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RAG code assistant")
    group = parser.add_mutually_exclusive_group()  # optional: no flag -> interactive
    group.add_argument("--index", metavar="REPO", help="Index a repository (run once)")
    group.add_argument("--ask", metavar="QUESTION", help="Ask a single question")
    args = parser.parse_args()
    if args.index:
        run_index(args.index)
    elif args.ask:
        run_ask(args.ask)
    else:
        run_chat()


if __name__ == "__main__":
    main()
