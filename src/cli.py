"""Stage 6 — command-line entry point wiring --index and --ask together."""

import argparse

from src import config
from src.chunker import chunk_repo
from src.embedder import embed_chunks
from src.index_store import build_index, load_index, save_index
from src.llm import answer
from src.retriever import retrieve


def run_index(repo_path: str) -> None:
    chunks = chunk_repo(repo_path)
    if not chunks:
        print("No Python chunks found.")
        return
    save_index(build_index(embed_chunks(chunks)), chunks)
    print(f"Indexed {len(chunks)} chunks -> {config.INDEX_DIR}/")


def run_ask(question: str) -> None:
    try:
        index, chunks = load_index()
    except FileNotFoundError as e:
        print(e)
        return
    results = retrieve(question, index, chunks)
    print(answer(question, results))
    print("\nSources:")
    for r in results:
        print(f"    {r.chunk.file_path}:{r.chunk.start_line}  ({r.score:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RAG code assistant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", metavar="REPO", help="Index a repository (run once)")
    group.add_argument("--ask", metavar="QUESTION", help="Ask about the indexed repo")
    args = parser.parse_args()
    if args.index:
        run_index(args.index)
    else:
        run_ask(args.ask)


if __name__ == "__main__":
    main()
