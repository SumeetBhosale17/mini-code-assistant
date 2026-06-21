"""Tiny retrieval eval: for each question, is the expected file in the top-k results?

Builds an index over src/ in memory (no pre-built .rag_index needed, and no LLM call),
runs retrieval, and reports recall@k. Run from the repo root:

    python -m evals.run_eval
"""

import json
from pathlib import Path

from src import config
from src.chunker import chunk_repo
from src.embedder import embed_chunks
from src.index_store import build_index
from src.retriever import retrieve

CASES = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))


def main() -> None:
    chunks = chunk_repo("src")
    index = build_index(embed_chunks(chunks))

    passed = 0
    for case in CASES:
        # min_score=0.0 measures pure ranking (recall), not threshold filtering
        results = retrieve(case["question"], index, chunks, min_score=0.0)
        files = [r.chunk.file_path for r in results]
        hit = any(case["expect"] in f for f in files)
        passed += hit
        print(f"[{'PASS' if hit else 'FAIL'}] {case['question']}")
        print(f"        want {case['expect']!r} in: {files}")

    print(f"\nRecall@{config.TOP_K}: {passed}/{len(CASES)} retrieved the expected file")


if __name__ == "__main__":
    main()
