"""
Query an existing veco-ai database via FAISS retrieval and Ollama.

This script does not vectorize or modify input files. It only loads an existing
database, retrieves relevant chunks, and asks the selected Ollama model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from veco_ai import Vectorize  # noqa: E402


DEFAULT_DB_CANDIDATES = [
    PROJECT_ROOT / "test" / "vector_db.json",
    PROJECT_ROOT / "test" / "vector_db_fwk.json",
    PROJECT_ROOT / "vector_db.json",
]
DEFAULT_DB = next((path for path in DEFAULT_DB_CANDIDATES if path.exists()), DEFAULT_DB_CANDIDATES[0])


def detect_vector_dim(db_path: Path) -> Optional[int]:
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for rec in data.get("outputdb", []):
        vec = rec.get("vector")
        if isinstance(vec, list) and vec:
            return len(vec)
    return None


def infer_embedding_model(vector_dim: Optional[int]) -> str:
    if vector_dim == 384:
        return "all-MiniLM-L6-v2"
    if vector_dim == 1024:
        return "BAAI/bge-m3"
    return "BAAI/bge-m3"


def print_result(result: dict, show_context: bool) -> None:
    print("\nAnswer:")
    print(result["answer"])

    sources = result.get("sources") or []
    if sources:
        print("\nSources:")
        for src in sources:
            print(f"- {src}")

    contexts = result.get("contexts") or []
    if not contexts:
        print(
            "\nNo context was retrieved. Check whether --embedding-model matches "
            "the model used to build the database."
        )
    elif show_context:
        print("\nRetrieved context:")
        for i, hit in enumerate(contexts, 1):
            source = hit.get("source", "?")
            chunk_idx = hit.get("chunk_idx", "?")
            text = (hit.get("text") or "").replace("\n", " ")
            print(f"\n[{i}] {source} / chunk {chunk_idx}")
            print(text[:1000])


def run_query(
    veco: Vectorize,
    db_path: Path,
    question: str,
    llm_model: str,
    top_k: int,
    show_context: bool,
) -> None:
    result = veco.query(
        database=str(db_path),
        question=question,
        llm_model=llm_model,
        top_k=top_k,
    )
    print_result(result, show_context=show_context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RAG query against an existing veco-ai database."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask against the database. Omit to start interactive mode.",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"Path to veco JSON database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--llm",
        default="gemma4:e4b",
        help="Ollama LLM model for the answer (default: gemma4:e4b)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="sentence-transformers model used when the DB was built",
    )
    parser.add_argument(
        "--embedder-backend",
        choices=["sbert", "ollama"],
        default="sbert",
        help="Embedding backend used for query embedding (default: sbert)",
    )
    parser.add_argument(
        "--ollama-embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model when --embedder-backend=ollama",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to pass to Ollama (default: 5)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context snippets after the answer",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    vector_dim = detect_vector_dim(db_path)
    embedding_model = args.embedding_model or infer_embedding_model(vector_dim)

    print(f"DB:              {db_path}")
    print(f"LLM:             {args.llm}")
    print(f"Embedder:        {args.embedder_backend}")
    if args.embedder_backend == "sbert":
        print(f"Embedding model: {embedding_model}")
    else:
        print(f"Ollama embed:    {args.ollama_embed_model}")
    if vector_dim:
        print(f"DB vector dim:   {vector_dim}")
    print(f"Top-k:           {args.top_k}")
    print("-" * 72)

    veco = Vectorize(
        preload_json_path=str(db_path),
        embedding_model=embedding_model,
        enable_audio=False,
        embedder_backend=args.embedder_backend,
        ollama_embed_model=args.ollama_embed_model,
    )

    try:
        if args.question:
            run_query(
                veco=veco,
                db_path=db_path,
                question=args.question,
                llm_model=args.llm,
                top_k=args.top_k,
                show_context=args.show_context,
            )
            return

        print("Interactive mode. Type 'exit', 'quit', or press Ctrl+C to stop.")
        while True:
            question = input("\nQuestion> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break
            try:
                run_query(
                    veco=veco,
                    db_path=db_path,
                    question=question,
                    llm_model=args.llm,
                    top_k=args.top_k,
                    show_context=args.show_context,
                )
            except RuntimeError as exc:
                print(f"\nQuery failed: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        veco.close()


if __name__ == "__main__":
    main()
