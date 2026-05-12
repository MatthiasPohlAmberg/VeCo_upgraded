"""
Minimal veco-ai example.

Requires an existing database. The script demonstrates the package-level API:

    import veco_ai as veco
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import veco_ai as veco  # noqa: E402


DB_CANDIDATES = [
    PROJECT_ROOT / "test" / "vector_db.json",
    PROJECT_ROOT / "test" / "vector_db_fwk.json",
    PROJECT_ROOT / "vector_db.json",
]


def first_existing_database() -> Path:
    for path in DB_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit("No database found. Run test/veco_test.py first.")


def detect_vector_dim(db_path: Path) -> Optional[int]:
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for rec in data.get("outputdb", []):
        vector = rec.get("vector")
        if isinstance(vector, list) and vector:
            return len(vector)
    return None


def infer_embedding_model(vector_dim: Optional[int]) -> str:
    if vector_dim == 384:
        return "all-MiniLM-L6-v2"
    if vector_dim == 1024:
        return "BAAI/bge-m3"
    return "BAAI/bge-m3"


def main() -> None:
    db_path = first_existing_database()
    question = "Welche Themen kommen in der Datenbank vor?"
    embedding_model = infer_embedding_model(detect_vector_dim(db_path))

    print(f"DB: {db_path}")
    print(f"Embedding model: {embedding_model}")
    print(f"Question: {question}")
    print("-" * 72)

    hits = veco.retrieve_context(
        database=str(db_path),
        query_text=question,
        top_k=3,
        embedding_model=embedding_model,
        enable_audio=False,
    )

    print("Top retrieval hits:")
    for i, hit in enumerate(hits, 1):
        source = hit.get("source", "?")
        text = (hit.get("text") or "").replace("\n", " ")
        print(f"{i}. {source}")
        print(f"   {text[:240]}")

    print("\nTo ask Ollama as well:")
    print(
        f'py -3.13 test\\query_llm.py --db "{db_path}" '
        f'--embedding-model {embedding_model} --llm gemma4:e4b "{question}"'
    )


if __name__ == "__main__":
    main()
