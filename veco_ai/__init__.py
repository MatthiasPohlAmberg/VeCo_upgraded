"""Public API for veco-ai.

The class-based API remains available via ``Vectorize``. For small scripts,
``import veco_ai as veco`` also exposes convenience functions that create an
engine, run one operation, and close it again.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .veco_ai import Vectorize, chunk_text

__version__ = "0.2.0"


def create(**kwargs: Any) -> Vectorize:
    """Create a reusable veco engine."""
    return Vectorize(**kwargs)


def open_database(database: str = "vector_db.json", **kwargs: Any) -> Vectorize:
    """Create a veco engine and load an existing database."""
    return Vectorize(preload_json_path=database, **kwargs)


def vectorize_file(
    inputfile: str,
    database: str = "vector_db.json",
    save: bool = True,
    use_compression: bool = False,
    use_diarization: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Vectorize one file into a database and return a small status payload."""
    engine = Vectorize(preload_json_path=database, **kwargs)
    try:
        before = engine.faiss_index.ntotal
        engine.vectorize(
            inputfile,
            use_compression=use_compression,
            use_diarization=use_diarization,
        )
        if save:
            engine.save_database(database)
        return {
            "database": database,
            "source": inputfile,
            "added_vectors": int(engine.faiss_index.ntotal - before),
            "vector_count": int(engine.faiss_index.ntotal),
        }
    finally:
        engine.close()


def retrieve_context(
    database: str,
    query_text: str,
    top_k: int = 5,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run FAISS retrieval against an existing database without calling an LLM."""
    engine = Vectorize(preload_json_path=database, **kwargs)
    try:
        return engine.retrieve_context(query_text, top_k=top_k)
    finally:
        engine.close()


def query_db(
    database: str,
    question: str,
    llm_model: str = "gemma3:12b",
    top_k: int = 5,
    include_summary: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run end-to-end RAG against an existing database via Ollama."""
    engine = Vectorize(preload_json_path=database, **kwargs)
    try:
        return engine.query(
            database=database,
            question=question,
            llm_model=llm_model,
            top_k=top_k,
            include_summary=include_summary,
        )
    finally:
        engine.close()


# Short aliases for ``import veco_ai as veco`` usage.
vectorize = vectorize_file
query = query_db


__all__ = [
    "Vectorize",
    "__version__",
    "chunk_text",
    "create",
    "open_database",
    "query",
    "query_db",
    "retrieve_context",
    "vectorize",
    "vectorize_file",
]
