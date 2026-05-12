"""
veco-ai test script
-------------------
Ingests all files in test_data/, runs a semantic search, and optionally
a RAG query via Ollama.

Deliberately uses lightweight models so the test runs quickly without GPU:
  - Embedding: all-MiniLM-L6-v2  (384-dim, ~80 MB, fast)
  - faster-whisper: base          (~150 MB, CPU-friendly)

Switch to production models in Vectorize() once the full stack is set up:
  - embedding_model="BAAI/bge-m3"
  - audio_model_size="large-v3-turbo"
"""

from __future__ import annotations
import sys
from pathlib import Path

# Works both when installed as a package and from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from veco_ai import Vectorize  # noqa: E402


DB_PATH = PROJECT_ROOT / "test" / "vector_db.json"
TEST_DATA = PROJECT_ROOT / "test_data"

# Test query — matches the English SCOTUS audio + PDF content in test_data
QUERY = "Worum geht es in den Dokumenten?"  # "What are the documents about?" in German
LLM_MODEL = "gemma4:e4b"


def main() -> None:
    print("=" * 60)
    print("veco-ai — test run")
    print("=" * 60)
    print(f"DB:        {DB_PATH}")
    print(f"Test data: {TEST_DATA}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    veco = Vectorize(
        preload_json_path=str(DB_PATH),
        embedding_model="BAAI/bge-m3",          # all-MiniLM-L6-v2 or BAAI/bge-m3
        audio_model_size="large-v3-turbo",                # base, large-v3-turbo, or large-v3-onnx
        language="de",                          # test files are German
        use_cosine=True,
        index_type="flat",
    )

    try:
        # ── Ingest ───────────────────────────────────────────────────────────
        if not TEST_DATA.is_dir():
            print(f"\nERROR: test_data directory not found: {TEST_DATA}")
            return

        files = sorted(f for f in TEST_DATA.iterdir() if f.is_file())
        print(f"\nIngesting {len(files)} file(s) ...")
        print("-" * 60)

        aborted = False
        for f in files:
            before = veco.faiss_index.ntotal
            try:
                veco.vectorize(str(f), use_compression=False)
                added = veco.faiss_index.ntotal - before
                status = "OK "
            except KeyboardInterrupt:
                print("\n  Aborted — saving progress so far ...")
                aborted = True
                break
            except Exception as exc:
                added = 0
                status = "ERR"
                print(f"  {status}  {f.name}: {exc}")
                continue
            print(f"  {status}  {f.name:<52}  +{added} chunk(s)")

        print(f"\nTotal vectors:  {veco.faiss_index.ntotal}")
        if aborted:
            return

        # ── Semantic search (no LLM needed) ──────────────────────────────────
        print(f'\n--- Semantic search: "{QUERY}" (top 3) ---')
        hits = veco.retrieve_context(QUERY, top_k=3)
        if hits:
            for i, hit in enumerate(hits, 1):
                snippet = hit.get("text", "")[:100].replace("\n", " ")
                print(f"  {i}. [{Path(hit.get('source', '?')).name}]  {snippet}…")
        else:
            print("  (no results — database empty?)")

        # ── RAG query (requires a running Ollama instance) ───────────────────
        print(f'\n--- RAG query (model: {LLM_MODEL}) ---')
        try:
            result = veco.query(
                database=str(DB_PATH),
                question=QUERY,
                llm_model=LLM_MODEL,
                top_k=3,
            )
            print(f"  Answer:  {result['answer']}")
            print(f"  Sources: {result.get('sources', [])}")
        except RuntimeError as exc:
            print(f"  Skipped — Ollama not reachable: {exc}")

    finally:
        veco.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
