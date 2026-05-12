# veco-ai

veco-ai is a Python 3.13+ multimodal vectorizer and RAG toolkit. It ingests documents, audio, video, images, and CAD files into a FAISS index and answers questions via a local Ollama LLM.  
Embeddings are stored inside a FAISS index and can optionally be persisted to JSON (fallback), SQLite, or MongoDB. The integrated RAG interface lets you query knowledge bases via local **Ollama** models.

## Stack

| Layer | Technology |
|---|---|
| **Docs** | `unstructured` — PDF, DOCX, PPTX, HTML, TXT, MD, … |
| **Audio** | `faster-whisper` — CTranslate2-backed Whisper transcription |
| **Video** | `FFmpeg` + `PySceneDetect` → frames; `faster-whisper` → transcript |
| **CAD** | `trimesh` (STL/OBJ/PLY/GLB) + `pythonOCC` (STEP/IGES, optional) |
| **Vision** | `Gemma4` via Ollama (images, video frames, CAD renders) |
| **Diarization** | optional `pyannote.audio` (requires `HF_TOKEN`) |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Index** | `FAISS` |
| **RAG** | `Ollama` (local LLM, e.g. `gemma3:12b`) |

## Features

- **Unified extraction**: one `vectorize()` call handles any supported file type
- **Multimodal**: images, video frames, and CAD renders described by Gemma4
- **Speaker diarization**: optional per-speaker transcripts (`pyannote.audio` + HF token required)
- **Video scene analysis**: scene cuts detected, each frame described
- **CAD support**: geometry metadata (volume, surface area, bounding box) + visual description
- **Chunking with overlap** for RAG-ready embeddings
- **Optional summaries** via Ollama (stored as metadata, never embedded)
- **FAISS index** for efficient similarity search
- **Persistence**: JSON fallback, SQLite, or MongoDB

## Project Structure

```
.
|-- veco_ai/
|   |-- __init__.py
|   |-- veco_ai.py              # Core vectorization library
|   |-- veco_diarization.py  # Optional speaker diarization pipeline
|   `-- veco_pic_describe.py # Optional image captioning helpers
|-- test/
|   |-- minimal_example.py   # Minimal package-level API example
|   |-- query_llm.py         # Query an existing DB via FAISS + Ollama
|   `-- veco_test.py         # Full example usage script
|-- requirements.txt
|-- pyproject.toml
|-- test_data/               # Sample files for testing
|-- vector_db.json           # Example JSON database (fallback storage)
`-- UML/                     # Architecture diagrams
```

## Dependencies

**Python packages** (see `requirements.txt`):
- `torch` >= 2.6, `torchaudio` >= 2.6
- `sentence-transformers` >= 3.0, `faiss-cpu` >= 1.9
- `unstructured[all-docs]` >= 0.15
- `faster-whisper` >= 1.2
- `scenedetect` >= 0.6, `opencv-python` >= 4.8
- `trimesh` >= 4.0
- `ollama` >= 0.4

**Optional speaker diarization:**
`pyannote.audio` is not part of the base install because its `lightning`
dependency is not available for every Python 3.13/Windows environment.
Install it separately only if pip can resolve it:

```bash
pip install pyannote.audio
```

**System dependencies** (must be on PATH):
- `ffmpeg` — audio/video decoding
- `tesseract` — OCR fallback for unstructured (optional)

**Optional (CAD — STEP/IGES):**
```bash
conda install -c conda-forge pythonocc-core
```

**HuggingFace token** (free, for optional speaker diarization):
1. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
2. `export HF_TOKEN=hf_...`

## Installation

### 1. Create a virtual environment

```bash
python3.13 -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS
```

### 2. Install the base dependencies

```bash
pip install veco_ai
```

For local development instead of the published wheel, install from source:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

After editable installation the console commands are available:

```bash
veco-ai --help
veco-ai-server
```

### 3. Configure PyTorch (optional)

Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your GPU/CPU setup.  
For a CPU-only environment the default `pip install` from the requirements is sufficient.

## Usage

### Example script (`test/veco_test.py`)

```bash
python test/veco_test.py
```

The script loads or creates `vector_db.json`, vectorizes all files in the `test_data/` folder, and saves the updated database.

### Minimal example

```bash
python test/minimal_example.py
```

This demonstrates the package-level API:

```python
import veco_ai as veco

hits = veco.retrieve_context(
    database="test/vector_db.json",
    query_text="Was steht zur Federsteifigkeit?",
    top_k=3,
    embedding_model="all-MiniLM-L6-v2",
    enable_audio=False,
)
```

### Query an existing database

Use `test/query_llm.py` when a database already exists and you only want to run
retrieval plus an Ollama answer. This does not vectorize files again.

Interactive mode:

```bash
python test/query_llm.py --db test/vector_db.json --llm gemma3:12b
```

Single question:

```bash
python test/query_llm.py --db test/vector_db.json --llm gemma3:12b "Was steht zur Federsteifigkeit?"
```

Show the retrieved chunks as well:

```bash
python test/query_llm.py --db test/vector_db.json --llm gemma3:12b --show-context "Was steht zur Federsteifigkeit?"
```

The embedding model used for querying must match the model used when the
database was built. The script infers common dimensions (`384` →
`all-MiniLM-L6-v2`, `1024` → `BAAI/bge-m3`), but you can set it explicitly:

```bash
python test/query_llm.py --db test/vector_db.json --embedding-model all-MiniLM-L6-v2 --llm gemma3:12b "Welche Themen kommen vor?"
```

### Direct usage in Python

```python
import veco_ai as veco

# JSON fallback backend
engine = veco.create(preload_json_path="vector_db.json")

# Vectorize a file
engine.vectorize("path/to/file.pdf", use_compression=True)

# Persist the database
engine.save_database("vector_db.json")

# Run a RAG query (Ollama required)
res = engine.query(
    database="vector_db.json",
    question="What is this document about?",
    llm_model="gemma3:12b",
)
print(res["answer"])
engine.close()
```

For one-shot scripts, use package-level helpers:

```python
import veco_ai as veco

result = veco.query(
    database="vector_db.json",
    question="Was steht im Dokument?",
    llm_model="gemma3:12b",
    embedding_model="BAAI/bge-m3",
    enable_audio=False,
)
```

## Architecture

The central class is `Vectorize`:

- **Input detection**: identifies the file type
- **Text extraction**: uses type-specific libraries
- **Optional compression**: generates summaries through Ollama
- **Chunking**: splits text into overlapping segments
- **Embedding**: performed with `sentence-transformers`
- **Storage**: FAISS index plus JSON/SQLite/MongoDB backends
- **RAG**: retrieves relevant context and optionally queries an Ollama model

## Retrieval Flow

The persisted database is a machine-readable search index, not an intelligent
agent. It stores text chunks, source metadata, and embedding vectors. At query
time, VeCo embeds the user question, asks FAISS for the closest chunk vectors,
maps the returned IDs back to plain text, and sends only those text chunks to
Ollama.

```text
question -> embedding -> FAISS search -> chunk IDs -> plain-text context -> Ollama answer
```

Ollama does not read the FAISS index directly. FAISS performs semantic
retrieval; Ollama receives normal text and writes the final answer. VeCo can
therefore stay a retrieval/vectorization module while an external agent
framework can orchestrate query rewriting, multi-query retrieval, reranking, or
answer formatting around it.

## Development

Install the development extras to run linting and tests:

```bash
pip install .[dev]
pytest
```

## License

The project is released under the terms of [CC0 1.0 Universal](LICENSE).
