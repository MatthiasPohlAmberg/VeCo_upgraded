# veco_ai.py
# -----------------------------------------------------------------------------
# veco-ai — Multimodal vectorizer and RAG toolkit
#
# Input stack:
#   Docs:   unstructured  (PDF, DOCX, PPTX, HTML, TXT, MD, …)
#   Audio:  faster-whisper
#   Video:  FFmpeg + PySceneDetect + faster-whisper (audio) + Gemma4 (frames)
#   CAD:    trimesh (STL/OBJ/PLY) + pythonOCC (STEP/IGES, optional)
#   Vision: Gemma4 via Ollama
#
# Storage:   FAISS (+ optional JSON / SQLite / MongoDB backends)
# Reasoning: Ollama (local LLM)
#
# v0.2: Upgraded vectorization + multimodal + API capabilities:
#   - BAAI/bge-m3 embedding model (multilingual SotA, 1024-dim)
#   - Cosine similarity via normalized embeddings + IndexFlatIP
#   - HNSW approximate nearest-neighbor index option
#   - CLIP for native image↔text embedding (use_clip=True)
#   - Parallel video frame description via ThreadPoolExecutor
#   - faster-whisper now optional when enable_audio=False
#   - OllamaEmbedder: unified local embedding via Ollama API (embedder_backend="ollama")
#   - docling for superior PDF/DOCX table & layout extraction
#   - faster-whisper large-v3-turbo default (4× faster than large-v3)
# -----------------------------------------------------------------------------

from __future__ import annotations
import atexit
import os
import signal
import sys
import base64
import hashlib
import importlib
import json
import logging
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- Hard dependencies -------------------------------------------------------

try:
    import torch  # type: ignore
except ImportError as exc:
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR: Optional[Exception] = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore
except ImportError as exc:
    BatchedInferencePipeline = None  # type: ignore
    WhisperModel = None  # type: ignore
    _FASTER_WHISPER_IMPORT_ERROR: Optional[Exception] = exc
else:
    _FASTER_WHISPER_IMPORT_ERROR = None

try:
    from pyannote.audio import Pipeline as _PyannotePipeline  # type: ignore
except Exception:
    _PyannotePipeline = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:
    SentenceTransformer = None  # type: ignore
    _SBERT_IMPORT_ERROR: Optional[Exception] = exc
else:
    _SBERT_IMPORT_ERROR = None

try:
    from faiss import (  # type: ignore
        IndexFlatL2,
        IndexFlatIP,
        IndexIDMap,
        IndexHNSWFlat,
    )
    _FAISS_IMPORT_ERROR = None
except ImportError as exc:
    IndexFlatL2 = IndexFlatIP = IndexIDMap = IndexHNSWFlat = None  # type: ignore
    _FAISS_IMPORT_ERROR = exc

# ---- PIL (for CLIP image encoding) ------------------------------------------

try:
    from PIL import Image as _PILImage  # type: ignore
    _PIL_OK = True
except ImportError:
    _PILImage = None  # type: ignore
    _PIL_OK = False

# ---- LibreOffice PATH fix (Windows) -----------------------------------------
# unstructured calls soffice via subprocess; on Windows it is rarely in PATH
# even when installed, so we probe the standard install locations once.

if sys.platform == "win32":
    _LO_CANDIDATES = [
        r"C:\Program Files\LibreOffice\program",
        r"C:\Program Files (x86)\LibreOffice\program",
    ]
    for _lo in _LO_CANDIDATES:
        if Path(_lo).is_dir():
            _cur_path = os.environ.get("PATH", "")
            if _lo.lower() not in _cur_path.lower():
                os.environ["PATH"] = _lo + os.pathsep + _cur_path
                logging.getLogger("veco").debug("Added LibreOffice to PATH: %s", _lo)
            break

# ---- Document parsing (unstructured) ----------------------------------------

try:
    from unstructured.partition.auto import partition as _unstructured_partition  # type: ignore
    _UNSTRUCTURED_OK = True
except ImportError:
    _unstructured_partition = None
    _UNSTRUCTURED_OK = False

# ---- Document parsing: docling (optional, better layout/table extraction) ---

try:
    from docling.document_converter import DocumentConverter as _DoclingConverter  # type: ignore
    _DOCLING_OK = True
except ImportError:
    _DoclingConverter = None  # type: ignore
    _DOCLING_OK = False

# ---- Document parsing: PyMuPDF (PDF fallback) --------------------------------

try:
    import fitz as _fitz  # type: ignore  # PyMuPDF
    _PYMUPDF_OK = True
except ImportError:
    _fitz = None  # type: ignore
    _PYMUPDF_OK = False

# ---- Document parsing: mammoth (DOC/DOCX fallback) --------------------------

try:
    import mammoth as _mammoth  # type: ignore
    _MAMMOTH_OK = True
except ImportError:
    _mammoth = None  # type: ignore
    _MAMMOTH_OK = False

# ---- Video scene detection (PySceneDetect) -----------------------------------

try:
    from scenedetect import detect as _scene_detect, ContentDetector as _ContentDetector  # type: ignore
    _SCENEDETECT_OK = True
except ImportError:
    _SCENEDETECT_OK = False

# ---- CAD: mesh (trimesh) -----------------------------------------------------

try:
    import trimesh  # type: ignore
    _TRIMESH_OK = True
except ImportError:
    trimesh = None  # type: ignore
    _TRIMESH_OK = False

# ---- CAD: STEP / IGES (pythonOCC, optional) ----------------------------------

try:
    from OCC.Extend.DataExchange import read_step_file, read_iges_file  # type: ignore
    from OCC.Core.GProp import GProp_GProps  # type: ignore
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties  # type: ignore
    from OCC.Core.Bnd import Bnd_Box  # type: ignore
    from OCC.Core.BRepBndLib import brepbndlib_Add  # type: ignore
    _OCC_OK = True
except ImportError:
    _OCC_OK = False

# ---- Optional: Ollama --------------------------------------------------------

try:
    import ollama  # type: ignore
except Exception:
    ollama = None


# ---- Logging -----------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("veco")


# ---- Helpers -----------------------------------------------------------------

def _require_dependency(module: Any, name: str, import_error: Optional[Exception]) -> None:
    if module is None:
        msg = (
            f"The library '{name}' is required for this feature. "
            "Please install it as documented in requirements.txt."
        )
        raise RuntimeError(msg) from import_error


def _try_import_storages() -> Any:
    try:
        return importlib.import_module("storages")
    except Exception:
        return None



def _relpath(p: str) -> str:
    try:
        return os.path.relpath(p, start=os.getcwd()) if os.path.isabs(p) else p
    except Exception:
        return p


def _cad_suffix(path: str) -> str:
    """Return the meaningful CAD suffix, ignoring numeric revision suffixes."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    while suffixes and suffixes[-1][1:].isdigit():
        suffixes.pop()
    return suffixes[-1] if suffixes else Path(path).suffix.lower()


def chunk_text(text: str, chunk_chars: int = 1800, overlap_chars: int = 200) -> List[str]:
    """Character-based chunking with sentence-boundary awareness and overlap."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    n, i = len(text), 0
    while i < n:
        end = min(i + chunk_chars, n)
        cut = text.rfind(".", i, end)
        if cut == -1 or cut < i + int(0.6 * chunk_chars):
            cut = end
        chunk = text[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut >= n:
            break
        i = max(0, cut - overlap_chars)
    return chunks


def _run_ffmpeg(*args: str, timeout: int = 300) -> bool:
    """Run ffmpeg with the given arguments; return True on success."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", *args],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg call failed: %s", exc)
        return False


# ---- Fallback embedder -------------------------------------------------------

class FallbackSentenceEmbedder:
    """Deterministic hashing-based embedder used when SBERT is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            digest = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).digest()
            expanded = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            tiled = np.tile(expanded, int(np.ceil(self.dim / expanded.size)))[: self.dim]
            vec = tiled / 255.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            vectors[idx] = vec
        return vectors


# ---- Ollama embedder (unified local API, no sentence-transformers needed) ---

class OllamaEmbedder:
    """
    Embedder using Ollama's native batch embedding API (ollama.embed).
    Eliminates the sentence-transformers dependency when backend="ollama".
    Recommended models (pull first):
      ollama pull nomic-embed-text    # 768-dim, fast, good quality
      ollama pull mxbai-embed-large   # 1024-dim, higher quality
      ollama pull BAAI/bge-m3         # 1024-dim, multilingual SotA
    """

    def __init__(self, model: str = "nomic-embed-text"):
        if ollama is None:
            raise RuntimeError("Ollama not available.")
        self.model = model
        # Detect embedding dimension via a probe call
        resp = ollama.embed(model=model, input=["dimension probe"])
        self._dim = len(resp["embeddings"][0])
        logger.info("OllamaEmbedder ready: model=%s, dim=%d", model, self._dim)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts: Any,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        resp = ollama.embed(model=self.model, input=list(texts))
        vecs = np.asarray(resp["embeddings"], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
        return vecs


# ---- Spinner -----------------------------------------------------------------

class Spinner:
    _CHARS = ["|", "/", "-", "\\"]

    def __init__(self, message: str = "Processing"):
        self.message = message
        self.stop_running = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        i = 1  # frame 0 already printed by start()
        while not self.stop_running:
            sys.stdout.write(f"\r{self.message} {self._CHARS[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

    def start(self) -> None:
        # Print frame 0 synchronously so the message is visible before the OS
        # schedules the animation thread (model loading starts immediately after).
        sys.stdout.write(f"{self.message} {self._CHARS[0]}")
        sys.stdout.flush()
        try:
            self.thread.start()
        except RuntimeError:
            pass

    def stop(self) -> None:
        self.stop_running = True
        try:
            self.thread.join(timeout=0.3)
        except RuntimeError:
            pass


# ---- Main class --------------------------------------------------------------

class Vectorize:
    """
    Converts documents, audio, video, images, and CAD files into FAISS-indexed
    embeddings for RAG queries via Ollama.

    Input stack:
      - Docs:   unstructured  (PDF, DOCX, PPTX, HTML, TXT, …)
      - Audio:  faster-whisper
      - Video:  FFmpeg + PySceneDetect + faster-whisper (audio) + Gemma4 (frames)
      - CAD:    trimesh (STL/OBJ/PLY) + pythonOCC (STEP/IGES, optional)
      - Vision: Gemma4 via Ollama + optional CLIP

    Key parameters:
      embedding_model  — sentence-transformers model (default: BAAI/bge-m3)
      use_cosine       — normalize embeddings and use IP (cosine) similarity
      index_type       — "flat" for exact search, "hnsw" for approximate ANN
      use_clip         — enable CLIP for native image↔text search
    """

    def __init__(
        self,
        default_model: str = "gemma3:12b",      # Ollama text LLM for RAG
        vision_model: str = "gemma4:e4b",            # Ollama multimodal for images/video/CAD
        embedding_model: str = "BAAI/bge-m3",   # sentence-transformers model
        preload_json_path: Optional[str] = "vector_db.json",
        storage: Optional[object] = None,
        storage_kind: Optional[str] = None,
        storage_kwargs: Optional[dict] = None,
        write_through: bool = True,
        enable_audio: bool = True,
        audio_model_size: str = "large-v3-turbo",  # 4× faster than large-v3; use "base"/"small" on CPU
        language: str = "de",
        batch_size: int = 8,
        hf_token: Optional[str] = None,
        fallback_embedding_dim: int = 1024,
        force_fallback_embedder: bool = False,
        embedder_backend: str = "sbert",         # "sbert" (sentence-transformers) | "ollama"
        ollama_embed_model: str = "nomic-embed-text",  # Ollama embed model when backend="ollama"
        use_cosine: bool = True,                 # cosine similarity via IndexFlatIP
        index_type: str = "flat",               # "flat" (exact) | "hnsw" (approx ANN)
        use_clip: bool = False,                  # enable CLIP for image embedding
    ) -> None:
        _require_dependency(torch, "torch", _TORCH_IMPORT_ERROR)
        # faster-whisper only required when audio is actually enabled
        if enable_audio:
            _require_dependency(WhisperModel, "faster-whisper", _FASTER_WHISPER_IMPORT_ERROR)
        _require_dependency(IndexFlatL2, "faiss-cpu", _FAISS_IMPORT_ERROR)
        _require_dependency(IndexIDMap, "faiss-cpu", _FAISS_IMPORT_ERROR)

        self.default_model = default_model
        self.vision_model = vision_model
        self.preload_json_path = _relpath(preload_json_path or "vector_db.json")
        self.write_through = write_through
        self.language = language
        self._batch_size = batch_size
        self._use_cosine = use_cosine
        self._index_type = index_type.lower()
        self.hf_token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

        self.outputdb: List[Dict[str, Any]] = []
        self.id_lookup: Dict[int, Dict[str, Any]] = {}
        self._next_vector_id = 0
        self._next_doc_id = 0
        self._active_db: Optional[str] = None

        # External storage (SQLite / MongoDB)
        self._ext_storage = None
        if storage is not None:
            self._ext_storage = storage
        elif storage_kind is not None:
            _stor = _try_import_storages()
            if _stor is not None:
                if storage_kind.lower() == "sqlite":
                    self._ext_storage = _stor.SqliteStorage(**(storage_kwargs or {}))
                elif storage_kind.lower() == "mongo":
                    self._ext_storage = _stor.MongoStorage(**(storage_kwargs or {}))
                else:
                    raise ValueError(f"Unknown storage_kind: {storage_kind!r}")
            else:
                logger.warning("storages.py not found — staying with JSON fallback.")

        # faster-whisper
        self._audio_requested = enable_audio
        self._audio_available = False
        self.whisper_model = None
        self.whisper_pipeline = None
        self.audio_model_size = audio_model_size
        self._diarize_pipeline: Any = None

        # CLIP state (set after model loading below)
        self._use_clip = False
        self._clip_dim = 0
        self._clip_lookup: Dict[int, Dict[str, Any]] = {}
        self._next_clip_id = 0

        spinner = Spinner("Initializing models")
        spinner.start()
        try:
            if self._audio_requested and WhisperModel is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                try:
                    self.whisper_model = WhisperModel(
                        audio_model_size,
                        device=device,
                        compute_type=compute_type,
                    )
                    self.whisper_pipeline = (
                        BatchedInferencePipeline(model=self.whisper_model)
                        if BatchedInferencePipeline is not None
                        else self.whisper_model
                    )
                    self._audio_available = True
                except Exception as exc:
                    logger.warning("faster-whisper init failed (%s). Audio disabled.", exc)

            # Embedder: Ollama native API, SBERT, or hash fallback
            self.embedder: Any = None
            if embedder_backend.lower() == "ollama" and ollama is not None:
                try:
                    self.embedder = OllamaEmbedder(model=ollama_embed_model)
                except Exception as exc:
                    logger.warning("OllamaEmbedder init failed (%s) — falling back to SBERT.", exc)
            if self.embedder is None:
                if force_fallback_embedder or SentenceTransformer is None:
                    self.embedder = FallbackSentenceEmbedder(dim=fallback_embedding_dim)
                    logger.info("Using fallback hash-based embedder (dim=%d).", fallback_embedding_dim)
                else:
                    self.embedder = SentenceTransformer(embedding_model)
                    logger.info("Loaded embedding model: %s", embedding_model)

            self._embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
            self.faiss_index = self._create_faiss_index(self._embedding_dim)
            logger.info(
                "FAISS index: type=%s, similarity=%s, dim=%d",
                self._index_type,
                "cosine (IP)" if use_cosine else "L2",
                self._embedding_dim,
            )

            # CLIP model for native image↔text embedding
            if use_clip and SentenceTransformer is not None:
                try:
                    self.clip_model: Any = SentenceTransformer("clip-ViT-B-32")
                    self._clip_dim = int(self.clip_model.get_sentence_embedding_dimension())
                    self._clip_index = IndexIDMap(IndexFlatIP(self._clip_dim))  # type: ignore[operator]
                    self._use_clip = True
                    logger.info("CLIP model loaded (dim=%d).", self._clip_dim)
                except Exception as exc:
                    logger.warning("CLIP init failed (%s). CLIP search disabled.", exc)

            if ollama is not None:
                self._check_ollama()
        finally:
            spinner.stop()

        self._closed = False
        self._close_lock = threading.Lock()

        if self._ext_storage is not None:
            self._bootstrap_from_storage()
        else:
            self.load_database()

        # Safe shutdown: save + close on any exit (normal, Ctrl+C, SIGTERM)
        atexit.register(self.close)
        if threading.current_thread() is threading.main_thread():
            for _sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    _prev = signal.getsignal(_sig)
                    def _make_handler(prev, signum=_sig):
                        def _handler(sig, frame):
                            self.close()
                            # Re-raise with the original handler so callers see KeyboardInterrupt
                            if callable(prev):
                                prev(sig, frame)
                            else:
                                signal.signal(sig, signal.SIG_DFL)
                                signal.raise_signal(sig)
                        return _handler
                    signal.signal(_sig, _make_handler(_prev))
                except (OSError, ValueError):
                    pass  # not available in this context (e.g. non-main thread, embedded)

    # ---- Infrastructure -------------------------------------------------------

    def _create_faiss_index(self, dim: int) -> Any:
        """Create a FAISS index based on configured type and similarity metric."""
        if self._index_type == "hnsw":
            # IndexHNSWFlat: fast approximate ANN, good for large databases (>100k vectors)
            hnsw = IndexHNSWFlat(dim, 32)  # type: ignore[operator]  M=32 connections per node
            hnsw.hnsw.efConstruction = 200  # higher = better quality, slower build
            hnsw.hnsw.efSearch = 64         # higher = better recall, slower search
            return IndexIDMap(hnsw)         # type: ignore[operator]
        elif self._use_cosine:
            # IndexFlatIP with normalized vectors = cosine similarity (exact)
            return IndexIDMap(IndexFlatIP(dim))  # type: ignore[operator]
        else:
            # IndexFlatL2: Euclidean distance (legacy)
            return IndexIDMap(IndexFlatL2(dim))  # type: ignore[operator]

    def _check_ollama(self) -> None:
        try:
            ollama.list()
        except Exception:
            logger.info("Ollama not reachable — LLM/vision features disabled.")

    @property
    def audio_available(self) -> bool:
        """True when faster-whisper is loaded and ready."""
        return bool(self._audio_available)

    # ---- Input type detection -------------------------------------------------

    def detect_input_type(self, path: str) -> str:
        p = str(path).lower()
        cad_ext = _cad_suffix(path)
        if p.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus")):
            return "audio"
        if p.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
            return "video"
        if p.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")):
            return "image"
        if cad_ext in (".step", ".stp", ".iges", ".igs", ".brep"):
            return "cad_occ"
        if cad_ext in (".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf", ".off"):
            return "cad_mesh"
        if cad_ext in (
            ".prt",
            ".asm",
            ".sldprt",
            ".sldasm",
            ".ipt",
            ".iam",
            ".catpart",
            ".catproduct",
            ".x_t",
            ".x_b",
        ):
            return "cad_native"
        # Everything else → unstructured (handles PDF, DOCX, PPTX, HTML, TXT, MD, …)
        return "doc"

    # ---- Extractors -----------------------------------------------------------

    def extract_text(self, inputfile: str) -> str:
        """Document text extraction — docling preferred (better tables/layout), unstructured fallback."""
        # docling: superior table, figure, and layout extraction for PDF, DOCX, PPTX, HTML
        if _DOCLING_OK:
            try:
                conv = _DoclingConverter()  # type: ignore[operator]
                result = conv.convert(inputfile)
                md = result.document.export_to_markdown()
                if md.strip():
                    return md
            except Exception as exc:
                logger.debug("docling skipped for %s: %s", _relpath(inputfile), exc)
        if _UNSTRUCTURED_OK:
            try:
                elements = _unstructured_partition(filename=inputfile)  # type: ignore[call-arg]
                lines = [
                    e.text
                    for e in elements
                    if getattr(e, "text", None) and e.text.strip()
                ]
                text = "\n\n".join(lines)
                if text.strip():
                    return text
            except Exception as exc:
                logger.warning(
                    "unstructured failed for %s (%s) — falling back to format-specific parser.",
                    _relpath(inputfile), exc,
                )
        # Format-specific fallbacks (no LibreOffice / unstructured_inference needed)
        suffix = Path(inputfile).suffix.lower()
        if suffix == ".pdf" and _PYMUPDF_OK:
            try:
                doc = _fitz.open(inputfile)
                pages = [page.get_text() for page in doc]
                doc.close()
                text = "\n\n".join(p for p in pages if p.strip())
                if text.strip():
                    return text
            except Exception as exc:
                logger.warning("PyMuPDF failed for %s: %s", _relpath(inputfile), exc)
        if suffix == ".docx" and _MAMMOTH_OK:
            try:
                with open(inputfile, "rb") as fh:
                    result = _mammoth.extract_raw_text(fh)
                text = result.value
                if text.strip():
                    return text
            except Exception as exc:
                logger.warning("mammoth failed for %s: %s", _relpath(inputfile), exc)
        # Plain-text fallback
        try:
            return Path(inputfile).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _describe_with_vision(self, image_path: str, context: str = "") -> str:
        """Send an image to the vision model via Ollama and return a text description."""
        if ollama is None:
            return f"[Vision unavailable: {Path(image_path).name}]"
        try:
            with open(image_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode()
            prompt = (
                "Describe this image in detail, focusing on technical and visual content. "
                "Be concise but complete."
            )
            if context:
                prompt = f"Context: {context}\n{prompt}"
            resp = ollama.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            )
            return (resp["message"]["content"] or "").strip()
        except Exception as exc:
            logger.warning(
                "Vision model '%s' failed for %s: %s", self.vision_model, image_path, exc
            )
            return f"[Vision description unavailable: {Path(image_path).name}]"

    def extract_text_from_image(self, inputfile: str) -> str:
        """Image description via vision LLM (Ollama)."""
        return self._describe_with_vision(inputfile)

    def _audio_placeholder(self, inputfile: str) -> str:
        return f"[AUDIO transcription unavailable: {Path(inputfile).name}]"

    def _transcribe_audio_segments(self, inputfile: str) -> Tuple[List[Dict[str, Any]], str]:
        """Transcribe an audio/video file with faster-whisper."""
        if self.whisper_pipeline is None:
            return [], self.language or "unknown"

        language = self.language or None
        kwargs: Dict[str, Any] = {
            "language": language,
            "vad_filter": True,
            "without_timestamps": False,
        }
        if self.whisper_pipeline is not self.whisper_model:
            kwargs["batch_size"] = self._batch_size
        else:
            kwargs["beam_size"] = 5

        segments_iter, info = self.whisper_pipeline.transcribe(inputfile, **kwargs)
        segments: List[Dict[str, Any]] = []
        for seg in segments_iter:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                    "text": text,
                }
            )
        detected_language = getattr(info, "language", None) or self.language or "unknown"
        return segments, str(detected_language)

    def extract_text_from_audio(self, inputfile: str) -> str:
        """ASR via faster-whisper (language configured on init, default 'de')."""
        if not self.audio_available or self.whisper_model is None:
            logger.info("Audio transcription unavailable for %s.", _relpath(inputfile))
            return self._audio_placeholder(inputfile)
        segments, _ = self._transcribe_audio_segments(inputfile)
        text = " ".join(seg["text"] for seg in segments).strip()
        return text if text else self._audio_placeholder(inputfile)

    def _describe_frames_parallel(
        self, frame_jobs: List[Tuple[str, str]], max_workers: int = 4
    ) -> List[str]:
        """
        Describe multiple video frames in parallel via the vision model.
        Returns descriptions in the original frame order.
        Parallelism benefits when Ollama is configured with OLLAMA_NUM_PARALLEL > 1.
        """
        if not frame_jobs:
            return []
        results: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._describe_with_vision, fp, ctx): i
                for i, (fp, ctx) in enumerate(frame_jobs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"[Frame description failed: {exc}]"
        return [results[i] for i in range(len(frame_jobs))]

    def extract_text_from_video(self, inputfile: str) -> str:
        """
        Video pipeline:
          1. FFmpeg → extract 16 kHz mono WAV → faster-whisper transcript
          2. PySceneDetect → detect scene cuts → extract centre frame per scene
          3. Vision model → describe frames in parallel (ThreadPoolExecutor)
        Falls back to audio-only if PySceneDetect is not installed.
        """
        parts: List[str] = []
        with tempfile.TemporaryDirectory(prefix="veco_video_") as tmpdir:
            # 1. Audio transcription
            audio_path = os.path.join(tmpdir, "audio.wav")
            if _run_ffmpeg("-i", inputfile, "-ac", "1", "-ar", "16000", "-vn", audio_path):
                transcript = self.extract_text_from_audio(audio_path)
                if transcript and not transcript.startswith("[AUDIO"):
                    parts.append(f"[AUDIO TRANSCRIPT]\n{transcript}")

            # 2. Scene detection + parallel frame description
            if _SCENEDETECT_OK:
                try:
                    scenes = _scene_detect(inputfile, _ContentDetector())  # type: ignore[call-arg]
                    frame_jobs: List[Tuple[str, str]] = []
                    scene_meta: List[Tuple[int, str, str]] = []  # (idx, start_tc, end_tc)

                    for i, (start, end) in enumerate(scenes[:24]):  # cap at 24 scenes
                        mid = (start.get_seconds() + end.get_seconds()) / 2
                        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
                        _run_ffmpeg(
                            "-ss", f"{mid:.3f}", "-i", inputfile,
                            "-frames:v", "1", "-q:v", "2", frame_path,
                        )
                        if os.path.exists(frame_path):
                            ctx = f"Video frame at {start.get_timecode()} – {end.get_timecode()}"
                            frame_jobs.append((frame_path, ctx))
                            scene_meta.append((i, start.get_timecode(), end.get_timecode()))

                    if frame_jobs:
                        descriptions = self._describe_frames_parallel(frame_jobs)
                        scene_lines = [
                            f"Scene {meta[0] + 1} [{meta[1]} – {meta[2]}]: {desc}"
                            for meta, desc in zip(scene_meta, descriptions)
                        ]
                        parts.append("[SCENE ANALYSIS]\n" + "\n".join(scene_lines))
                except Exception as exc:
                    logger.warning("PySceneDetect failed for %s: %s", inputfile, exc)

        return "\n\n".join(parts) if parts else self._audio_placeholder(inputfile)

    # ---- CAD extractors -------------------------------------------------------

    def _extract_cad_occ(self, inputfile: str) -> str:
        """Geometry metadata from STEP/IGES files via pythonOCC."""
        ext = _cad_suffix(inputfile)
        lines = [
            f"[CAD MODEL: {Path(inputfile).name}]",
            f"Format: {ext.lstrip('.')}  |  Engine: pythonOCC",
        ]
        try:
            shape = (
                read_step_file(inputfile)   # type: ignore[name-defined]
                if ext in (".step", ".stp")
                else read_iges_file(inputfile)  # type: ignore[name-defined]
            )
            props = GProp_GProps()  # type: ignore[name-defined]
            brepgprop_VolumeProperties(shape, props)  # type: ignore[name-defined]
            lines.append(f"Volume: {props.Mass():.4f}")
            brepgprop_SurfaceProperties(shape, props)  # type: ignore[name-defined]
            lines.append(f"Surface area: {props.Mass():.4f}")
            bbox = Bnd_Box()  # type: ignore[name-defined]
            brepbndlib_Add(shape, bbox)  # type: ignore[name-defined]
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            lines.append(
                f"Bounding box: {xmax - xmin:.2f} × {ymax - ymin:.2f} × {zmax - zmin:.2f}"
            )
        except Exception as exc:
            logger.warning("pythonOCC extraction failed for %s: %s", inputfile, exc)
            lines.append(f"(Geometry extraction failed: {exc})")
        return "\n".join(lines)

    def _extract_cad_trimesh(self, inputfile: str) -> str:
        """Geometry metadata from mesh files (STL/OBJ/PLY/…) via trimesh."""
        ext = _cad_suffix(inputfile)
        lines = [
            f"[CAD MODEL: {Path(inputfile).name}]",
            f"Format: {ext.lstrip('.')}  |  Engine: trimesh",
        ]
        try:
            mesh = trimesh.load(inputfile, force="mesh")  # type: ignore[union-attr]
            lines += [
                f"Vertices: {len(mesh.vertices):,}",
                f"Faces: {len(mesh.faces):,}",
                f"Watertight: {mesh.is_watertight}",
            ]
            if mesh.is_watertight:
                lines.append(f"Volume: {mesh.volume:.6f}")
            lines.append(f"Surface area: {mesh.area:.6f}")
            dims = mesh.bounds[1] - mesh.bounds[0]
            lines.append(f"Bounding box: {dims[0]:.4f} × {dims[1]:.4f} × {dims[2]:.4f}")
        except Exception as exc:
            logger.warning("trimesh extraction failed for %s: %s", inputfile, exc)
            lines.append(f"(Geometry extraction failed: {exc})")
        return "\n".join(lines)

    def _extract_cad_native_placeholder(self, inputfile: str) -> str:
        """Metadata for native CAD files that need conversion before geometry parsing."""
        ext = _cad_suffix(inputfile)
        return "\n".join(
            [
                f"[CAD MODEL: {Path(inputfile).name}]",
                f"Format: {ext.lstrip('.')}  |  Engine: unsupported native CAD",
                "Geometry extraction skipped: convert this file to STEP/STP, IGES/IGS, STL, OBJ, PLY, or GLB for vectorization.",
            ]
        )

    def extract_text_from_cad(self, inputfile: str) -> str:
        """
        CAD pipeline:
          - STEP/IGES/BREP → pythonOCC (if installed), else trimesh
          - STL/OBJ/PLY/GLB/… → trimesh
          - Native CAD files (PRT/SLDPRT/IPT/…) → metadata only unless converted
          - Optional: render mesh → vision model visual description
        Install pythonocc-core via: conda install -c conda-forge pythonocc-core
        """
        ext = _cad_suffix(inputfile)
        is_occ_format = ext in (".step", ".stp", ".iges", ".igs", ".brep")
        is_mesh_format = ext in (".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf", ".off")

        if is_occ_format and _OCC_OK:
            geo_text = self._extract_cad_occ(inputfile)
        elif is_mesh_format and _TRIMESH_OK:
            geo_text = self._extract_cad_trimesh(inputfile)
        elif not is_occ_format and not is_mesh_format:
            return self._extract_cad_native_placeholder(inputfile)
        else:
            return (
                f"[CAD: no parser available for {Path(inputfile).name}. "
                "Install trimesh or pythonocc-core.]"
            )

        # Optional: render the mesh and describe it with vision model
        vision_desc = ""
        if _TRIMESH_OK and ollama is not None:
            with tempfile.TemporaryDirectory(prefix="veco_cad_") as tmpdir:
                render_path = os.path.join(tmpdir, "render.png")
                try:
                    mesh = trimesh.load(inputfile, force="mesh")  # type: ignore[union-attr]
                    png_bytes = mesh.scene().save_image(resolution=[512, 512], visible=False)
                    if png_bytes:
                        with open(render_path, "wb") as fh:
                            fh.write(png_bytes)
                        vision_desc = self._describe_with_vision(
                            render_path,
                            context=f"3D CAD model render: {Path(inputfile).name}",
                        )
                except Exception as exc:
                    logger.debug(
                        "CAD render skipped (rendering backend unavailable): %s", exc
                    )

        if vision_desc:
            return f"{geo_text}\n\n[VISUAL DESCRIPTION]\n{vision_desc}"
        return geo_text

    # ---- Diarization ----------------------------------------------------------

    def _diarize_audio(self, inputfile: str) -> Optional[str]:
        """
        Speaker diarization pipeline:
          1. faster-whisper transcription with segment timestamps
          2. pyannote.audio diarization (requires HF_TOKEN)
          3. Assign each transcript segment to the speaker with max overlap
        """
        if not self.audio_available or self.whisper_model is None:
            return None
        if not self.hf_token:
            logger.warning("No HF_TOKEN set — speaker diarization disabled.")
            return None
        if _PyannotePipeline is None:
            logger.warning("pyannote.audio not installed — speaker diarization disabled.")
            return None

        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        try:
            source_path = inputfile
            temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
            if self.detect_input_type(inputfile) == "video":
                temp_dir = tempfile.TemporaryDirectory(prefix="veco_diarize_")
                source_path = os.path.join(temp_dir.name, "audio.wav")
                ok = _run_ffmpeg(
                    "-i", inputfile, "-ac", "1", "-ar", "16000", "-vn", source_path
                )
                if not ok:
                    logger.warning("Could not extract audio for diarization: %s", _relpath(inputfile))
                    return None

            segments, _ = self._transcribe_audio_segments(source_path)
            if not segments:
                return None

            # Lazy-init diarization pipeline (cached on self)
            if self._diarize_pipeline is None:
                try:
                    try:
                        self._diarize_pipeline = _PyannotePipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=self.hf_token,
                        )
                    except TypeError:
                        self._diarize_pipeline = _PyannotePipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            token=self.hf_token,
                        )
                    if torch is not None and hasattr(self._diarize_pipeline, "to"):
                        self._diarize_pipeline.to(torch.device(device))
                except Exception as exc:
                    logger.warning("Diarization pipeline init failed: %s", exc)
                    return None

            diarization = self._diarize_pipeline(source_path)

            lines = [
                f"{self._speaker_for_segment(diarization, seg['start'], seg['end'])}: {seg['text']}"
                for seg in segments
                if seg.get("text", "").strip()
            ]
            return "\n".join(lines) if lines else None

        except Exception as exc:
            logger.warning("Diarization failed for %s: %s", _relpath(inputfile), exc)
            return None
        finally:
            if "temp_dir" in locals() and temp_dir is not None:
                temp_dir.cleanup()

    @staticmethod
    def _speaker_for_segment(diarization: Any, start: float, end: float) -> str:
        best_speaker = "SPEAKER_?"
        best_overlap = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap = max(0.0, min(float(end), float(turn.end)) - max(float(start), float(turn.start)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(speaker)
        return best_speaker

    # ---- CLIP: native image↔text embedding ------------------------------------

    def _embed_image_clip(self, image_path: str) -> Optional[np.ndarray]:
        """
        Embed an image directly into CLIP space (512-dim).
        Returns a normalized (1, dim) float32 array, or None on failure.
        """
        if not self._use_clip or not _PIL_OK:
            return None
        try:
            img = _PILImage.open(image_path).convert("RGB")  # type: ignore[union-attr]
            vec = self.clip_model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32).reshape(1, -1)
        except Exception as exc:
            logger.warning("CLIP image embedding failed for %s: %s", image_path, exc)
            return None

    def _add_clip_record(
        self, image_path: str, clip_vec: np.ndarray, doc_id: int, text_desc: str = ""
    ) -> None:
        """Add an image to the CLIP index and persist the record in outputdb."""
        cid = self._next_clip_id
        self._next_clip_id += 1
        cid_arr = np.array([cid], dtype=np.int64)
        self._clip_index.add_with_ids(clip_vec, cid_arr)
        rec: Dict[str, Any] = {
            "kind": "clip_embedding",
            "clip_id": cid,
            "doc_id": doc_id,
            "source": _relpath(image_path),
            "text": text_desc[:300],  # short description for display
            "clip_vector": clip_vec[0].tolist(),
        }
        self._clip_lookup[cid] = rec
        self.outputdb.append(rec)
        if self._ext_storage is not None and self.write_through:
            self._ext_storage.upsert(rec)

    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find indexed images semantically similar to the given image via CLIP.
        Requires use_clip=True on init.
        """
        if not self._use_clip:
            raise RuntimeError("CLIP not enabled. Pass use_clip=True on init.")
        vec = self._embed_image_clip(image_path)
        if vec is None:
            return []
        _, I = self._clip_index.search(vec, top_k)
        return [
            self._clip_lookup[int(r)]
            for r in I[0].tolist()
            if r != -1 and int(r) in self._clip_lookup
        ]

    def search_by_text_clip(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the image index using a text query via CLIP (cross-modal retrieval).
        Requires use_clip=True on init.

        Example: search_by_text_clip("red sports car") returns the most visually
        matching images from the indexed collection.
        """
        if not self._use_clip:
            raise RuntimeError("CLIP not enabled. Pass use_clip=True on init.")
        try:
            vec = self.clip_model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            _, I = self._clip_index.search(vec, top_k)
            return [
                self._clip_lookup[int(r)]
                for r in I[0].tolist()
                if r != -1 and int(r) in self._clip_lookup
            ]
        except Exception as exc:
            logger.warning("CLIP text search failed: %s", exc)
            return []

    # ---- Embedding & index ----------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        vecs = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._use_cosine,  # normalize for cosine similarity
        )
        return np.asarray(vecs, dtype=np.float32)

    def _reserve_vector_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.int64)
        start = self._next_vector_id
        self._next_vector_id += count
        return np.arange(start, start + count, dtype=np.int64)

    def _allocate_doc_id(self) -> int:
        doc_id = self._next_doc_id
        self._next_doc_id += 1
        return doc_id

    def _reset_in_memory_state(self) -> None:
        self.outputdb.clear()
        self.id_lookup.clear()
        self._next_vector_id = 0
        self._next_doc_id = 0
        if getattr(self, "_embedding_dim", None) is not None:
            self.faiss_index = self._create_faiss_index(self._embedding_dim)
        if self._use_clip and self._clip_dim > 0:
            self._clip_index = IndexIDMap(IndexFlatIP(self._clip_dim))  # type: ignore[operator]
            self._clip_lookup.clear()
            self._next_clip_id = 0

    def _track_existing_record(self, rec: Dict[str, Any]) -> None:
        # CLIP records go to the CLIP index, not the main text index
        if rec.get("kind") == "clip_embedding":
            if self._use_clip:
                clip_vec = rec.get("clip_vector")
                cid = rec.get("clip_id")
                if clip_vec is not None and cid is not None:
                    try:
                        arr = np.asarray(clip_vec, dtype=np.float32).reshape(1, -1)
                        if arr.shape[1] == self._clip_dim:
                            self._clip_index.add_with_ids(
                                arr, np.array([int(cid)], dtype=np.int64)
                            )
                            self._clip_lookup[int(cid)] = rec
                            if int(cid) >= self._next_clip_id:
                                self._next_clip_id = int(cid) + 1
                    except Exception as exc:
                        logger.warning("Skipped CLIP vector for clip_id=%s: %s", cid, exc)
            return  # don't add to main text index

        rid = rec.get("id")
        if rid is None:
            return
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            return

        vec = rec.get("vector")
        if isinstance(vec, (list, tuple)):
            try:
                arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
                if arr.shape[1] == self._embedding_dim:
                    self.faiss_index.add_with_ids(arr, np.array([rid_int], dtype=np.int64))
                    self.id_lookup[rid_int] = rec
            except Exception as exc:
                logger.warning("Skipped vector for record %s: %s", rid_int, exc)

        if rid_int >= self._next_vector_id:
            self._next_vector_id = rid_int + 1

        doc_id = rec.get("doc_id")
        if doc_id is not None:
            try:
                doc_int = int(doc_id)
                if doc_int >= self._next_doc_id:
                    self._next_doc_id = doc_int + 1
            except (TypeError, ValueError):
                pass

    def _add_records(
        self,
        vectors: np.ndarray,
        chunks: List[str],
        source: str,
        doc_id: int,
    ) -> None:
        assert vectors.shape[0] == len(chunks)
        ids = self._reserve_vector_ids(len(chunks))
        if ids.size == 0:
            return
        self.faiss_index.add_with_ids(vectors, ids)
        src = _relpath(source)
        for local_idx, (rid, chunk, vec) in enumerate(zip(ids.tolist(), chunks, vectors)):
            rec: Dict[str, Any] = {
                "id": int(rid),
                "doc_id": int(doc_id),
                "chunk_idx": local_idx,
                "text": chunk,
                "source": src,
                "vector": vec.tolist(),
            }
            self.outputdb.append(rec)
            self.id_lookup[int(rid)] = rec
            if self._ext_storage is not None and self.write_through:
                self._ext_storage.upsert(rec)

    # ---- Persistence ----------------------------------------------------------

    def save_database(self, json_path: Optional[str] = None) -> None:
        path = json_path or self.preload_json_path
        Path(path).write_text(
            json.dumps({"outputdb": self.outputdb}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._ext_storage is not None and not self.write_through:
            for rec in self.outputdb:
                self._ext_storage.upsert(rec)
        logger.info("JSON saved: %s", _relpath(str(path)))

    def load_database(self, json_path: Optional[str] = None) -> None:
        self._reset_in_memory_state()
        path = json_path or self.preload_json_path
        if not Path(path).exists():
            logger.info("No JSON found (%s) — starting empty.", _relpath(str(path)))
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vec_cnt = clip_cnt = total = 0
        for rec in data.get("outputdb", []):
            self.outputdb.append(rec)
            total += 1
            if rec.get("kind") == "clip_embedding":
                before = self._clip_index.ntotal if self._use_clip else 0
                self._track_existing_record(rec)
                if self._use_clip and self._clip_index.ntotal > before:
                    clip_cnt += 1
            else:
                before = self.faiss_index.ntotal
                self._track_existing_record(rec)
                if self.faiss_index.ntotal > before:
                    vec_cnt += 1
        logger.info(
            "JSON loaded: %d text vectors, %d CLIP vectors (%d total records).",
            vec_cnt, clip_cnt, total,
        )

    def _bootstrap_from_storage(self) -> None:
        self._reset_in_memory_state()
        vec_cnt = total = 0
        for rec in self._ext_storage.load_all():
            self.outputdb.append(rec)
            total += 1
            before = self.faiss_index.ntotal
            self._track_existing_record(rec)
            if self.faiss_index.ntotal > before:
                vec_cnt += 1
        logger.info("Storage loaded: %d records, %d with embeddings.", total, vec_cnt)

    def _switch_database(self, database: str) -> None:
        db = (database or "").strip()
        if getattr(self, "_active_db", None) == db:
            return
        self._reset_in_memory_state()
        if self._ext_storage is not None:
            try:
                self._ext_storage.close()
            except Exception:
                pass
            self._ext_storage = None

        lower = db.lower()
        stor_mod = _try_import_storages()

        if lower.endswith(".json") or lower == "":
            if db:
                self.preload_json_path = db
            self.load_database(self.preload_json_path)
            self._active_db = db
            return
        if lower.endswith((".sqlite", ".db")):
            if stor_mod is None:
                raise RuntimeError("SQLite requires 'storages.py'.")
            self._ext_storage = stor_mod.SqliteStorage(db_path=db)
            self._bootstrap_from_storage()
            self._active_db = db
            return
        if lower.startswith(("mongodb://", "mongodb+srv://")):
            if stor_mod is None:
                raise RuntimeError("MongoDB requires 'storages.py'.")
            self._ext_storage = stor_mod.MongoStorage(uri=db, db_name="veco_db", collection="entries")
            self._bootstrap_from_storage()
            self._active_db = db
            return
        raise ValueError(f"Unknown database format: {database!r}")

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        # Persist whatever has been ingested so far
        try:
            self.save_database()
            logger.info("Database saved on close (%d vectors).", self.faiss_index.ntotal)
        except Exception as exc:
            logger.warning("Could not save database on close: %s", exc)
        if self._ext_storage is not None:
            try:
                self._ext_storage.close()
            except Exception as exc:
                logger.warning("Storage close error: %s", exc)

    # ---- LLM / summarization --------------------------------------------------

    def build_compression_prompt(self, text: str) -> str:
        return (
            "Summarise the following text as an executive summary (5–8 bullet points).\n\n"
            f"TEXT:\n{text}\n"
        )

    def ask_llm(self, prompt: str, model: Optional[str] = None) -> str:
        if ollama is None:
            raise RuntimeError("Ollama not available.")
        resp = ollama.generate(model=model or self.default_model, prompt=prompt)
        return (resp.get("response") or "").strip()

    # ---- Ingest pipeline ------------------------------------------------------

    def vectorize(
        self,
        inputfile: str,
        use_compression: bool = False,
        model: Optional[str] = None,
        use_diarization: Optional[bool] = None,   # None = AUTO (aktiviert wenn HF_TOKEN gesetzt)
    ) -> None:
        """
        Full ingest pipeline:
          1. Detect input type
          2. Extract text / multimodal description
          3. Chunk with overlap
          4. Embed with SBERT (BAAI/bge-m3 by default)
          5. Store in FAISS + persistence backend
          6. For images: also embed with CLIP if use_clip=True
          7. Optional: LLM summary (stored as metadata, never embedded)
        """
        spinner = Spinner("Vectorizing")
        spinner.start()
        try:
            input_type = self.detect_input_type(inputfile)
            logger.info("Input type: %s  (%s)", input_type, _relpath(inputfile))

            raw_text = ""

            # Auto-detect diarization: enable when HF token + audio is available
            if use_diarization is None and input_type in {"audio", "video"}:
                use_diarization = bool(self.hf_token) and self.audio_available

            # Diarization path (speaker-tagged transcript)
            if use_diarization and input_type in {"audio", "video"}:
                dia_text = self._diarize_audio(inputfile)
                if dia_text:
                    raw_text = dia_text

            # Standard extraction path
            if not raw_text:
                if input_type == "doc":
                    raw_text = self.extract_text(inputfile)
                elif input_type == "image":
                    raw_text = self.extract_text_from_image(inputfile)
                elif input_type == "audio":
                    raw_text = self.extract_text_from_audio(inputfile)
                elif input_type == "video":
                    raw_text = self.extract_text_from_video(inputfile)
                elif input_type in ("cad_occ", "cad_mesh", "cad_native"):
                    raw_text = self.extract_text_from_cad(inputfile)

            raw_text = (raw_text or "").strip()
            if not raw_text:
                logger.warning("No text extracted from %s.", _relpath(inputfile))
                return

            chunks = chunk_text(raw_text, chunk_chars=1800, overlap_chars=200) or [raw_text]
            vectors = self.embed_texts(chunks)
            doc_id = self._allocate_doc_id()
            self._add_records(vectors, chunks, source=str(inputfile), doc_id=doc_id)

            # CLIP embedding for images (native multimodal, no text description needed)
            if input_type == "image" and self._use_clip:
                clip_vec = self._embed_image_clip(inputfile)
                if clip_vec is not None:
                    self._add_clip_record(
                        inputfile, clip_vec, doc_id, text_desc=raw_text[:300]
                    )
                    logger.info("CLIP embedding added for %s.", _relpath(inputfile))

            if use_compression:
                try:
                    summary = self.ask_llm(
                        self.build_compression_prompt(raw_text), model or self.default_model
                    )
                except Exception as exc:
                    logger.warning("Summarization failed: %s", exc)
                    summary = None
                if summary:
                    meta: Dict[str, Any] = {
                        "id": int(10_000_000_000 + doc_id),
                        "doc_id": int(doc_id),
                        "chunk_idx": -1,
                        "kind": "doc_summary",
                        "text": "",
                        "summary": summary,
                        "source": _relpath(str(inputfile)),
                    }
                    self.outputdb.append(meta)
                    if self._ext_storage is not None and self.write_through:
                        self._ext_storage.upsert(meta)

            # Persist after every file so a crash never loses more than one file's work
            if self.write_through:
                self.save_database()
        finally:
            spinner.stop()

    # ---- Retrieval / RAG ------------------------------------------------------

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = self.embed_texts([query])
        if self.faiss_index.ntotal == 0:
            return []
        _, I = self.faiss_index.search(qv, top_k)
        return [
            self.id_lookup[int(rid)]
            for rid in I[0].tolist()
            if rid != -1 and int(rid) in self.id_lookup
        ]

    def query_with_context(
        self,
        question: str,
        top_k: int = 5,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        ctx = self.retrieve_context(question, top_k=top_k)
        response: Dict[str, Any] = {"question": question, "contexts": ctx}
        if include_summary and ctx:
            doc_ids = {c.get("doc_id") for c in ctx if c.get("doc_id") is not None}
            summaries = [
                r for r in self.outputdb
                if r.get("kind") == "doc_summary" and r.get("doc_id") in doc_ids
            ]
            if summaries:
                response["summaries"] = summaries
        return response

    def _build_rag_prompt(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        ctx_text = "\n\n".join(c.get("text", "") for c in contexts if c.get("text"))
        sum_text = ""
        if summaries:
            parts = [s.get("summary", "") for s in summaries if s.get("summary")]
            if parts:
                sum_text = "\n\nSUMMARY (document level):\n" + "\n".join(parts)
        return (
            "Answer the question strictly based on the following context.\n"
            'If the answer is not in the context, respond: "Not available in the provided context."\n\n'
            f"CONTEXT:\n{ctx_text}{sum_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER (concise, German):\n"
        )

    def query(
        self,
        database: str,
        question: str,
        llm_model: str,
        top_k: int = 5,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """End-to-end RAG query: load DB → retrieve context → Ollama answer."""
        if ollama is None:
            raise RuntimeError("Ollama not available.")
        self._switch_database(database)
        contexts = self.retrieve_context(question, top_k=top_k)
        summaries: List[Dict[str, Any]] = []
        if include_summary and contexts:
            doc_ids = {c.get("doc_id") for c in contexts if c.get("doc_id") is not None}
            summaries = [
                r for r in self.outputdb
                if r.get("kind") == "doc_summary" and r.get("doc_id") in doc_ids
            ]
        prompt = self._build_rag_prompt(question, contexts, summaries)
        try:
            resp = ollama.generate(model=llm_model, prompt=prompt)
            answer = (resp.get("response") or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc
        result: Dict[str, Any] = {
            "question": question,
            "model": llm_model,
            "answer": answer,
            "contexts": contexts,
            "sources": list({c.get("source") for c in contexts if c.get("source")}),
        }
        if include_summary and summaries:
            result["summaries"] = summaries
        return result


# ---- CLI ---------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="veco-ai — Multimodal Vectorize & RAG")
    ap.add_argument("input", nargs="?", help="File to ingest (doc/image/audio/video/CAD)")
    ap.add_argument("--compress", action="store_true", help="Generate and store a document summary")
    ap.add_argument("--json", default="vector_db.json", help="JSON database file")
    ap.add_argument("--use-sqlite", default=None, help="Path to SQLite DB")
    ap.add_argument("--use-mongo", default=None, help="MongoDB URI")
    ap.add_argument("--mongo-db", default="veco_db")
    ap.add_argument("--mongo-col", default="entries")
    ap.add_argument(
        "--diarize", default=None, choices=["true", "false"],
        help="Force speaker diarization on/off (default: AUTO for audio/video)",
    )
    ap.add_argument(
        "--vision-model", default="gemma4",
        help="Ollama model for image/video/CAD descriptions (default: gemma4)",
    )
    ap.add_argument(
        "--llm", default="gemma3:12b",
        help="Ollama LLM for RAG answers (default: gemma3:12b)",
    )
    ap.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="sentence-transformers model for text embedding (default: BAAI/bge-m3)",
    )
    ap.add_argument(
        "--index-type", default="flat", choices=["flat", "hnsw"],
        help="FAISS index type: flat (exact) or hnsw (approx ANN, faster at scale)",
    )
    ap.add_argument(
        "--use-clip", action="store_true",
        help="Enable CLIP for native image-text search (requires use on images)",
    )
    ap.add_argument(
        "--embedder-backend", default="sbert", choices=["sbert", "ollama"],
        help="Embedding backend: sbert (sentence-transformers, default) or ollama (unified local API)",
    )
    ap.add_argument(
        "--ollama-embed-model", default="nomic-embed-text",
        help="Ollama model for embeddings when --embedder-backend=ollama (default: nomic-embed-text)",
    )
    ap.add_argument(
        "--search-image", default=None,
        help="Search indexed images visually similar to this image (requires --use-clip)",
    )
    ap.add_argument(
        "--search-text-clip", default=None,
        help="Find images matching this text query via CLIP (requires --use-clip)",
    )
    args = ap.parse_args()

    storage_kind = storage_kwargs = None
    if args.use_sqlite:
        storage_kind, storage_kwargs = "sqlite", {"db_path": args.use_sqlite}
    elif args.use_mongo:
        storage_kind = "mongo"
        storage_kwargs = {
            "uri": args.use_mongo,
            "db_name": args.mongo_db,
            "collection": args.mongo_col,
        }

    veco = Vectorize(
        default_model=args.llm,
        vision_model=args.vision_model,
        embedding_model=args.embedding_model,
        preload_json_path=args.json,
        storage_kind=storage_kind,
        storage_kwargs=storage_kwargs,
        index_type=args.index_type,
        use_clip=args.use_clip,
        embedder_backend=args.embedder_backend,
        ollama_embed_model=args.ollama_embed_model,
    )

    diarize_flag: Optional[bool] = (
        None if args.diarize is None else args.diarize == "true"
    )

    if args.input:
        veco.vectorize(args.input, use_compression=args.compress, use_diarization=diarize_flag)
        veco.save_database(args.json)
        res = veco.query_with_context("What is this about?", top_k=5)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.search_image:
        results = veco.search_by_image(args.search_image, top_k=5)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.search_text_clip:
        results = veco.search_by_text_clip(args.search_text_clip, top_k=5)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        print("No input provided. Examples:")
        print("  python veco_ai.py report.pdf --compress")
        print("  python veco_ai.py interview.wav --diarize true")
        print("  python veco_ai.py meeting.mp4")
        print("  python veco_ai.py assembly.step")
        print("  python veco_ai.py photo.jpg --use-clip")
        print("  python veco_ai.py --search-text-clip 'rotes Auto' --use-clip")
        print("  python veco_ai.py --embedding-model BAAI/bge-m3 --index-type hnsw report.pdf")

    veco.close()


if __name__ == "__main__":
    main()
