"""
pdf_upload_handler.py
---------------------
Handles all PDF ingestion logic for DocuMind:
  - In-memory PDF text extraction via pypdf
  - Intelligent text chunking with configurable size and overlap
  - Embedding generation via sentence-transformers (all-MiniLM-L6-v2)
  - Pure-Python vector store using FAISS (no C++ compilation needed)
  - Persistent storage via JSON + FAISS binary files

No temporary files are written to disk during processing (except the
final index save).
"""

import hashlib
import io
import json
import logging
import os
import re
from typing import Generator

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 80
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384          # output dimension for all-MiniLM-L6-v2
INDEX_DIR: str = "./faiss_index"
INDEX_FILE: str = os.path.join(INDEX_DIR, "index.faiss")
META_FILE: str = os.path.join(INDEX_DIR, "metadata.json")


# ── In-memory singletons ──────────────────────────────────────────────────────
# _metadata is a list where index i corresponds to FAISS vector row i.
# Each entry: {"id": str, "source": str, "chunk_index": int, "text": str}

_embedding_model: SentenceTransformer | None = None
_faiss_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] = []


# ── Singleton helpers ─────────────────────────────────────────────────────────

def get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (loaded once per session)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_index_and_meta() -> tuple:
    """
    Return the (index, metadata) pair.
    Loads from disk on first call; creates a fresh empty index if none exists.
    """
    global _faiss_index, _metadata

    if _faiss_index is not None:
        return _faiss_index, _metadata

    os.makedirs(INDEX_DIR, exist_ok=True)

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        logger.info("Loading FAISS index from %s", INDEX_FILE)
        try:
            _faiss_index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
            logger.info("Loaded %d vectors from disk.", _faiss_index.ntotal)
            return _faiss_index, _metadata
        except Exception as exc:
            logger.warning("Could not load index from disk: %s — starting fresh.", exc)

    logger.info("Creating new FAISS index (dim=%d).", EMBEDDING_DIM)
    _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    _metadata = []
    return _faiss_index, _metadata


def _save_index() -> None:
    """Persist the FAISS index and metadata to disk."""
    index, meta = _get_index_and_meta()
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    logger.info("Saved %d vectors to disk.", index.ntotal)


# ── Core pipeline steps ───────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Extract all text from a PDF supplied as raw bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw PDF file content.
    filename : str
        Original filename, used only for log messages.

    Returns
    -------
    str
        Concatenated text from every page.

    Raises
    ------
    ValueError
        If the PDF contains no extractable text.
    """
    logger.info("Extracting text from '%s' (%d bytes).", filename, len(file_bytes))
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = _clean_extracted_text(text)
            if text.strip():
                pages_text.append(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract text from page %d: %s", page_num, exc)

    full_text = "\n".join(pages_text)

    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The file may be a scanned/image-based PDF. OCR is not currently supported."
        )

    logger.info(
        "Extracted %d characters from %d pages of '%s'.",
        len(full_text), len(pages_text), filename,
    )
    return full_text


def _clean_extracted_text(text: str) -> str:
    """Normalise whitespace and remove common PDF extraction artefacts."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

    Parameters
    ----------
    text : str
        Full document text.
    chunk_size : int
        Target length of each chunk in characters.
    overlap : int
        Characters shared between consecutive chunks.

    Returns
    -------
    list[str]
        Ordered list of text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            break_pos = text.rfind(" ", start, end)
            if break_pos > start:
                end = break_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, start + 1)  # guard against infinite loop

    logger.debug("Split text into %d chunks (size=%d, overlap=%d).", len(chunks), chunk_size, overlap)
    return chunks


def _make_chunk_id(filename: str, chunk_index: int, chunk_text_content: str) -> str:
    """Generate a deterministic content-addressable ID for a chunk."""
    fingerprint = f"{filename}::{chunk_index}::{chunk_text_content[:100]}"
    return hashlib.md5(fingerprint.encode()).hexdigest()  # noqa: S324


def _remove_by_source(filename: str, index: faiss.IndexFlatIP, meta: list) -> int:
    """
    Remove all vectors whose source == filename.

    FAISS IndexFlatIP has no in-place delete, so we rebuild without them.
    Returns the count of removed entries.
    """
    keep_indices = [i for i, m in enumerate(meta) if m["source"] != filename]
    removed = len(meta) - len(keep_indices)

    if removed == 0:
        return 0

    if not keep_indices:
        index.reset()
        meta.clear()
    else:
        kept_vectors = np.vstack(
            [index.reconstruct(i) for i in keep_indices]
        ).astype("float32")
        index.reset()
        index.add(kept_vectors)
        kept_meta = [meta[i] for i in keep_indices]
        meta.clear()
        meta.extend(kept_meta)

    logger.info("Removed %d chunks for '%s'. %d remain.", removed, filename, len(meta))
    return removed


def embed_and_index_chunks(
    chunks: list[str],
    filename: str,
    model: SentenceTransformer,
) -> int:
    """
    Embed chunks and upsert them into the FAISS index.

    Existing entries for the same filename are replaced (upsert semantics).

    Returns
    -------
    int
        Number of chunks indexed.
    """
    if not chunks:
        logger.warning("No chunks to index for '%s'.", filename)
        return 0

    index, meta = _get_index_and_meta()

    # Remove stale entries for this file
    _remove_by_source(filename, index, meta)

    # Embed in batches — normalize_embeddings=True makes inner product == cosine sim
    BATCH_SIZE = 64
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        vecs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype("float32")

    new_meta = [
        {
            "id": _make_chunk_id(filename, i, chunk),
            "source": filename,
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]

    index.add(embeddings)
    meta.extend(new_meta)

    _save_index()
    logger.info("Indexed %d chunks from '%s'.", len(chunks), filename)
    return len(chunks)


# ── Public query API ──────────────────────────────────────────────────────────

def query_index(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Search the FAISS index for the top-k most similar chunks.

    Parameters
    ----------
    query_embedding : list[float]
        L2-normalised query vector (length == EMBEDDING_DIM).
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict: text, source, chunk_index, score (cosine similarity 0–1).
    """
    index, meta = _get_index_and_meta()
    if index.ntotal == 0:
        return []

    effective_k = min(top_k, index.ntotal)
    query_vec = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vec, effective_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = meta[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


# ── Public entry point ────────────────────────────────────────────────────────

def process_uploaded_pdf(
    file_bytes: bytes,
    filename: str,
) -> Generator[str, None, int]:
    """
    Full ingestion pipeline for a single PDF file.

    Yields status strings for Streamlit progress display.
    Returns total chunks indexed via StopIteration.value.
    """
    yield f"📄 Extracting text from **{filename}**..."
    full_text = extract_text_from_pdf(file_bytes, filename)
    yield f"✅ Extracted **{len(full_text):,}** characters."

    yield f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})..."
    chunks = chunk_text(full_text)
    yield f"✅ Created **{len(chunks)}** chunks."

    yield "🔢 Generating embeddings and indexing into FAISS..."
    model = get_embedding_model()
    num_indexed = embed_and_index_chunks(chunks, filename, model)
    yield f"✅ Indexed **{num_indexed}** chunks from **{filename}**."

    return num_indexed


def get_indexed_document_count() -> int:
    """Return the total number of chunks in the index."""
    index, _ = _get_index_and_meta()
    return index.ntotal


def get_indexed_sources() -> list[str]:
    """Return a deduplicated sorted list of indexed source filenames."""
    _, meta = _get_index_and_meta()
    return sorted({m["source"] for m in meta if "source" in m})


def delete_document_from_index(filename: str) -> int:
    """Remove all chunks for *filename*. Returns count of deleted chunks."""
    index, meta = _get_index_and_meta()
    removed = _remove_by_source(filename, index, meta)
    if removed:
        _save_index()
    return removed