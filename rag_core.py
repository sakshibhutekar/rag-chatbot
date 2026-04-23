"""
rag_core.py
-----------
Retrieval-Augmented Generation core for DocuMind.

Responsibilities:
  - Embed an incoming user query using the same model used at index time
  - Retrieve the top-k most semantically similar chunks from ChromaDB
  - Construct a grounded, hallucination-resistant prompt
  - Stream the response from a locally running Ollama / Llama 3.2 model
  - Return structured results (answer + source citations) to the caller
"""

import logging
from dataclasses import dataclass, field

import ollama
from sentence_transformers import SentenceTransformer

from pdf_upload_handler import get_embedding_model, get_indexed_document_count, query_index

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_MODEL: str = "llama3.2:3b"   # model tag as shown in `ollama list`
TOP_K: int = 5                       # number of chunks to retrieve per query
MAX_CONTEXT_CHARS: int = 3000        # hard cap on total context fed to the LLM


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its metadata and relevance score."""
    text: str
    source: str
    chunk_index: int
    distance: float          # cosine distance (lower = more similar)

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance to a 0–1 relevance score (1 = perfect match)."""
        return round(max(0.0, 1.0 - self.distance), 4)


@dataclass
class RAGResponse:
    """Complete response object returned by ``answer_question``."""
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    model_used: str = OLLAMA_MODEL
    num_chunks_retrieved: int = 0

    @property
    def has_context(self) -> bool:
        return len(self.retrieved_chunks) > 0

    @property
    def unique_sources(self) -> list[str]:
        return sorted({c.source for c in self.retrieved_chunks})


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K,
    model: SentenceTransformer | None = None,
) -> list[RetrievedChunk]:
    """
    Embed *query* and perform a cosine similarity search against the FAISS index.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    top_k : int
        Maximum number of chunks to retrieve.
    model : SentenceTransformer, optional
        Embedding model instance; uses the shared singleton if not provided.

    Returns
    -------
    list[RetrievedChunk]
        Retrieved chunks sorted by relevance (most relevant first).
        Returns an empty list if the knowledge base is empty.
    """
    if get_indexed_document_count() == 0:
        logger.warning("FAISS index is empty — no chunks to retrieve.")
        return []

    if model is None:
        model = get_embedding_model()

    logger.info("Embedding query and retrieving top-%d chunks...", top_k)
    # normalize_embeddings=True so inner product == cosine similarity
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = query_index(query_embedding, top_k)

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            text=r["text"],
            source=r.get("source", "unknown"),
            chunk_index=r.get("chunk_index", -1),
            distance=1.0 - r["score"],   # convert cosine sim → cosine distance
        )
        for r in results
    ]

    logger.info("Retrieved %d chunks. Top relevance: %.4f", len(chunks), chunks[0].relevance_score if chunks else 0)
    return chunks


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Construct a grounded RAG prompt that instructs the model to answer
    strictly from the provided context.

    The prompt uses a clear system instruction + context block + question
    structure.  An explicit instruction to acknowledge missing information
    is the primary hallucination-mitigation mechanism.

    Parameters
    ----------
    query : str
        The user's question.
    chunks : list[RetrievedChunk]
        Retrieved context chunks (most relevant first).

    Returns
    -------
    str
        A fully formatted prompt string ready to pass to Ollama.
    """
    if not chunks:
        return (
            "You are a helpful assistant.\n\n"
            "No documents have been uploaded to the knowledge base yet, "
            "so you have no context to draw from.\n\n"
            f"User question: {query}\n\n"
            "Please let the user know they need to upload a PDF document first."
        )

    # Build the context block, respecting the character budget
    context_parts: list[str] = []
    total_chars = 0
    for i, chunk in enumerate(chunks, start=1):
        entry = f"[Source {i}: {chunk.source}, chunk {chunk.chunk_index}]\n{chunk.text}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            logger.debug("Context budget reached at chunk %d — truncating.", i)
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise, helpful assistant that answers questions strictly based on the provided document context.

CONTEXT FROM UPLOADED DOCUMENTS:
{context_block}

INSTRUCTIONS:
- Answer the question using ONLY the information present in the context above.
- If the context contains a clear answer, provide it in a well-structured, readable format.
- If the context is partially relevant, use what is available and note any gaps.
- If the context does not contain enough information to answer the question, say exactly:
  "I could not find sufficient information in the uploaded documents to answer this question."
- Do NOT use any knowledge from outside the provided context.
- Cite the source filename(s) when referencing specific information (e.g., "According to report.pdf, ...").
- Keep your answer concise and factual.

QUESTION: {query}

ANSWER:"""

    return prompt


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer_streaming(prompt: str, model: str = OLLAMA_MODEL):
    """
    Stream the model's response token by token from Ollama.

    Parameters
    ----------
    prompt : str
        The fully constructed prompt.
    model : str
        Ollama model tag to use.

    Yields
    ------
    str
        Incremental text chunks as they are generated.

    Raises
    ------
    ConnectionError
        If the Ollama server is not reachable.
    RuntimeError
        If the requested model is not available locally.
    """
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            # Support both object-style (new) and dict-style (old) Ollama responses
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            else:
                token = (chunk.get("message") or {}).get("content") or ""
            if token:
                yield token

    except ollama.ResponseError as exc:
        if "model" in str(exc).lower() and "not found" in str(exc).lower():
            raise RuntimeError(
                f"Model '{model}' is not available. "
                f"Run `ollama pull {model}` in your terminal to download it."
            ) from exc
        raise

    except Exception as exc:
        if "connection" in str(exc).lower() or "refused" in str(exc).lower():
            raise ConnectionError(
                "Cannot reach the Ollama server. "
                "Make sure Ollama is running (`ollama serve` or open the Ollama app)."
            ) from exc
        raise


def generate_answer(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Non-streaming wrapper around ``generate_answer_streaming``.

    Collects all tokens and returns the complete answer string.
    Useful for testing or contexts where streaming is not needed.
    """
    return "".join(generate_answer_streaming(prompt, model))


# ── Public entry point ────────────────────────────────────────────────────────

def answer_question(
    query: str,
    top_k: int = TOP_K,
    stream: bool = False,
) -> RAGResponse:
    """
    End-to-end RAG pipeline: retrieve → prompt → generate → return.

    For Streamlit streaming, set *stream=False* and use
    ``stream_answer_question`` instead.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    top_k : int
        Number of chunks to retrieve from ChromaDB.
    stream : bool
        If True, the ``answer`` field will be empty and you should use
        ``generate_answer_streaming`` directly for token-level streaming.

    Returns
    -------
    RAGResponse
        Dataclass containing the answer, retrieved chunks, and metadata.
    """
    if not query.strip():
        return RAGResponse(answer="Please enter a question.", query=query)

    # Step 1: Retrieve
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    # Step 2: Build prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate
    if stream:
        answer = ""   # Caller should use generate_answer_streaming directly
    else:
        answer = generate_answer(prompt)

    return RAGResponse(
        answer=answer,
        retrieved_chunks=chunks,
        query=query,
        model_used=OLLAMA_MODEL,
        num_chunks_retrieved=len(chunks),
    )


def stream_answer_question(query: str, top_k: int = TOP_K) -> tuple:
    """
    Streaming variant of ``answer_question`` designed for Streamlit's
    token loop.

    Returns
    -------
    tuple[list[RetrievedChunk], Generator]
        (chunks, token_generator) — iterate the generator for tokens,
        then use chunks to build a RAGResponse afterwards.
    """
    chunks = retrieve_relevant_chunks(query, top_k=top_k)
    prompt = build_prompt(query, chunks)
    return chunks, generate_answer_streaming(prompt)


# ── Health check ──────────────────────────────────────────────────────────────

def check_ollama_connection(model: str = OLLAMA_MODEL) -> tuple[bool, str]:
    """
    Verify that Ollama is running and the required model is available.

    Returns
    -------
    tuple[bool, str]
        (True, success_message) or (False, error_message)
    """
    try:
        models = ollama.list()
        # Support both dict-style (old) and object-style (new) Ollama client responses
        model_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        available = [
            m["name"] if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", ""))
            for m in model_list
        ]
        # Match on model name prefix (e.g. "llama3.2:3b" matches "llama3.2:3b-instruct-q4_K_M")
        if not any(model.split(":")[0] in m for m in available):
            return False, (
                f"Model '{model}' not found. Available models: {available or 'none'}. "
                f"Run `ollama pull {model}` to download it."
            )
        return True, f"Ollama is running. Model '{model}' is available. ✅"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama is not reachable: {exc}. Make sure Ollama is running."