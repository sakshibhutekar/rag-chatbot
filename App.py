"""
app.py
------
DocuMind — Local RAG Chatbot for PDF Documents
Main Streamlit application.

Run with:
    streamlit run app.py
"""

import traceback

import streamlit as st

from pdf_upload_handler import (
    delete_document_from_index,
    get_indexed_document_count,
    get_indexed_sources,
    process_uploaded_pdf,
)
from rag_core import (
    RAGResponse,
    OLLAMA_MODEL,
    TOP_K,
    check_ollama_connection,
    retrieve_relevant_chunks,
    build_prompt,
    generate_answer_streaming,
)

# ── Easter egg ────────────────────────────────────────────────────────────────
try:
    import antigravity  # noqa: F401
except ImportError:
    pass


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — Local RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all required session state keys on first run."""
    defaults = {
        "messages": [],
        "rag_responses": {},       # keyed by message index -> RAGResponse
        "last_rag_response": None, # kept for backward compat
        "indexed_files": set(),
        "ollama_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stChatInput textarea { font-size: 15px; }
        .sidebar-section-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888;
            margin-top: 12px;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: render source chunks ──────────────────────────────────────────────
# Defined here — BEFORE any call site — to avoid NameError on first render
def _render_sources(response: RAGResponse, show_scores: bool = False) -> None:
    """Render retrieved source chunks as an expandable section."""
    if not response or not response.retrieved_chunks:
        return

    with st.expander(
        f"📚 Retrieved context — {len(response.retrieved_chunks)} chunk(s) "
        f"from {len(response.unique_sources)} source(s)",
        expanded=False,
    ):
        for j, chunk in enumerate(response.retrieved_chunks, start=1):
            score_badge = (
                f" · relevance: **{chunk.relevance_score:.2%}**" if show_scores else ""
            )
            st.markdown(
                f"**Chunk {j}** · `{chunk.source}` (chunk #{chunk.chunk_index}){score_badge}"
            )
            st.text_area(
                label=f"chunk_{j}_text",
                value=chunk.text,
                height=120,
                disabled=True,
                label_visibility="collapsed",
                key=f"source_chunk_{j}_{id(response)}_{j}",
            )
            if j < len(response.retrieved_chunks):
                st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("Local RAG · Offline-first · Open-source")
    st.divider()

    # System status
    st.markdown('<p class="sidebar-section-title">System Status</p>', unsafe_allow_html=True)

    if st.button("🔄 Check Ollama", use_container_width=True):
        st.session_state.ollama_status = None

    if st.session_state.ollama_status is None:
        with st.spinner("Checking Ollama..."):
            _ok, _msg = check_ollama_connection(OLLAMA_MODEL)
            st.session_state.ollama_status = (_ok, _msg)

    ollama_ok, ollama_msg = st.session_state.ollama_status
    if ollama_ok:
        st.success(ollama_msg, icon="✅")
    else:
        st.error(ollama_msg, icon="❌")
        st.info(
            f"**To fix:** Open a terminal and run:\n\n"
            f"```\nollama pull {OLLAMA_MODEL}\n```\n\n"
            "Then make sure the Ollama app is running.",
            icon="💡",
        )

    st.divider()

    # Knowledge base stats
    st.markdown('<p class="sidebar-section-title">Knowledge Base</p>', unsafe_allow_html=True)

    chunk_count = get_indexed_document_count()
    indexed_sources = get_indexed_sources()

    col1, col2 = st.columns(2)
    col1.metric("Chunks", f"{chunk_count:,}")
    col2.metric("Documents", len(indexed_sources))

    if indexed_sources:
        with st.expander("📂 Indexed files", expanded=False):
            for src in indexed_sources:
                col_name, col_btn = st.columns([4, 1])
                col_name.markdown(f"`{src}`")
                if col_btn.button("🗑️", key=f"del_{src}", help=f"Remove {src} from index"):
                    deleted = delete_document_from_index(src)
                    st.session_state.indexed_files.discard(src)
                    st.toast(f"Removed {deleted} chunks from '{src}'", icon="🗑️")
                    st.rerun()

    st.divider()

    # PDF upload
    st.markdown('<p class="sidebar-section-title">Upload Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add them to the knowledge base.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.indexed_files
        ]

        if new_files:
            if st.button(
                f"📥 Index {len(new_files)} file{'s' if len(new_files) > 1 else ''}",
                use_container_width=True,
                type="primary",
            ):
                for uploaded_file in new_files:
                    with st.status(
                        f"Processing **{uploaded_file.name}**...",
                        expanded=True,
                        state="running",
                    ) as status_box:
                        try:
                            file_bytes = uploaded_file.read()
                            pipeline = process_uploaded_pdf(file_bytes, uploaded_file.name)
                            total_chunks = 0
                            try:
                                while True:
                                    status_msg = next(pipeline)
                                    st.write(status_msg)
                            except StopIteration as exc:
                                total_chunks = exc.value

                            st.session_state.indexed_files.add(uploaded_file.name)
                            status_box.update(
                                label=f"✅ {uploaded_file.name} — {total_chunks} chunks indexed",
                                state="complete",
                                expanded=False,
                            )
                        except ValueError as exc:
                            status_box.update(label=f"❌ {uploaded_file.name}", state="error")
                            st.error(str(exc))
                        except Exception as exc:  # noqa: BLE001
                            status_box.update(label=f"❌ {uploaded_file.name}", state="error")
                            st.error(f"Unexpected error: {exc}")
                            st.code(traceback.format_exc(), language="text")

                st.rerun()
        else:
            st.success("All uploaded files are already indexed.", icon="✅")

    st.divider()

    # Retrieval settings
    st.markdown('<p class="sidebar-section-title">Retrieval Settings</p>', unsafe_allow_html=True)
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="How many text chunks are retrieved per question. More = more context but slower.",
    )
    show_scores = st.toggle("Show relevance scores", value=False)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_rag_response = None
        st.session_state.rag_responses = {}
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("💬 Ask Your Documents")

if chunk_count == 0:
    st.info(
        "**No documents indexed yet.** Upload one or more PDF files using the sidebar to get started.",
        icon="📄",
    )
elif not ollama_ok:
    st.warning(
        "Ollama is not running. Answers cannot be generated until the model is available.",
        icon="⚠️",
    )

# Chat history display
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for every assistant message that has a stored RAGResponse
        if message["role"] == "assistant":
            rag_resp = st.session_state.rag_responses.get(i)
            if rag_resp is not None:
                _render_sources(rag_resp, show_scores)


# Chat input
if query := st.chat_input(
    "Ask a question about your documents...",
    disabled=(chunk_count == 0),
):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_relevant_chunks(query, top_k=top_k)
            prompt = build_prompt(query, chunks)

        answer = ""

        if not chunks:
            answer = (
                "I could not find any relevant information in the uploaded documents. "
                "Please make sure you have uploaded and indexed at least one PDF."
            )
            st.markdown(answer)
        else:
            answer_placeholder = st.empty()
            full_answer: list[str] = []

            try:
                for token in generate_answer_streaming(prompt, OLLAMA_MODEL):
                    full_answer.append(token)
                    answer_placeholder.markdown("".join(full_answer) + "▌")

                answer_placeholder.markdown("".join(full_answer))
                answer = "".join(full_answer)

            except (ConnectionError, RuntimeError) as exc:
                answer = f"⚠️ **Error:** {exc}"
                st.error(answer)
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ **Unexpected error:** {exc}"
                st.error(answer)
                st.code(traceback.format_exc())

        response = RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            query=query,
            model_used=OLLAMA_MODEL,
            num_chunks_retrieved=len(chunks),
        )
        st.session_state.last_rag_response = response
        _render_sources(response, show_scores)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Store RAGResponse keyed by the index of this new assistant message
    assistant_msg_index = len(st.session_state.messages) - 1
    st.session_state.rag_responses[assistant_msg_index] = response


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🧠 **DocuMind** · Local RAG · "
    f"Model: `{OLLAMA_MODEL}` · "
    "Embeddings: `all-MiniLM-L6-v2` · "
    "Vector DB: ChromaDB · "
    "Built with Streamlit"
)