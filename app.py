"""
CodeRAG — chat with any public GitHub repository using a real
RAG (Retrieval-Augmented Generation) pipeline.

Run with:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from utils.validation import parse_github_url, check_repository_exists_and_public, ValidationError
from utils.repository_utils import make_collection_name, format_size
from rag.github_loader import clone_repository
from rag.file_processor import process_repository
from rag.code_chunker import chunk_repository
from rag.embeddings import EmbeddingModel, DEFAULT_MODEL_NAME
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import generate_answer, GenerationError

load_dotenv()

st.set_page_config(page_title="CodeRAG", page_icon="🧠", layout="wide")

EXAMPLE_QUESTIONS = [
    "What does this project do?",
    "Explain the project architecture.",
    "Where is authentication implemented?",
    "How does the frontend communicate with the backend?",
    "Where is the database connection?",
    "Explain the login flow.",
    "What technologies are used?",
    "Which file contains the main API?",
    "Explain this project like I'm a beginner.",
]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def _init_state():
    defaults = {
        "repo_full_name": None,
        "collection_name": None,
        "stats": None,
        "messages": [],           # [{role, content, retrieved(list), sources(list)}]
        "top_k": 5,
        "similarity_threshold": 0.2,
        "pending_question": None,
        "ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


@st.cache_resource(show_spinner=False)
def _get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner=False)
def _get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(DEFAULT_MODEL_NAME)


vector_store = _get_vector_store()
embedding_model = _get_embedding_model()
retriever = Retriever(vector_store, embedding_model)


# --------------------------------------------------------------------------
# Ingestion pipeline
# --------------------------------------------------------------------------

def analyze_repository(url: str):
    try:
        identity = parse_github_url(url)
        check_repository_exists_and_public(identity)
    except ValidationError as e:
        st.error(e.user_message)
        return

    status = st.status("Analyzing repository...", expanded=True)
    cloned = None
    try:
        status.write("1. Fetching repository")
        cloned = clone_repository(identity)
        status.write("✓ 1. Fetching repository")

        collection_name = make_collection_name(identity.owner, identity.repo, cloned.commit_sha)

        if vector_store.collection_exists(collection_name):
            status.write("✓ 2-5. Found cached index — skipping re-embedding")
            existing_count = vector_store.count(collection_name)
            files_processed = None  # unknown from cache alone
            chunks_created = existing_count
        else:
            status.write("2. Extracting files")
            processed_files = process_repository(cloned.local_path)
            if not processed_files:
                status.update(label="No usable files found", state="error")
                st.error(
                    "❌ This repository doesn't contain any supported text/code files "
                    "to index."
                )
                return
            status.write(f"✓ 2. Extracting files ({len(processed_files)} files)")

            status.write("3. Creating chunks")
            chunks = chunk_repository(processed_files, identity.full_name)
            if not chunks:
                status.update(label="No chunkable content found", state="error")
                st.error("❌ Could not extract any chunkable content from this repository.")
                return
            status.write(f"✓ 3. Creating chunks ({len(chunks)} chunks)")

            status.write("4. Generating embeddings & 5. Building vector index")
            progress_bar = st.progress(0.0)

            def _on_progress(done, total):
                progress_bar.progress(done / total if total else 1.0)

            vector_store.index_chunks(
                collection_name, chunks, embedding_model, progress_callback=_on_progress
            )
            status.write("✓ 4. Generating embeddings")
            status.write("✓ 5. Building vector index")

            files_processed = len(processed_files)
            chunks_created = len(chunks)

        status.write("✓ 6. RAG system ready")
        status.update(label="RAG system ready", state="complete")

        st.session_state.repo_full_name = identity.full_name
        st.session_state.collection_name = collection_name
        st.session_state.stats = {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
        }
        st.session_state.messages = []
        st.session_state.ready = True

    finally:
        if cloned:
            cloned.cleanup()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Repository")
    if st.session_state.repo_full_name:
        st.code(st.session_state.repo_full_name, language=None)
    else:
        st.caption("No repository analyzed yet.")

    st.markdown("### RAG Configuration")
    st.caption(f"Embedding model: `{DEFAULT_MODEL_NAME}`")
    st.session_state.top_k = st.slider("Top-K", min_value=1, max_value=15, value=st.session_state.top_k)
    st.session_state.similarity_threshold = st.slider(
        "Similarity threshold", min_value=0.0, max_value=1.0,
        value=st.session_state.similarity_threshold, step=0.05,
    )

    st.markdown("### Repository Statistics")
    if st.session_state.stats:
        files_processed = st.session_state.stats.get("files_processed")
        st.metric("Files", files_processed if files_processed is not None else "—")
        st.metric("Chunks", st.session_state.stats.get("chunks_created", "—"))
    else:
        st.caption("—")

    st.markdown("### Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Repository", use_container_width=True):
            if st.session_state.collection_name:
                vector_store.delete_collection(st.session_state.collection_name)
            for key in ("repo_full_name", "collection_name", "stats", "messages", "ready"):
                st.session_state[key] = [] if key == "messages" else (False if key == "ready" else None)
            st.rerun()
    with col_b:
        if st.button("Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set — answers won't be generated until it is.")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("CodeRAG")
st.caption("Chat with any GitHub repository using RAG")

st.markdown("##### GitHub Repository URL")
url_col, button_col = st.columns([5, 1])
with url_col:
    repo_url = st.text_input(
        "GitHub Repository URL", placeholder="https://github.com/owner/project",
        label_visibility="collapsed",
    )
with button_col:
    analyze_clicked = st.button("Analyze Repository", type="primary", use_container_width=True)

if analyze_clicked:
    analyze_repository(repo_url)

if st.session_state.ready and st.session_state.stats:
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = st.session_state.stats
    c1.metric("Repository", st.session_state.repo_full_name)
    c2.metric("Files Processed", stats.get("files_processed") or "cached")
    c3.metric("Chunks Created", stats.get("chunks_created", "—"))
    c4.metric("Embedding Model", DEFAULT_MODEL_NAME)
    c5.metric("Status", "✓ RAG Ready")


# --------------------------------------------------------------------------
# Example questions
# --------------------------------------------------------------------------

if st.session_state.ready:
    with st.expander("💡 Try asking"):
        cols = st.columns(3)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 3].button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = q


# --------------------------------------------------------------------------
# Chat interface
# --------------------------------------------------------------------------

def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            retrieved = msg.get("retrieved", [])
            if retrieved:
                with st.expander("🔍 Retrieved Context"):
                    for i, chunk in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**{i}. {chunk.file_path}**  \n"
                            f"Lines {chunk.start_line}-{chunk.end_line} · "
                            f"Relevance: {chunk.relevance:.2f}"
                        )
                        with st.expander(f"View snippet — {chunk.file_path}", expanded=False):
                            st.code(chunk.content, language=chunk.file_type or None)
            sources = msg.get("sources", [])
            if sources:
                st.markdown("**📄 Sources**")
                st.markdown("\n".join(f"- `{s}`" for s in sources))


def ask_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

    retrieved = retriever.retrieve(
        st.session_state.collection_name,
        question,
        top_k=st.session_state.top_k,
        similarity_threshold=st.session_state.similarity_threshold,
    )

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    try:
        result = generate_answer(question, retrieved, conversation_history=history)
        answer_text = result.text
        sources = result.sources
    except GenerationError as e:
        answer_text = e.user_message
        sources = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "retrieved": retrieved,
            "sources": sources,
        }
    )


if st.session_state.ready:
    for msg in st.session_state.messages:
        render_message(msg)

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        ask_question(q)
        st.rerun()

    user_question = st.chat_input("Ask a question about this repository...")
    if user_question:
        ask_question(user_question)
        st.rerun()
else:
    st.info("Paste a public GitHub repository URL above and click **Analyze Repository** to get started.")
