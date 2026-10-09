import uuid
from pathlib import Path

import streamlit as st

from app.agent import ask, build_agent
from app.config import LLM_MODEL, UPLOAD_DIR
from app.ingestion import ingest_files
from app.vectorstore import build_vector_store

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PDF Intelligence System", page_icon="📄")
st.title("📄 PDF Intelligence System")

# ---------- data kept between reruns ----------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def process_uploads(uploaded_files):
    """Save the PDFs, split them into chunks, index them, build the agent."""
    paths = []
    for f in uploaded_files:
        path = UPLOAD_DIR / Path(f.name).name  # .name removes any folder tricks
        path.write_bytes(f.getvalue())
        paths.append(path)

    chunks = ingest_files(paths)
    if not chunks:
        raise ValueError("No readable text found. The PDF may be a scan.")

    st.session_state.agent = build_agent(build_vector_store(chunks), chunks)

def show_sources(sources):
    """Show an expandable list of where the answer came from."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['file']}** · page {s['page']} · match {s['score']:.2f}")
            st.caption(s["snippet"] + "...")

# ---------- sidebar ----------
with st.sidebar:
    st.caption(f"Model: {LLM_MODEL}")
    if st.button("Start over"):
        st.session_state.agent = None
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ---------- screen 1: upload ----------
if st.session_state.agent is None:
    uploaded = st.file_uploader(
        "Select PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded:
        try:
            with st.spinner("Processing..."):
                process_uploads(uploaded)
            st.rerun()
        except Exception as e:
            st.error(f"Could not process the files: {e}")

# ---------- screen 2: chat ----------
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources", []))

    query = st.chat_input("Ask anything about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Thinking..."):
            answer, sources = ask(st.session_state.agent, query, st.session_state.thread_id)

        with st.chat_message("assistant"):
            st.markdown(answer)
            show_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )