import uuid
from pathlib import Path

import streamlit as st

from app.agent import ask, build_agent
from app.config import LLM_MODEL, UPLOAD_DIR
from app.ingestion import ingest_files
from app.vectorstore import VectorIndex

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PDF Intelligence System", page_icon="📄")
st.title("📄 PDF Intelligence System")


@st.cache_resource
def get_index():
    """One saved index, shared by every rerun of the script."""
    return VectorIndex()


index = get_index()

# ---------- data kept between reruns ----------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def new_chat():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def rebuild_agent():
    """Build the agent from everything saved in the index."""
    chunks = index.all_chunks()
    st.session_state.agent = build_agent(index, chunks) if chunks else None


def add_pdfs(uploaded_files):
    """Save the PDFs, split them into chunks, add them to the saved index."""
    paths = []
    for f in uploaded_files:
        path = UPLOAD_DIR / Path(f.name).name  # .name removes any folder tricks
        path.write_bytes(f.getvalue())
        paths.append(path)

    chunks = ingest_files(paths)
    if not chunks:
        raise ValueError("No readable text found. The PDF may be a scan.")

    index.add(chunks)
    rebuild_agent()
    new_chat()  # a new agent has no memory of the old chat


def show_sources(sources):
    """Show an expandable list of where the answer came from."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['file']}** · page {s['page']} · match {s['score']:.2f}")
            st.caption(s["snippet"] + "...")


# After a restart, load the saved documents automatically
if st.session_state.agent is None and index.count() > 0:
    rebuild_agent()

# ---------- sidebar ----------
with st.sidebar:
    st.caption(f"Model: {LLM_MODEL}")

    st.subheader("Indexed documents")
    files = index.sources()
    if files:
        for name in files:
            st.write(f"• {name}")
    else:
        st.write("None yet")

    uploaded = st.file_uploader(
        "Add PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"upload_{st.session_state.uploader_key}",
    )
    if uploaded:
        try:
            with st.spinner("Indexing..."):
                add_pdfs(uploaded)
            st.session_state.uploader_key += 1  # empties the uploader
            st.rerun()
        except Exception as e:
            st.error(f"Could not process the files: {e}")

    st.divider()
    if st.button("New chat"):
        new_chat()
        st.rerun()
    if st.button("Clear all documents"):
        index.clear()
        st.session_state.agent = None
        new_chat()
        st.rerun()

# ---------- main area ----------
if st.session_state.agent is None:
    st.info("Add one or more PDF files in the sidebar to get started.")
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