import uuid

import requests
import streamlit as st

from app.config import API_URL, LLM_MODEL

st.set_page_config(page_title="PDF Intelligence System", page_icon="📄")
st.title("📄 PDF Intelligence System")


# ---------- talking to the API ----------
def api(method, path, timeout=120, **kwargs):
    """Call the backend. If it isn't running, show a clear message."""
    try:
        return requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at {API_URL}. "
            "Start it in another terminal: uvicorn app.main:app --port 8000"
        )
        st.stop()


def error_text(resp):
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return resp.text


# ---------- data kept between reruns ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None


def new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = str(uuid.uuid4())


def show_sources(sources):
    """Show an expandable list of where the answer came from."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['file']}** · page {s['page']} · match {s['score']:.2f}")
            st.caption(s["snippet"] + "...")


# ---------- sidebar ----------
resp = api("GET", "/documents", timeout=15)
docs = resp.json() if resp.ok else {"files": [], "chunks": 0}

with st.sidebar:
    st.caption(f"Model: {LLM_MODEL}")
    st.caption(f"API: {API_URL}")

    st.subheader("Indexed documents")
    if docs["files"]:
        for name in docs["files"]:
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
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        with st.spinner("Indexing..."):
            r = api("POST", "/documents", files=files, timeout=600)
        if r.ok:
            new_chat()
        else:
            st.session_state.upload_error = error_text(r)
        st.session_state.uploader_key += 1  # empties the uploader
        st.rerun()
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
        st.session_state.upload_error = None

    st.divider()
    if st.button("New chat"):
        new_chat()
        st.rerun()
    if st.button("Clear all documents"):
        api("DELETE", "/documents", timeout=60)
        new_chat()
        st.rerun()

# ---------- main area ----------
if docs["chunks"] == 0:
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
            r = api(
                "POST",
                "/ask",
                json={"question": query, "conversation_id": st.session_state.conversation_id},
                timeout=180,
            )
        if r.ok:
            data = r.json()
            answer, sources = data["answer"], data["sources"]
        else:
            answer, sources = f"Error: {error_text(r)}", []

        with st.chat_message("assistant"):
            st.markdown(answer)
            show_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )