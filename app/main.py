import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.agent import build_agent
from app.config import UPLOAD_DIR
from app.ingestion import ingest_files
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    UploadResponse,
)
from app.vectorstore import VectorIndex

app = FastAPI(title="PDF Intelligence API", version="0.1.0")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
index = VectorIndex()

_lock = threading.Lock()  # the agent keeps per-question state, so one question at a time
_agent = None


def _get_agent():
    """Build the agent from the saved index the first time it's needed."""
    global _agent
    if _agent is None:
        chunks = index.all_chunks()
        if not chunks:
            return None
        _agent = build_agent(index, chunks)
    return _agent


@app.get("/health")
def health():
    return {"status": "ok", "chunks": index.count()}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    return DocumentsResponse(files=index.sources(), chunks=index.count())


@app.post("/documents", response_model=UploadResponse)
def upload_documents(files: list[UploadFile] = File(...)):
    global _agent
    paths = []
    for f in files:
        name = Path(f.filename or "").name  # .name removes any folder tricks
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{name}' is not a PDF file.")
        path = UPLOAD_DIR / name
        path.write_bytes(f.file.read())
        paths.append(path)

    try:
        chunks = ingest_files(paths)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read the PDF: {e}")
    if not chunks:
        raise HTTPException(
            status_code=422, detail="No readable text found. The PDF may be a scan."
        )

    with _lock:
        index.add(chunks)
        _agent = None  # rebuilt with the new chunks on the next question

    return UploadResponse(
        added_files=[p.name for p in paths],
        chunks_added=len(chunks),
        total_chunks=index.count(),
    )


@app.delete("/documents")
def clear_documents():
    global _agent
    with _lock:
        index.clear()
        _agent = None
    return {"status": "cleared"}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    with _lock:
        agent = _get_agent()
        if agent is None:
            raise HTTPException(
                status_code=409, detail="No documents indexed yet. Upload a PDF first."
            )
        answer, sources = agent.ask(req.question, req.conversation_id)
    return AskResponse(answer=answer, sources=sources)