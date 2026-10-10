import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

PDF = ("files", ("syllabus.pdf", b"%PDF-1.4 fake bytes", "application/pdf"))

CHUNKS = [
    Document(page_content="Module 1: Python", metadata={"source": "syllabus.pdf", "page": 0}),
    Document(page_content="Module 2: Pandas", metadata={"source": "syllabus.pdf", "page": 1}),
]


class FakeAgent:
    """Stands in for the real LLM agent, so tests need no API key."""

    def ask(self, question, conversation_id):
        sources = [{"file": "syllabus.pdf", "page": 1, "score": 0.5, "snippet": "snippet"}]
        return f"echo: {question}", sources


@pytest.fixture
def api(tmp_path, monkeypatch):
    import app.vectorstore as vs

    # fake embeddings: no model download, no real vectors
    monkeypatch.setattr(vs, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    import app.main as main

    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "index", vs.VectorIndex(persist_dir=tmp_path / "chroma", name="test"))
    monkeypatch.setattr(main, "_agent", None)
    monkeypatch.setattr(main, "build_agent", lambda index, chunks: FakeAgent())
    monkeypatch.setattr(main, "ingest_files", lambda paths: list(CHUNKS))
    return main, TestClient(main.app)


def test_health_on_an_empty_index(api):
    _, client = api
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}


def test_asking_before_any_upload_returns_409(api):
    _, client = api
    r = client.post("/ask", json={"question": "Explain Module 1", "conversation_id": "t"})
    assert r.status_code == 409


def test_upload_list_ask_and_clear(api):
    _, client = api

    r = client.post("/documents", files=[PDF])
    assert r.status_code == 200
    assert r.json()["added_files"] == ["syllabus.pdf"]
    assert r.json()["total_chunks"] == 2

    assert client.get("/documents").json() == {"files": ["syllabus.pdf"], "chunks": 2}

    r = client.post("/ask", json={"question": "Explain Module 1", "conversation_id": "t"})
    assert r.status_code == 200
    assert r.json()["answer"] == "echo: Explain Module 1"
    assert r.json()["sources"][0]["page"] == 1

    assert client.delete("/documents").json() == {"status": "cleared"}
    assert client.get("/documents").json() == {"files": [], "chunks": 0}
    r = client.post("/ask", json={"question": "again", "conversation_id": "t"})
    assert r.status_code == 409


def test_uploading_the_same_file_twice_does_not_duplicate_chunks(api):
    # Regression test for the duplicate-chunks bug
    _, client = api
    first = client.post("/documents", files=[PDF]).json()
    second = client.post("/documents", files=[PDF]).json()
    assert first["total_chunks"] == second["total_chunks"] == 2


def test_non_pdf_upload_is_rejected_with_400(api):
    _, client = api
    r = client.post("/documents", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


def test_unreadable_pdf_returns_422(api, monkeypatch):
    main, client = api

    def broken(paths):
        raise ValueError("broken file")

    monkeypatch.setattr(main, "ingest_files", broken)
    assert client.post("/documents", files=[PDF]).status_code == 422


def test_pdf_without_text_returns_422(api, monkeypatch):
    main, client = api
    monkeypatch.setattr(main, "ingest_files", lambda paths: [])
    r = client.post("/documents", files=[PDF])
    assert r.status_code == 422
    assert "No readable text" in r.json()["detail"]


def test_empty_question_is_rejected_with_422(api):
    _, client = api
    r = client.post("/ask", json={"question": "", "conversation_id": "t"})
    assert r.status_code == 422

def test_demo_mode_blocks_upload_and_clear(api, monkeypatch):
    main, client = api
    monkeypatch.setattr(main, "DEMO_MODE", True)

    assert client.post("/documents", files=[PDF]).status_code == 403
    assert client.delete("/documents").status_code == 403
    assert client.get("/documents").status_code == 200   