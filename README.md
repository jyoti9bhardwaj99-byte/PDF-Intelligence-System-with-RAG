# 📄 PDF Intelligence System with RAG

Ask questions about your PDFs and get answers with **page citations**. A **FastAPI** backend does the work (ingestion, hybrid retrieval, answer generation) and a **Streamlit** app is the client. The system refuses to answer when the documents don't contain the answer.

## Features

- **Hybrid retrieval**: BM25 keyword search plus embeddings, merged with Reciprocal Rank Fusion (RRF), so exact labels like "Module 8.5" are found as reliably as descriptive questions
- **Page citations**: every answer lists the file, page and match score of the chunks it used
- **"Not found" guard**: if no passage is relevant, the API returns a fixed message instead of letting the model guess
- **Grounded answers**: a strict prompt tells the model to use only retrieved text and to copy names and topics exactly
- **Heading-aware chunking**: PDFs with headings such as `Module N:` are cut at each heading, even across page breaks, so a section stays in one chunk
- **Persistent index**: vectors are stored in Chroma, so documents survive restarts; re-uploading a file does not create duplicates
- **REST API** (FastAPI) with typed request and response models and interactive docs at `/docs`
- **Evaluation script**: measures retrieval quality (Hit@k and MRR) for vector-only vs hybrid search

## Architecture

```
Browser ──► Streamlit (screen only) ──► FastAPI backend ──► Chroma (saved vectors) + BM25 + Groq LLM
```

```
Upload PDFs ──► Load pages ──► Heading-aware chunking ──► Chroma (vectors) + BM25 (keywords)

Question ──► Hybrid search (vector + BM25, merged with RRF)
         ──► Guard: best similarity too low? ──► "I couldn't find this in your uploaded documents."
         ──► LLM answers only from retrieved passages ──► Answer + sources (file, page, score)
```

## API

Interactive docs: `http://localhost:8000/docs`

| Endpoint | What it does |
|---|---|
| `GET /health` | Server status and number of indexed chunks |
| `GET /documents` | List indexed files |
| `POST /documents` | Upload one or more PDFs and index them |
| `DELETE /documents` | Clear the index |
| `POST /ask` | Ask a question; returns the answer and its sources |

Example:

```bash
curl -X POST http://localhost:8000/documents -F "files=@report.pdf"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Explain Module 8.5", "conversation_id": "demo"}'
```

## Tech stack

Python · FastAPI · Streamlit · LangChain · LangGraph · Groq LLM API · Hugging Face embeddings (`all-MiniLM-L6-v2`) · Chroma · `rank-bm25` · PyPDF · Pydantic

## Project structure

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI application and endpoints |
| `app/schemas.py` | Request and response models |
| `app/config.py` | All settings (models, chunk sizes, top-k, thresholds, API URL) |
| `app/ingestion.py` | PDF loading and heading-aware chunking |
| `app/vectorstore.py` | Chroma index with content-hash chunk IDs |
| `app/retrieval.py` | Hybrid retriever (vector + BM25 + RRF) |
| `app/agent.py` | LLM agent, retrieval tool, citations, not-found guard |
| `streamlit_app.py` | Chat UI that calls the API |
| `eval/golden.jsonl` | Test questions with expected passages |
| `eval/run_eval.py` | Retrieval evaluation (Hit@k, MRR) |

## Setup

```bash
git clone https://github.com/jyoti9bhardwaj99-byte/PDF-Intelligence-System-with-RAG.git
cd PDF-Intelligence-System-with-RAG
python -m venv env
env\Scripts\activate          # Mac/Linux: source env/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your free [Groq API key](https://console.groq.com/keys):

```
GROQ_API_KEY=your_key_here
```

## Run

Start the backend and the UI in two terminals:

```bash
# terminal 1: API
uvicorn app.main:app --port 8000

# terminal 2: UI
streamlit run streamlit_app.py
```

Add PDFs in the sidebar, then ask questions. Each answer shows an expandable **Sources** list. If the API is not running, the UI says so.

## Evaluation

```bash
python -m eval.run_eval
```

The script runs 12 answerable questions and 2 out-of-scope questions against a sample PDF and reports where the correct chunk ranks.

| Method | Hit@5 | MRR |
|---|---|---|
| Vector-only | 92% | 0.79 |
| **Hybrid (BM25 + vector, RRF)** | **100%** | **0.85** |

Out-of-scope questions correctly refused: 2/2.

Hybrid search helped most on exact-label questions: "Explain Module 8.5" moved from rank 4 to rank 2, and "What is covered in Module 0?" went from missed to found. Descriptive questions were already ranked first by both methods.

*Caveat: this is a small test set (12 questions, one 10-page document), so treat the numbers as directional, not as a benchmark.*

## Engineering notes

- **Problem:** asking "Explain Module 8.5" returned "not found" even though the text existed. Embeddings cannot tell "Module 8.5" from "Module 11".
- **Fixes:**
  1. Added BM25 keyword search and merged both rankings with RRF.
  2. Applied the "not found" guard to the best match instead of every chunk, so correct but low-scoring chunks were not dropped.
  3. Chose chunk boundaries by heading instead of by page: Module 8.5 crosses a page break, and its mini project had been separated from its heading.
  4. Tightened the prompt after the model added details that were not in the PDF.
  5. Made chunk IDs content hashes based on the file name, after a path-based ID caused duplicate chunks on re-upload.
- **Takeaway:** retrieval failures and generation failures are different problems and need different fixes. The evaluation script isolates the retrieval side.

## Limitations

- No user accounts: all documents share one index
- Questions are processed one at a time (a lock protects the agent's per-question state)
- Scanned PDFs without a text layer are not supported (no OCR)
- Heading-aware chunking looks for headings like `Module N:`, `Chapter N:` and `Unit N:`; other PDFs fall back to page-based splitting
- The evaluation set is small and covers one document

## Roadmap

- Automated tests (pytest) and CI
- Docker and deployment
- User accounts with per-user document storage
