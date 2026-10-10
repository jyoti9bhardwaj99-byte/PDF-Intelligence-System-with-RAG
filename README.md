# 📄 PDF Intelligence System with RAG

Ask questions about your PDFs and get answers with **page citations**. The system uses **hybrid retrieval** (BM25 keyword search + embeddings, merged with Reciprocal Rank Fusion), and refuses to answer when the documents don't contain the answer.

## Features

- **Hybrid retrieval**: exact-word matching (BM25) combined with meaning-based search (embeddings), merged with RRF, so labels like "Module 8.5" are found as reliably as descriptive questions
- **Page citations**: every answer lists the file, page and match score of the chunks it used
- **"Not found" guard**: if no passage is relevant, the app replies with a fixed message instead of letting the model guess
- **Grounded answers**: a strict prompt tells the model to use only retrieved text and to copy names and topics exactly
- **Heading-aware chunking**: PDFs with headings such as `Module N:` are cut at each heading, even across page breaks, so a section stays in one chunk
- **Evaluation script**: measures retrieval quality (Hit@k and MRR) for vector-only vs hybrid search
- **Streamlit chat UI** with multi-PDF upload and a separate chat memory per browser session

## How it works

```
Upload PDFs
   │
   ▼
Load pages ──► Heading-aware chunking (fallback: page-based splitting)
   │
   ▼
Index: embeddings (vector search) + BM25 (keyword search)
   │
   ▼
Question ──► Hybrid search (vector + BM25, merged with RRF)
   │
   ▼
Guard: best similarity too low? ──► "I couldn't find this in your uploaded documents."
   │
   ▼
LLM answers only from retrieved passages ──► Answer + sources (file, page, score)
```

## Tech stack

Python · LangChain · LangGraph · Groq LLM API · Hugging Face embeddings (`all-MiniLM-L6-v2`) · `rank-bm25` · PyPDF · Streamlit

## Project structure

| File | Purpose |
|---|---|
| `streamlit_app.py` | Chat interface: upload, questions, sources |
| `app/config.py` | All settings (models, chunk sizes, top-k, thresholds) |
| `app/ingestion.py` | PDF loading and heading-aware chunking |
| `app/vectorstore.py` | Embeddings and vector store |
| `app/retrieval.py` | Hybrid retriever (vector + BM25 + RRF) |
| `app/agent.py` | LLM agent, retrieval tool, citations, not-found guard |
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

Change the model or any other setting in `app/config.py`.

## Run

```bash
streamlit run streamlit_app.py
```

Upload one or more PDFs, then ask questions. Each answer shows an expandable **Sources** list with the file, page number and match score of the chunks used.

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

- **Problem:** asking "Explain Module 8.5" returned "not found" even though the text existed. Embeddings can't tell "Module 8.5" from "Module 11".
- **Fixes:**
  1. Added BM25 keyword search and merged both rankings with RRF.
  2. Applied the "not found" guard to the best match instead of every chunk, so correct but low-scoring chunks weren't dropped.
  3. Chose chunk boundaries by heading instead of by page: Module 8.5 crosses a page break, and its mini project had been separated from its heading.
  4. Tightened the prompt after the model added details that were not in the PDF.
- **Takeaway:** retrieval failures and generation failures are different problems and need different fixes. The evaluation script isolates the retrieval side.

## Limitations

- The vector index is in memory, so documents must be uploaded again after a restart
- Scanned PDFs without a text layer are not supported (no OCR)
- Heading-aware chunking looks for headings like `Module N:`, `Chapter N:` and `Unit N:`; other PDFs fall back to page-based splitting
- The evaluation set is small and covers one document

## Roadmap

- Persistent vector store (Chroma)
- FastAPI backend with the Streamlit app as a client
- User accounts and per-user document storage
- Automated tests, Docker, and deployment
##
