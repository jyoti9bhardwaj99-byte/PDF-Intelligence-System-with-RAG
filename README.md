# 📄 PDF Intelligence System with Retrieval-Augmented Generation (RAG)

An intelligent PDF question-answering system that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Upload one or multiple PDF documents
- Extract and process PDF content
- Split documents into smaller text chunks
- Generate semantic embeddings using Hugging Face
- Retrieve relevant document content using similarity search
- Generate answers using Groq LLM
- Interactive chat interface using Streamlit
- Session-based conversation memory

## 🧠 How It Works

```text
PDF Upload
    ↓
PDF Document Loading
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
In-Memory Vector Store
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
Generated Answer

🛠️ Technologies Used

Python
Streamlit
LangChain
LangGraph
Hugging Face
Sentence Transformers
Groq
PyPDF
InMemoryVectorStore

📁 Project Structure

PDF-Intelligence-System-with-RAG/
│
├── app/
│   ├── doc_files/
│   │   ├── data_science_syllabus.pdf
│   │   └── medical_report.pdf
│   │
│   ├── main.py
│   └── .gitignore
│
├── .gitignore
├── requirements.txt
└── README.md

⚙️ Installation

1. Clone the repository
git clone https://github.com/jyoti9bhardwaj99-byte/PDF-Intelligence-System-with-RAG.git

2. Open the project
cd PDF-Intelligence-System-with-RAG

3. Create a virtual environment
python -m venv env
Activate it on Windows:
env\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
Replace your_groq_api_key with your actual Groq API key.

▶️ Run the Application
streamlit run app/main.py

The application will open in your browser at:

http://localhost:8501

🔎 RAG Pipeline

The system follows these steps:

User uploads PDF documents.
PDF text is extracted.
Text is split into chunks.
Embeddings are generated using all-MiniLM-L6-v2.
Embeddings are stored in an in-memory vector store.
User asks a question.
Relevant document chunks are retrieved.
Retrieved context is provided to the LLM.
The LLM generates an answer based on the document content.

Future Improvements
Persistent vector database
PDF page/source citations
Support for additional document formats
Improved conversational memory
User authentication
Cloud deployment
Better document metadata filtering
