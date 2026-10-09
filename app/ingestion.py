from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest_files(paths):
    """Take a list of PDF file paths and return a list of text chunks."""
    pages = []
    for path in paths:
        pages.extend(PyPDFLoader(str(path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(pages)