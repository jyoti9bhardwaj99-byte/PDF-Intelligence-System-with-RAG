import hashlib
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_DIR, EMBEDDING_MODEL


@lru_cache
def get_embeddings():
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _chunk_id(doc):
    """Same chunk -> same ID, so uploading the same PDF twice doesn't create duplicates."""
    name = Path(doc.metadata.get("source", "")).name  # file name only, not the full path
    raw = f"{name}|{doc.metadata.get('page')}|{doc.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class VectorIndex:
    """A saved vector index (Chroma) that behaves like the old in-memory store."""

    def __init__(self, persist_dir=CHROMA_DIR, name="docs"):
        self.persist_dir = str(persist_dir)
        self.name = name
        self.db = self._open()

    def _open(self):
        return Chroma(
            collection_name=self.name,
            embedding_function=get_embeddings(),
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks):
        """Save chunks. Only keep simple metadata (file and page)."""
        clean = [
            Document(
                page_content=c.page_content,
                metadata={
                    "source": str(c.metadata.get("source", "")),
                    "page": int(c.metadata.get("page", 0)),
                },
            )
            for c in chunks
        ]
        if clean:
            self.db.add_documents(clean, ids=[_chunk_id(c) for c in clean])

    def count(self):
        return self.db._collection.count()

    def all_chunks(self):
        """Read every saved chunk back (used to rebuild the BM25 keyword index)."""
        data = self.db.get()
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]

    def sources(self):
        """Names of the files that are indexed."""
        data = self.db.get()
        return sorted({Path(m["source"]).name for m in data["metadatas"]})

    def clear(self):
        """Delete everything and start with an empty index."""
        self.db.delete_collection()
        self.db = self._open()

    def similarity_search_with_score(self, query, k):
        """Chroma gives distance (smaller = better). Convert to similarity (bigger = better)."""
        results = self.db.similarity_search_with_score(query, k=k)
        return [(doc, 1 - dist) for doc, dist in results]