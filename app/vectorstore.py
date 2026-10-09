from functools import lru_cache

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


@lru_cache
def get_embeddings():
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks):
    """Turn a list of chunks into a searchable vector store."""
    return InMemoryVectorStore.from_documents(
        documents=chunks, embedding=get_embeddings()
    )