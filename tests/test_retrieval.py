from langchain_core.documents import Document

from app.retrieval import HybridRetriever, tokenize


class FakeVectorStore:
    """Pretends to be the vector index: returns chunks in a fixed 'by meaning' order."""

    def __init__(self, ordered_chunks):
        self.ordered = ordered_chunks

    def similarity_search_with_score(self, query, k):
        return [(c, 0.30 - 0.01 * i) for i, c in enumerate(self.ordered[:k])]


def chunk(text, page=0):
    return Document(page_content=text, metadata={"source": "a.pdf", "page": page})


CHUNKS = [
    chunk("Module 8: Time Series Forecasting ARIMA"),
    chunk("Module 11: GenAI app building LangChain"),
    chunk("Module 9: NLP text cleaning"),
    chunk("Module 8.5: Big Data with PySpark"),  # ranked LAST by the fake vector search
]


def test_tokenize_keeps_decimal_numbers_together():
    assert tokenize("Explain Module 8.5!") == ["explain", "module", "8.5"]


def test_keyword_search_rescues_an_exact_label():
    store = FakeVectorStore(CHUNKS)
    query = "What is 8.5 about?"

    # vector search alone misses the right chunk in its top 2...
    vector_top2 = [c for c, _ in store.similarity_search_with_score(query, 2)]
    assert CHUNKS[3] not in vector_top2

    # ...but hybrid search puts it first
    results, _ = HybridRetriever(store, CHUNKS).search(query, k=2)
    assert results[0][0] is CHUNKS[3]


def test_best_vector_score_is_reported_for_the_guard():
    _, best = HybridRetriever(FakeVectorStore(CHUNKS), CHUNKS).search("anything", k=2)
    assert abs(best - 0.30) < 1e-9


def test_without_keyword_matches_vector_order_is_kept():
    results, _ = HybridRetriever(FakeVectorStore(CHUNKS), CHUNKS).search("zzzz", k=2)
    assert results[0][0] is CHUNKS[0]