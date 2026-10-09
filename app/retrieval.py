import re

from rank_bm25 import BM25Okapi

from app.config import TOP_K

RRF_K = 60  # standard constant: higher means rank differences matter less


def tokenize(text):
    """Lowercase words. A number like 8.5 stays ONE token."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _key(doc):
    """A unique ID for a chunk, so the two searches can be matched up."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


class HybridRetriever:
    def __init__(self, vector_store, chunks):
        self.vector_store = vector_store
        self.chunks = chunks
        self.by_key = {_key(c): c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query, k=TOP_K):
        """Return (list of (chunk, vector_score), best_vector_score)."""
        # 1) vector search, ranked by meaning
        n = min(len(self.chunks), 200)
        vec = self.vector_store.similarity_search_with_score(query, k=n)
        vec_scores = {_key(d): s for d, s in vec}
        vec_rank = {_key(d): i for i, (d, _) in enumerate(vec)}

        # 2) keyword search, ranked by BM25 (chunks with no matching word are skipped)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        bm_rank = {
            _key(self.chunks[i]): r
            for r, i in enumerate(order)
            if scores[i] > 0
        }

        # 3) merge: add reciprocal-rank points from both lists
        fused = {}
        for key, r in vec_rank.items():
            fused[key] = fused.get(key, 0) + 1 / (RRF_K + r + 1)
        for key, r in bm_rank.items():
            fused[key] = fused.get(key, 0) + 1 / (RRF_K + r + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        results = [(self.by_key[key], vec_scores.get(key, 0.0)) for key in top]
        best_vector = max(vec_scores.values(), default=0.0)
        return results, best_vector