"""Retrieval evaluation: vector-only vs hybrid. Run: python -m eval.run_eval"""
import json
from pathlib import Path

from app.config import MIN_SIMILARITY, TOP_K
from app.ingestion import ingest_files
from app.retrieval import HybridRetriever
from app.vectorstore import VectorIndex

PDF = Path("app/doc_files/data_science_syllabus.pdf")
GOLDEN = Path("eval/golden.jsonl")


def first_hit_rank(chunks, marker):
    """1-based position of the first chunk containing the marker, or None."""
    for i, chunk in enumerate(chunks, start=1):
        if marker in chunk.page_content:
            return i
    return None


def main():
    chunks = ingest_files([PDF])
    store = VectorIndex(persist_dir=Path("data/chroma_eval"), name="eval")
    store.clear()
    store.add(chunks)
    hybrid = HybridRetriever(store, chunks)

    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    answerable = [q for q in questions if not q.get("should_refuse")]
    refuse = [q for q in questions if q.get("should_refuse")]

    ranks = {"vector-only": [], "hybrid": []}
    print(f"{'question':<62}{'vector':>8}{'hybrid':>8}")
    for q in answerable:
        vec_docs = [d for d, _ in store.similarity_search_with_score(q["question"], k=TOP_K)]
        hyb_docs = [d for d, _ in hybrid.search(q["question"], k=TOP_K)[0]]
        rv = first_hit_rank(vec_docs, q["must_contain"])
        rh = first_hit_rank(hyb_docs, q["must_contain"])
        ranks["vector-only"].append(rv)
        ranks["hybrid"].append(rh)
        print(f"{q['question'][:60]:<62}{str(rv or '-'):>8}{str(rh or '-'):>8}")

    print(f"\nRetrieval over {len(answerable)} questions (top {TOP_K}):")
    for name, rs in ranks.items():
        hit = sum(r is not None for r in rs) / len(rs)
        mrr = sum(1 / r for r in rs if r) / len(rs)
        print(f"  {name:<12} Hit@{TOP_K} = {hit:.0%}   MRR = {mrr:.2f}")

    if refuse:
        blocked = 0
        for q in refuse:
            _, best_score = hybrid.search(q["question"], k=TOP_K)
            blocked += best_score < MIN_SIMILARITY
        print(f"\nOut-of-scope questions correctly refused: {blocked}/{len(refuse)}")


if __name__ == "__main__":
    main()