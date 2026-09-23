import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import SessionLocal
import search_index
from vector_search import search_semantic
from filtered_search import search_filtered
from reranker import rerank
from eval.benchmark import BENCHMARK
from eval.metrics import recall_at_k, mean_reciprocal_rank

K = 10


def get_bm25_only_ids(query: str, top_k: int) -> list[int]:
    index = search_index.get_index()
    results = index.search(query, top_k=top_k)
    return [doc_id for doc_id, _ in results]


def get_hybrid_ids(db, query: str, top_k: int) -> list[int]:
    response = search_filtered(db, query, top_k=top_k)
    return [r["id"] for r in response["results"]]


def get_hybrid_reranked_ids(db, query: str, top_k: int) -> list[int]:
    response = search_filtered(db, query, top_k=top_k)
    reranked = rerank(response["results"])
    return [r["id"] for r in reranked]


def evaluate_method(name: str, get_ids_fn, db=None) -> dict:
    recalls = []
    mrrs = []

    for item in BENCHMARK:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        if db is not None:
            retrieved_ids = get_ids_fn(db, query, K)
        else:
            retrieved_ids = get_ids_fn(query, K)

        recalls.append(recall_at_k(retrieved_ids, relevant_ids, K))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "method": name,
        "avg_recall_at_10": round(sum(recalls) / len(recalls), 4),
        "avg_mrr": round(sum(mrrs) / len(mrrs), 4),
    }


def main():
    db = SessionLocal()
    try:
        search_index.get_index()  # ensure BM25 index is built

        results = [
            evaluate_method("BM25 only", get_bm25_only_ids),
            evaluate_method("Hybrid (BM25 + Dense, RRF)", get_hybrid_ids, db=db),
            evaluate_method("Hybrid + Reranked", get_hybrid_reranked_ids, db=db),
        ]

        print(f"\nEvaluated on {len(BENCHMARK)} queries, K={K}\n")
        print(f"{'Method':<30} {'Recall@10':<12} {'MRR':<8}")
        print("-" * 50)
        for r in results:
            print(f"{r['method']:<30} {r['avg_recall_at_10']:<12} {r['avg_mrr']:<8}")

    finally:
        db.close()


if __name__ == "__main__":
    main()