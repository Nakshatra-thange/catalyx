import os
import sys

from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import SessionLocal
import search_index
from filtered_search import search_filtered
from reranker import rerank

from eval.benchmark import BENCHMARK
from eval.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


K = 10
QUERY_TYPES = ("literal", "semantic")


def get_bm25_only_ids(query: str, top_k: int) -> list[int]:
    results = search_index.get_index().search(query, top_k=top_k)
    return [doc_id for doc_id, _ in results]


def get_hybrid_ids(db, query: str, top_k: int) -> list[int]:
    # search_filtered uses page/page_size following the Day 9 API change.
    response = search_filtered(db, query, page=1, page_size=top_k)
    return [result["id"] for result in response["results"]]


def get_hybrid_reranked_ids(db, query: str, top_k: int) -> list[int]:
    response = search_filtered(db, query, page=1, page_size=top_k)
    return [result["id"] for result in rerank(response["results"])]


def audit_ground_truth(db) -> None:
    """Prove every stored relevance set still equals the catalog query result."""
    for item in BENCHMARK:
        rows = db.execute(
            text(f"SELECT id FROM products WHERE {item['ground_truth_where']} ORDER BY id")
        ).fetchall()
        actual_ids = {row.id for row in rows}
        if actual_ids != item["relevant_ids"]:
            raise RuntimeError(
                f"Ground-truth drift for {item['query']!r}: "
                f"expected {sorted(item['relevant_ids'])}, got {sorted(actual_ids)}"
            )
        if len(actual_ids) > 20:
            raise RuntimeError(
                f"{item['query']!r} has {len(actual_ids)} relevant products; "
                "replace it or exclude it from Recall@K."
            )


def evaluate_method(name: str, get_ids_fn, queries: list[dict], db=None) -> dict:
    recalls = []
    precisions = []
    reciprocal_ranks = []

    for item in queries:
        if db is None:
            retrieved_ids = get_ids_fn(item["query"], K)
        else:
            retrieved_ids = get_ids_fn(db, item["query"], K)

        relevant_ids = item["relevant_ids"]
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, K))
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, K))
        reciprocal_ranks.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "method": name,
        "avg_recall_at_10": sum(recalls) / len(recalls),
        "avg_precision_at_10": sum(precisions) / len(precisions),
        "avg_mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def print_results(query_type: str, results: list[dict], query_count: int) -> None:
    print(f"{query_type.title()} queries ({query_count}), K={K}")
    print(f"{'Method':<30} {'Recall@10':>10} {'Precision@10':>14} {'MRR':>10}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['method']:<30} "
            f"{result['avg_recall_at_10']:>10.4f} "
            f"{result['avg_precision_at_10']:>14.4f} "
            f"{result['avg_mrr']:>10.4f}"
        )


def main() -> None:
    db = SessionLocal()
    try:
        audit_ground_truth(db)
        print(f"Ground-truth audit passed for {len(BENCHMARK)} queries.\n")

        # Build once before timing/evaluating the three retrieval variants.
        search_index.get_index()

        methods = [
            ("BM25 only", get_bm25_only_ids, None),
            ("Hybrid (BM25 + Dense, RRF)", get_hybrid_ids, db),
            ("Hybrid + Reranked", get_hybrid_reranked_ids, db),
        ]
        for query_type in QUERY_TYPES:
            queries = [item for item in BENCHMARK if item["query_type"] == query_type]
            results = [
                evaluate_method(name, get_ids_fn, queries, method_db)
                for name, get_ids_fn, method_db in methods
            ]
            print_results(query_type, results, len(queries))
            print()
    finally:
        db.close()


if __name__ == "__main__":
    main()
