from sqlalchemy.orm import Session
from sqlalchemy import text
import search_index
from vector_search import search_semantic
from rrf import reciprocal_rank_fusion


def search_hybrid(db: Session, query: str, top_k: int = 10, candidate_pool: int = 30) -> list[dict]:
    # Pull a larger candidate pool from each method before fusing,
    # so RRF has enough signal to work with beyond just the final top_k.
    bm25_index = search_index.get_index()
    bm25_results = bm25_index.search(query, top_k=candidate_pool)
    bm25_ids = [doc_id for doc_id, _ in bm25_results]

    semantic_results = search_semantic(db, query, top_k=candidate_pool)
    semantic_ids = [r["id"] for r in semantic_results]

    fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])
    top_fused_ids = [doc_id for doc_id, _ in fused[:top_k]]
    fused_scores = {doc_id: score for doc_id, score in fused}

    if not top_fused_ids:
        return []

    rows = db.execute(
        text("SELECT id, title, price, category, brand, rating FROM products WHERE id = ANY(:ids)"),
        {"ids": top_fused_ids}
    ).fetchall()
    row_map = {row.id: row for row in rows}

    ordered = [
        {
            "id": pid,
            "title": row_map[pid].title,
            "price": float(row_map[pid].price),
            "category": row_map[pid].category,
            "brand": row_map[pid].brand,
            "rating": float(row_map[pid].rating),
            "rrf_score": round(fused_scores[pid], 5),
            "in_bm25": pid in bm25_ids,
            "in_semantic": pid in semantic_ids,
        }
        for pid in top_fused_ids if pid in row_map
    ]

    return ordered