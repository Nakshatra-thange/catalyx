from sqlalchemy.orm import Session
from sqlalchemy import text
import search_index
from vector_search import search_semantic
from rrf import reciprocal_rank_fusion
from query_parser import parse_query


def _get_candidate_ids(db: Session, filters: dict) -> set[int] | None:
    """
    Pre-filtering: get the set of product IDs matching structured filters
    BEFORE running BM25/vector search, so ranking only happens within
    products that already satisfy price/category constraints.

    Returns None if no filters are set (meaning: don't restrict candidates).
    """
    if not any(filters.values()):
        return None

    conditions = []
    params = {}

    if filters.get("price_min") is not None:
        conditions.append("price >= :price_min")
        params["price_min"] = filters["price_min"]

    if filters.get("price_max") is not None:
        conditions.append("price <= :price_max")
        params["price_max"] = filters["price_max"]

    if filters.get("category") is not None:
        conditions.append("category = :category")
        params["category"] = filters["category"]

    where_clause = " AND ".join(conditions)
    rows = db.execute(text(f"SELECT id FROM products WHERE {where_clause}"), params).fetchall()

    return {row.id for row in rows}


def search_filtered(db: Session, raw_query: str, top_k: int = 10, candidate_pool: int = 30) -> dict:
    parsed = parse_query(raw_query)
    semantic_text = parsed["semantic_text"]
    filters = parsed["filters"]

    allowed_ids = _get_candidate_ids(db, filters)

    bm25_index = search_index.get_index()
    bm25_raw = bm25_index.search(semantic_text, top_k=candidate_pool * 3)
    bm25_ids = [doc_id for doc_id, _ in bm25_raw]
    if allowed_ids is not None:
        bm25_ids = [doc_id for doc_id in bm25_ids if doc_id in allowed_ids][:candidate_pool]
    else:
        bm25_ids = bm25_ids[:candidate_pool]

    semantic_raw = search_semantic(db, semantic_text, top_k=candidate_pool * 3)
    semantic_ids = [r["id"] for r in semantic_raw]
    if allowed_ids is not None:
        semantic_ids = [pid for pid in semantic_ids if pid in allowed_ids][:candidate_pool]
    else:
        semantic_ids = semantic_ids[:candidate_pool]

    fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]
    fused_scores = {doc_id: score for doc_id, score in fused}

    results = []
    if top_ids:
        rows = db.execute(
            text("SELECT id, title, price, category, brand, rating, stock FROM products WHERE id = ANY(:ids)"),
            {"ids": top_ids}
        ).fetchall()
        
        row_map = {row.id: row for row in rows}
        results = [
            {
                "id": pid,
                "title": row_map[pid].title,
                "price": float(row_map[pid].price),
                "category": row_map[pid].category,
                "brand": row_map[pid].brand,
                "rating": float(row_map[pid].rating),
                "stock": row_map[pid].stock,
                "rrf_score": round(fused_scores[pid], 5),
            }
            for pid in top_ids if pid in row_map
        ]
    return {
        "query": raw_query,
        "parsed_semantic_text": semantic_text,
        "applied_filters": filters,
        "results": results,
    }