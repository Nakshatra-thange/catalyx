from sqlalchemy import text
from sqlalchemy.orm import Session
from embeddings import embed_text


def search_semantic(db: Session, query: str, top_k: int = 10) -> list[dict]:
    query_vector = embed_text(query)

    # <=> is pgvector's cosine distance operator (1 - cosine_similarity).
    # Since our embeddings are normalized, this is equivalent to using
    # the dot product, but <=> is the clearest to read and reason about.
    rows = db.execute(
        text("""
            SELECT id, title, price, category, brand, rating,
                   1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity
            FROM products
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT :top_k
        """),
        {"query_vector": str(query_vector), "top_k": top_k}
    ).fetchall()

    return [
        {
            "id": row.id,
            "title": row.title,
            "price": float(row.price),
            "category": row.category,
            "brand": row.brand,
            "rating": float(row.rating),
            "similarity": round(float(row.similarity), 4),
        }
        for row in rows
    ]