from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy import text
from vector_search import search_semantic
from hybrid_search import search_hybrid
from filtered_search import search_filtered
from sqlalchemy.orm import Session
from db import get_db
import ingest
import search_index
from reranker import rerank
import cache

app = FastAPI(title="Catalyx", version="0.3.0")


@app.on_event("startup")
def startup_event():
    search_index.build_index()


@app.get("/health")
def health():
    return {"status": "ok", "service": "catalyx"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT COUNT(*) FROM products")).scalar()
    return {"status": "ok", "product_count": result}


@app.post("/embed")
def trigger_embedding(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest.run)
    return {"status": "embedding_started", "note": "check logs or /health/embeddings for progress"}


@app.get("/health/embeddings")
def health_embeddings(db: Session = Depends(get_db)):
    total = db.execute(text("SELECT COUNT(*) FROM products")).scalar()
    embedded = db.execute(text("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")).scalar()
    return {"total": total, "embedded": embedded, "remaining": total - embedded}


@app.get("/search/keyword")
def search_keyword(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    index = search_index.get_index()
    results = index.search(q, top_k=top_k)

    if not results:
        return {"query": q, "results": []}

    ids = [r[0] for r in results]
    scores = {r[0]: r[1] for r in results}

    rows = db.execute(
        text("SELECT id, title, price, category, brand, rating FROM products WHERE id = ANY(:ids)"),
        {"ids": ids}
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
            "bm25_score": round(scores[pid], 4),
        }
        for pid in ids if pid in row_map
    ]

    return {"query": q, "results": ordered}

@app.get("/search/semantic")
def search_semantic_endpoint(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    results = search_semantic(db, q, top_k=top_k)
    return {"query": q, "results": results}

@app.post("/search/reindex")
def reindex():
    index = search_index.build_index()
    return {"status": "reindexed", "product_count": index.N}

@app.get("/search/hybrid")
def search_hybrid_endpoint(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    results = search_hybrid(db, q, top_k=top_k)
    return {"query": q, "results": results}

@app.get("/search")
def search_endpoint(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    cached_response = cache.get_cached(q, top_k)
    if cached_response is not None:
        cached_response["cache_hit"] = True
        return cached_response

    response = search_filtered(db, q, top_k=top_k)
    response["results"] = rerank(response["results"])
    response["cache_hit"] = False

    cache.set_cached(q, top_k, response)
    return response