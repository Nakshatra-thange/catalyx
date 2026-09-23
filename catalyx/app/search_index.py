from sqlalchemy import text
from db import SessionLocal
from bm25 import BM25Index

_bm25_index: BM25Index | None = None


def build_index() -> BM25Index:
    global _bm25_index
    db = SessionLocal()
    try:
        rows = db.execute(
            text("SELECT id, title, description, category, brand FROM products")
        ).fetchall()

        documents = {
            row.id: f"{row.title}. {row.description}. Category: {row.category}. Brand: {row.brand}."
            for row in rows
        }

        index = BM25Index()
        index.build(documents)
        _bm25_index = index
        print(f"BM25 index built over {index.N} products.")
        return index
    finally:
        db.close()


def get_index() -> BM25Index:
    if _bm25_index is None:
        return build_index()
    return _bm25_index