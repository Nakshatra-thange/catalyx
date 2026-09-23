import sys
from sqlalchemy import text
from db import SessionLocal
from embeddings import embed_batch

BATCH_SIZE = 100


def build_embedding_text(title: str, description: str, category: str, brand: str) -> str:
    return f"{title}. {description}. Category: {category}. Brand: {brand}."


def run():
    db = SessionLocal()
    try:
        rows = db.execute(
            text("SELECT id, title, description, category, brand FROM products WHERE embedding IS NULL ORDER BY id")
        ).fetchall()

        total = len(rows)
        print(f"Found {total} products without embeddings.")

        if total == 0:
            print("Nothing to do.")
            return

        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = [build_embedding_text(r.title, r.description, r.category, r.brand) for r in batch]
            vectors = embed_batch(texts)

            for row, vector in zip(batch, vectors):
                db.execute(
                    text("UPDATE products SET embedding = :embedding WHERE id = :id"),
                    {"embedding": str(vector), "id": row.id}
                )
            db.commit()
            print(f"Embedded {min(i + BATCH_SIZE, total)}/{total}")

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    run()