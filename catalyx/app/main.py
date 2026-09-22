from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from db import get_db

app = FastAPI(title="Catalyx", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "catalyx"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT COUNT(*) FROM products")).scalar()
    return {"status": "ok", "product_count": result}