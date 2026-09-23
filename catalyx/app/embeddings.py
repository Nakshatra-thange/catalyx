from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 outputs 384-dim vectors — matches VECTOR(384) in init.sql
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return vectors.tolist()