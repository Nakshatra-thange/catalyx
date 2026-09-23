import math


def cosine_similarity_manual(vec_a: list[float], vec_b: list[float]) -> float:
    """
    cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)

    A . B    = dot product = sum(a_i * b_i for each dimension i)
    ||A||    = magnitude (L2 norm) of A = sqrt(sum(a_i^2))
    ||B||    = magnitude (L2 norm) of B = sqrt(sum(b_i^2))

    Result ranges from -1 (opposite) to 1 (identical direction).
    For normalized (unit-length) embedding vectors, this is equivalent
    to the dot product alone — which is why embed_text() in embeddings.py
    uses normalize_embeddings=True.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    from embeddings import embed_text

    v1 = embed_text("waterproof hiking jacket")
    v2 = embed_text("rain-resistant trekking coat")
    v3 = embed_text("wireless bluetooth headphones")

    print("Similarity (jacket vs trekking coat):", cosine_similarity_manual(v1, v2))
    print("Similarity (jacket vs headphones):   ", cosine_similarity_manual(v1, v3))