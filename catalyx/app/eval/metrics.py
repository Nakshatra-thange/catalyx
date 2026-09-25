def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of the complete relevance set retrieved in the first ``k``."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of the first ``k`` ranks occupied by relevant products.

    The denominator is always ``k``. Missing results therefore count as
    non-relevant positions, which keeps scores comparable across methods.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Reciprocal rank of the first relevant product (zero if none is found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
