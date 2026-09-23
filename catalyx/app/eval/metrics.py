def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """
    Recall@K = (relevant items found in top K results) / (total relevant items)

    Measures: of everything that SHOULD have been found, how much did
    we actually surface in the top K? Doesn't penalize ranking order
    within the top K — only whether relevant items appear at all.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    return len(found) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Reciprocal Rank (for one query) = 1 / (rank of the FIRST relevant
    result), or 0 if no relevant result appears at all.

    MRR = average of reciprocal rank across all queries in the benchmark.

    Measures: how quickly does the user see a relevant result? A system
    that puts the first relevant item at rank 1 scores 1.0 for that
    query; at rank 5 it scores 0.2. This penalizes making the user
    scroll, unlike Recall@K which doesn't care about order.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0