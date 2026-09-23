from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF).

    For each item, its RRF score is the sum, across all ranked lists it
    appears in, of 1 / (k + rank), where rank is its 1-indexed position
    in that list.

    score(d) = sum over lists L of: 1 / (k + rank_L(d))

    Why this works well:
    - It only needs RANKS, not raw scores — so it sidesteps the problem
      of BM25 scores and cosine similarities living on totally different
      scales (BM25 can be e.g. 0-20+, cosine similarity is -1 to 1).
    - The constant k (commonly 60) dampens the impact of very high ranks
      dominating the fused score, and smooths out the difference between
      e.g. rank 1 and rank 2 vs rank 50 and rank 51.
    - An item ranked well in EITHER list gets a meaningful boost, and an
      item ranked well in BOTH lists rises to the top — exactly the
      "combine the strengths of both methods" behavior we want.
    """
    scores: dict[int, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused