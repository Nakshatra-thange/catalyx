def rerank(results: list[dict]) -> list[dict]:
    """
    Hand-built reranking: blends retrieval relevance (RRF score) with
    business signals, instead of relying on relevance alone.

    final_score = (0.6 * normalized_rrf) + (0.25 * normalized_rating)
                  + (0.15 * stock_availability_bonus)

    Weights are deliberately simple and tunable — the point is to show
    you can reason about WHY relevance shouldn't be the only signal:
    a highly "relevant" but out-of-stock or poorly-rated product is a
    worse result in practice than a slightly less relevant, well-stocked,
    highly-rated one.
    """
    if not results:
        return results

    max_rrf = max(r["rrf_score"] for r in results) or 1.0

    for r in results:
        normalized_rrf = r["rrf_score"] / max_rrf
        normalized_rating = r["rating"] / 5.0
        stock_bonus = 1.0 if r["stock"] > 0 else 0.0

        r["final_score"] = round(
            (0.6 * normalized_rrf) + (0.25 * normalized_rating) + (0.15 * stock_bonus),
            5
        )

    return sorted(results, key=lambda r: r["final_score"], reverse=True)