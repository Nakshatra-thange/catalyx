import re

# Known categories from our catalog — used to detect category mentions in queries.
KNOWN_CATEGORIES = [
    "Jackets", "Shoes", "Backpacks", "Tents", "Sleeping Bags",
    "Headlamps", "Water Bottles", "Hiking Poles", "Gloves", "Socks"
]

PRICE_UNDER_PATTERN = re.compile(r"under\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
PRICE_OVER_PATTERN = re.compile(r"over\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
PRICE_BETWEEN_PATTERN = re.compile(
    r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE
)


def parse_query(raw_query: str) -> dict:
    """
    Splits a free-text query into:
      - semantic_text: the cleaned query to send to BM25/vector search
      - filters: structured constraints (price_min, price_max, category)

    This is intentionally rule-based, not LLM-based — it's fast, free,
    deterministic, and easy to unit test. An LLM-based parser would be
    a reasonable upgrade later but adds latency/cost for a task this
    pattern-matchable.
    """
    filters = {"price_min": None, "price_max": None, "category": None}
    cleaned = raw_query

    between_match = PRICE_BETWEEN_PATTERN.search(cleaned)
    if between_match:
        low, high = float(between_match.group(1)), float(between_match.group(2))
        filters["price_min"] = min(low, high)
        filters["price_max"] = max(low, high)
        cleaned = PRICE_BETWEEN_PATTERN.sub("", cleaned)
    else:
        under_match = PRICE_UNDER_PATTERN.search(cleaned)
        if under_match:
            filters["price_max"] = float(under_match.group(1))
            cleaned = PRICE_UNDER_PATTERN.sub("", cleaned)

        over_match = PRICE_OVER_PATTERN.search(cleaned)
        if over_match:
            filters["price_min"] = float(over_match.group(1))
            cleaned = PRICE_OVER_PATTERN.sub("", cleaned)

    for category in KNOWN_CATEGORIES:
        singular = category[:-1] if category.endswith("s") else category
        if re.search(rf"\b{re.escape(singular)}\b", cleaned, re.IGNORECASE) or \
           re.search(rf"\b{re.escape(category)}\b", cleaned, re.IGNORECASE):
            filters["category"] = category
            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return {
        "semantic_text": cleaned if cleaned else raw_query,
        "filters": filters,
    }


if __name__ == "__main__":
    tests = [
        "waterproof jacket under $100",
        "hiking shoes between $50 and $150",
        "backpacks over 80 dollars",
        "warm gloves",
    ]
    for t in tests:
        print(t, "->", parse_query(t))