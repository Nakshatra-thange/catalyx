BENCHMARK = [
    # --- Literal keyword-match queries (BM25 should do well) ---
    {
        "query": "waterproof jacket for hiking",
        "relevant_ids": {892, 987, 1706, 2464, 2673, 4786, 5581, 6691, 7085, 7616},
    },
    {
        "query": "lightweight backpack under $100",
        "relevant_ids": {611, 4871, 3263, 2253, 5939, 1792, 7127, 6148, 5840, 3211},
    },
    {
        "query": "insulated gloves for cold weather",
        "relevant_ids": {518, 1779, 4928, 5086},
    },
    {
        "query": "durable hiking pole",
        "relevant_ids": {59, 505, 771, 949, 1306},
    },
    {
        "query": "breathable tent for camping",
        "relevant_ids": {936, 1345, 2216, 3266, 3691, 4684, 5179, 5321, 6173, 6786},
    },

    # --- Semantic/paraphrase queries (no literal keyword overlap — tests
    #     whether dense/hybrid retrieval adds value over BM25 alone) ---
    {
        "query": "warm gear for cold weather",
        "relevant_ids": {518, 4301, 6092, 6280},
    },
    {
        "query": "protection from strong wind",
        "relevant_ids": {1, 4862, 7431, 7657, 5928, 5825, 2475, 2252, 7155, 4204},
    },
    {
        "query": "equipment that packs down small",
        "relevant_ids": {195, 5339, 1732, 4441, 3681, 6172},
    },
    {
        "query": "gear that keeps you dry in the rain",
        "relevant_ids": {4639, 6859, 4609, 7868, 7239, 3770, 3537, 409, 5643, 7233, 5474},
    },
]