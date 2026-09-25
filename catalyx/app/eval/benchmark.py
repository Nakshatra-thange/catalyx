"""Catalog-backed evaluation cases.

The old benchmark used a handful of IDs sampled from retrieval results.  That
made Recall@K incorrect: the synthetic catalog contains many near-duplicate
products for every adjective/category pair.  Each case below has a bounded
ground-truth predicate and the complete ID set returned by that predicate in
the 8,000-product catalog.  ``run_eval.py`` re-runs every predicate against
Postgres and fails if the stored set drifts.

All broad original queries (for example, ``waterproof jacket for hiking`` and
``warm gear for cold weather``) had well over 20 relevant products.  They were
replaced rather than scored with Recall@10, because a top-10 result cannot
meaningfully recall an open-ended set of near-identical catalog entries.
"""


BENCHMARK = [
    # --- Literal / keyword-overlap queries. BM25 should be strong here: the
    #     brand, adjective, and product type appear literally in product titles.
    {
        "query": "PeakGear breathable tent",
        "query_type": "literal",
        "ground_truth_where": "brand = 'PeakGear' AND title ILIKE '%Breathable%' AND category = 'Tents'",
        "relevant_ids": {1338, 1713, 4106, 4412, 4816, 5321},
    },
    {
        "query": "NorthRidge durable hiking pole",
        "query_type": "literal",
        "ground_truth_where": "brand = 'NorthRidge' AND title ILIKE '%Durable%' AND category = 'Hiking Poles'",
        "relevant_ids": {59, 2316, 3894, 4570, 5271, 6269, 6564, 7040, 7959},
    },
    {
        "query": "TrailMaster waterproof water bottle",
        "query_type": "literal",
        "ground_truth_where": "brand = 'TrailMaster' AND title ILIKE '%Waterproof%' AND category = 'Water Bottles'",
        "relevant_ids": {184, 1388, 3099, 3491, 3715, 5465},
    },
    {
        "query": "BaseCamp lightweight backpack",
        "query_type": "literal",
        "ground_truth_where": "brand = 'BaseCamp' AND title ILIKE '%Lightweight%' AND category = 'Backpacks'",
        "relevant_ids": {346, 705, 4269, 4717, 4720, 6149, 6732},
    },
    {
        "query": "RidgeLine quick-dry glove",
        "query_type": "literal",
        "ground_truth_where": "brand = 'RidgeLine' AND title ILIKE '%Quick-dry%' AND category = 'Gloves'",
        "relevant_ids": {1492, 1566, 1595, 1633, 2110, 3176, 3406, 3924, 5762, 7856},
    },
    {
        "query": "AlpineEdge packable sleeping bag",
        "query_type": "literal",
        "ground_truth_where": "brand = 'AlpineEdge' AND title ILIKE '%Packable%' AND category = 'Sleeping Bags'",
        "relevant_ids": {972, 2260, 4075, 5401, 5513, 7033},
    },
    {
        "query": "WildPath rugged sock",
        "query_type": "literal",
        "ground_truth_where": "brand = 'WildPath' AND title ILIKE '%Rugged%' AND category = 'Socks'",
        "relevant_ids": {133, 491, 1151, 1753, 2418, 5015, 6964, 6988},
    },
    {
        "query": "StormGuard windproof headlamp",
        "query_type": "literal",
        "ground_truth_where": "brand = 'StormGuard' AND title ILIKE '%Windproof%' AND category = 'Headlamps'",
        "relevant_ids": {232, 2205, 2767, 2980, 3824, 3955, 4841, 4862, 5533, 5910, 5928, 6002, 6642, 7294},
    },

    # --- Semantic / paraphrase queries. The query deliberately omits the
    #     catalog's adjective and category words; relevance is the exact
    #     concept/category/price predicate, not every same-category product.
    {
        "query": "warm hand protection under $80",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Insulated%' AND category = 'Gloves' AND price <= 80",
        "relevant_ids": {229, 1944, 2696, 2926, 3498, 3823, 4527, 5521, 5884, 6809, 6944},
    },
    {
        "query": "rain shelter to pitch under $80",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Waterproof%' AND category = 'Tents' AND price <= 80",
        "relevant_ids": {25, 1086, 1489, 2131, 2658, 2835, 2905, 3234, 3450, 4801, 4807, 4947, 5199, 5813, 5998, 6260, 6510, 6794, 7055},
    },
    {
        "query": "gear that folds into a small carry load under $50",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Packable%' AND category = 'Backpacks' AND price <= 50",
        "relevant_ids": {273, 1007, 1052, 1316, 1792, 2435, 2667, 3211, 6083, 6172, 6533},
    },
    {
        "query": "footwear that dries fast after a stream crossing under $70",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Quick-dry%' AND category = 'Socks' AND price <= 70",
        "relevant_ids": {434, 2286, 2723, 2737, 2879, 3195, 3256, 3552, 4299, 5457, 5932, 6836, 6897, 7133, 7428, 7717},
    },
    {
        "query": "gust-proof light for after dark under $70",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Windproof%' AND category = 'Headlamps' AND price <= 70",
        "relevant_ids": {649, 900, 2190, 2234, 5148, 5509, 6109, 6243, 6341, 6453, 7010, 7442},
    },
    {
        "query": "easy-to-carry foot covering under $80",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Lightweight%' AND category = 'Socks' AND price <= 80",
        "relevant_ids": {831, 948, 1362, 1499, 2021, 2312, 2805, 4223, 5703, 5988, 6179, 6207, 6624, 6871, 6910, 7041, 7503, 7967},
    },
    {
        "query": "tough container for a trail drink under $80",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Durable%' AND category = 'Water Bottles' AND price <= 80",
        "relevant_ids": {192, 645, 932, 986, 1347, 1436, 1917, 2178, 2386, 4236, 4953, 5170, 5404, 5437, 6435, 7060, 7703, 7722},
    },
    {
        "query": "warm bedroll for freezing camps under $70",
        "query_type": "semantic",
        "ground_truth_where": "title ILIKE '%Insulated%' AND category = 'Sleeping Bags' AND price <= 70",
        "relevant_ids": {183, 1327, 1609, 1766, 2115, 3109, 3453, 3598, 3626, 3702, 3867, 4124, 4433, 4614, 5235, 5725, 7832},
    },
]
