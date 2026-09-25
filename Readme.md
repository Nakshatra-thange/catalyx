# Catalyx

A hybrid semantic search engine for a product catalog, built to understand retrieval systems from first principles rather than wrapping an LLM API. No generation step — the product *is* the retrieval and ranking pipeline.

## What it does

Given a free-text query like `"waterproof jacket under $100"`, Catalyx:

1. **Parses** structured filters (price, category) out of the free-text query
2. **Retrieves** candidates via two independent methods:
   - BM25 keyword search (implemented from scratch)
   - Dense vector search over sentence embeddings (pgvector, cosine similarity)
3. **Fuses** both ranked lists using Reciprocal Rank Fusion (implemented from scratch)
4. **Reranks** results by blending relevance with rating and stock availability
5. **Caches** and **paginates** the final response

## Architecture

```
                        ┌─────────────┐
   client ── HTTP ──▶   │   FastAPI   │
                        │  (main.py)  │
                        └──────┬──────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                       │
        ▼                      ▼                       ▼
 ┌─────────────┐      ┌────────────────┐      ┌───────────────┐
 │ query_parser │      │  search_index   │      │ vector_search │
 │ (filters)    │      │  (BM25, memory) │      │  (pgvector)   │
 └─────────────┘      └────────────────┘      └───────┬───────┘
                               │                       │
                               └───────────┬───────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │  rrf.py (fuse)   │
                                   └────────┬─────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │  reranker.py     │
                                   └────────┬─────────┘
                                            ▼
                         ┌──────────────────┴──────────────────┐
                         ▼                                     ▼
                 ┌───────────────┐                    ┌────────────────┐
                 │ Redis (cache) │                     │ Postgres +     │
                 │ TTL 5 min     │                     │ pgvector       │
                 └───────────────┘                     │ (products,     │
                                                        │  embeddings)   │
                                                        └────────────────┘
```

| Component | Responsibility | If it fails |
|---|---|---|
| FastAPI (`main.py`) | Request validation, routing, logging, orchestration | Requests get a structured `422`/`500` instead of a silent crash |
| BM25 index (`bm25.py`, in-memory) | Keyword-relevance scoring, built once at startup | `/search/keyword` and the BM25 half of hybrid fail; `/search/reindex` rebuilds it |
| pgvector (Postgres) | Stores product embeddings, serves cosine-similarity search | Semantic and hybrid search fail; keyword search still works — degraded, not fully down |
| Redis | Caches repeated `/search` responses (5 min TTL) | Cache miss on every request — slower but still correct, since results are recomputed, not stale |
| Reranker (`reranker.py`) | Blends RRF relevance with rating/stock into a final order | Falls back to raw RRF order — still relevant, just unweighted by business signals |

## AI pipeline

**Chunking / normalization:** each product's title, description, category, and brand are concatenated into a single embedding-ready string (`ingest.py`).

**Embedding:** `all-MiniLM-L6-v2` (sentence-transformers), 384-dim, normalized to unit length so cosine similarity reduces to a dot product.

**BM25 (from scratch, `bm25.py`):** standard Okapi BM25 — term frequency, inverse document frequency, and document-length normalization, with `k1=1.5`, `b=0.75`. No library used; every term in the formula is implemented directly.

**Dense retrieval:** pgvector's `<=>` cosine-distance operator over the stored embeddings. Cosine similarity was also implemented manually once (`cosine_manual.py`) as a correctness check before relying on pgvector's optimized version.

**Fusion — Reciprocal Rank Fusion (from scratch, `rrf.py`):**

```
score(d) = Σ over ranked lists L containing d:  1 / (k + rank_L(d))     (k = 60)
```

RRF was chosen over blending raw scores because BM25 scores and cosine similarities live on incompatible scales (BM25 is roughly 0–20+, cosine similarity is –1 to 1). RRF only needs rank position, sidestepping normalization entirely.

**Query understanding (`query_parser.py`):** rule-based regex extraction of price ranges (`under $X`, `over $X`, `between $X and $Y`) and category keywords — deliberately not LLM-based, since it's fast, free, deterministic, and easy to unit test for this scope.

**Filtering:** pre-filtering — structured constraints are applied in SQL *before* BM25/vector candidates are ranked, guaranteeing every returned result satisfies the filter. Trade-off: a narrow filter combined with a small candidate pool can return fewer than `top_k` results. Post-filtering with a widening search radius would fix this at the cost of added complexity, and was intentionally out of scope here.

**Reranking (`reranker.py`, from scratch):** a hand-built weighted scorer, not a cross-encoder model:

```
final_score = 0.6 × normalized_rrf + 0.25 × normalized_rating + 0.15 × stock_bonus
```

Chosen for transparency and tunability over an opaque model, at this project's scale.

## Evaluation

### Methodology

A hand-verified benchmark of 16 queries, evenly split between two categories:

- **Literal queries (8):** brand + adjective + category combinations that appear near-verbatim in product titles (e.g. `"PeakGear breathable tent"`) — BM25 should excel here.
- **Semantic queries (8):** paraphrased descriptions with **no literal keyword overlap** with the target adjective (e.g. `"rain shelter to pitch under $80"` for waterproof tents) — tests whether dense/hybrid retrieval adds value BM25 structurally cannot provide.

Every query's `relevant_ids` is the **exact, complete set** returned by a verified SQL predicate against the live catalog (brand/category/price-bounded), re-checked against Postgres on every eval run — not sampled from one method's own output, which was an early mistake in this project (see *Lessons learned* below).

### Results (K=10)

**Literal queries (8)**

| Method | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| BM25 only | 0.9643 | 0.7750 | 1.0000 |
| Hybrid (BM25 + Dense, RRF) | 0.9643 | 0.7750 | 1.0000 |
| Hybrid + Reranked | 0.9643 | 0.7750 | 1.0000 |

**Semantic queries (8)**

| Method | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| BM25 only | 0.0000 | 0.0000 | 0.0000 |
| Hybrid (BM25 + Dense, RRF) | 0.1114 | 0.1625 | 0.3167 |
| Hybrid + Reranked | 0.1114 | 0.1625 | 0.4449 |

### Interpretation

On literal queries, all three methods score identically — but not because hybrid is doing nothing. Manual inspection (see `docs/eval_notes.md` or below) confirms BM25 and hybrid return **different** top-10 lists (different order, different items in positions 7–10), they just both retrieve **all 6 relevant items inside the top 10** — a ceiling effect. When BM25 already finds 100% of what's relevant, there's no headroom left for hybrid to improve the score, even though it is genuinely fusing two independent signals under the hood.

On semantic queries — constructed to share **zero literal keywords** with the target concept — BM25 scores exactly 0.0000 across every metric, while hybrid retrieval recovers meaningful, non-zero recall, precision, and MRR. Reranking further improves MRR (0.3167 → 0.4449) by pushing the first genuinely relevant result higher, since stock/rating signals correlate positively with relevance in this dataset.

**Takeaway:** hybrid retrieval is not "always better" — it's equal to BM25 when a query is lexically easy, and the *only* thing standing between zero and meaningful recall when a query requires semantic understanding. This is a more precise and defensible claim than either "hybrid always wins" or "hybrid doesn't matter," and it's exactly the kind of trade-off a production ranking system has to reason about query-by-query.

### A finding worth stating explicitly: synthetic catalog limitations

Early benchmark queries (e.g. `"durable hiking pole"`, `"waterproof jacket for hiking"`) were built by sampling IDs out of one method's top-10 results rather than verifying the true relevant set. Auditing those queries against the full catalog later revealed the actual relevant-item counts were far larger than assumed:

| Original query | True matches in catalog |
|---|---|
| `waterproof jacket for hiking` | 73 |
| `insulated gloves for cold weather` | 82 |
| `breathable tent for camping` | 94 |
| `durable hiking pole` | 105 |
| `warm gear for cold weather` | 885 |
| `protection from strong wind` | 850 |
| `equipment that packs down small` | 905 |
| `gear that keeps you dry in the rain` | 888 |

Recall@10 is mathematically capped near `10 / true_count` — several of these queries could never have scored above ~0.10 regardless of retrieval quality, because the catalog's random brand × adjective × category generation produces heavy near-duplication. These queries were replaced with brand/price-bounded variants (6–19 true matches) so Recall@10 stays a meaningful metric. This was found and fixed during benchmark construction, and is a known, explicit limitation of using a randomly-generated synthetic catalog rather than real-world product data.

## Backend engineering

- **Validation:** Pydantic models (`schemas.py`) reject empty queries and out-of-range pagination before they reach search logic (`422`, not a silent bad response).
- **Pagination:** `page` / `page_size` params slice a fused, deduplicated candidate list rather than re-running retrieval per page.
- **Caching:** Redis, keyed per query + page, 5-minute TTL. TTL-based invalidation was chosen over explicit invalidation on catalog writes, because correctly invalidating only the cached queries a given product update could affect isn't cheap to compute at this scale — a deliberate, statable trade-off.
- **Observability:** every request gets a UUID `request_id`, logged on entry and exit (with status code and duration) and returned in the `X-Request-ID` response header, so any request can be traced end-to-end through the logs.

## Data architecture

| Store | Holds | Why |
|---|---|---|
| PostgreSQL | Product rows, metadata, embeddings (via pgvector) | Single source of truth; running one database instead of a separate vector DB keeps operational complexity down at this scale (~8K products) |
| Redis | Cached search responses | Fast, ephemeral, TTL-native — a natural fit for a cache that's allowed to go stale for a few minutes |
| In-memory (process) | BM25 inverted index | Rebuilt at startup / on `/search/reindex`; avoids persisting a second index structure for this project's scope |

## Tech stack

- **API:** FastAPI (Python) — async, Pydantic validation, minimal boilerplate
- **Database:** PostgreSQL + pgvector — one database instead of two (Postgres + a dedicated vector DB), justified at this scale; would reconsider at millions of products or heavier filtering load
- **Cache:** Redis
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers), run locally — avoids external API cost/latency during heavy benchmark iteration
- **Containerization:** Docker Compose (API + Redis; Postgres hosted on Neon/Supabase)

## What's implemented from scratch vs. using libraries

**From scratch:** BM25 (term frequency, IDF, length normalization), cosine similarity (manual proof-of-concept), Reciprocal Rank Fusion, Recall@K, Precision@K, MRR, the reranking scorer, rule-based query parsing.

**Using libraries:** sentence-transformers for embedding generation, pgvector for production-scale vector search (backed by the same cosine-similarity math proven manually), FastAPI/SQLAlchemy for the API/DB layer, Redis for caching.

## How to run

```bash
docker-compose up --build
curl http://localhost:8000/health
curl http://localhost:8000/health/db
curl http://localhost:8000/health/embeddings
curl "http://localhost:8000/search?q=waterproof%20jacket%20under%20%24100"
```

Requires a `.env` with `DATABASE_URL` (Postgres + pgvector, e.g. a free Neon or Supabase instance) and `REDIS_URL`. See `.env.example`.

To reproduce the evaluation:

```bash
docker exec -it catalyx_api python eval/run_eval.py
```

## Design decisions (interview-ready summary)

- **pgvector over a dedicated vector DB** — fewer moving parts at this scale; would switch to Qdrant/similar if filtering or index size became the bottleneck.
- **RRF over raw score blending** — sidesteps incompatible BM25/cosine score scales by using rank position only.
- **Pre-filtering over post-filtering** — guarantees filter correctness; accepts the risk of under-filling `top_k` on narrow filters.
- **Hand-built reranker over a cross-encoder** — transparent, tunable, and easier to debug than an opaque model at this project's scope.
- **TTL cache invalidation over explicit invalidation** — trades a small staleness window for not having to compute which cached queries a given write could affect.
- **Query-verified benchmark over method-sampled benchmark** — an evaluation set built from one retrieval method's own output silently caps that method's competitors' scores; ground truth has to come from the data itself.