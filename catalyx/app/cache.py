import os
import json
import hashlib
import redis

_client = None
CACHE_TTL_SECONDS = 300  # 5 minutes


def get_client():
    global _client
    if _client is None:
        _client = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    return _client


def make_cache_key(query: str, top_k: int) -> str:
    raw = f"search:{query.lower().strip()}:{top_k}"
    return "catalyx:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached(query: str, top_k: int) -> dict | None:
    client = get_client()
    key = make_cache_key(query, top_k)
    cached = client.get(key)
    if cached:
        return json.loads(cached)
    return None


def set_cached(query: str, top_k: int, response: dict):
    client = get_client()
    key = make_cache_key(query, top_k)
    client.setex(key, CACHE_TTL_SECONDS, json.dumps(response))