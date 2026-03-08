"""
Cache routes.

POST /lookup  {embedding, api_key}                    → {response} or {response: null}
POST /store   {embedding, query, response, api_key}   → 200 OK
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.db import get_redis_backend, get_supabase_backend
from thriftlm.backends.redis_backend import RedisBackend
from thriftlm.backends.supabase_backend import SupabaseBackend

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LookupRequest(BaseModel):
    embedding: List[float]
    api_key: str


class LookupResponse(BaseModel):
    response: Optional[str] = None


class StoreRequest(BaseModel):
    embedding: List[float]
    query: str
    response: str
    api_key: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/lookup", response_model=LookupResponse)
async def lookup(
    body: LookupRequest,
    redis: RedisBackend = Depends(get_redis_backend),
    supabase: SupabaseBackend = Depends(get_supabase_backend),
) -> LookupResponse:
    """Find a semantically similar cached response.

    Checks Redis first (exact embedding hash); on a miss falls through to
    Supabase pgvector cosine similarity search. A Supabase hit is written
    back to Redis before returning. Returns ``{response: null}`` on a miss.
    """
    # 1. Redis fast-path (microseconds, exact hash match).
    cached = redis.get(body.embedding)
    if cached is not None:
        return LookupResponse(response=cached)

    # 2. Supabase vector search (milliseconds, cosine similarity).
    cached = supabase.lookup(body.embedding, body.api_key)
    if cached is not None:
        redis.set(body.embedding, cached)
        return LookupResponse(response=cached)

    return LookupResponse(response=None)


@router.post("/store", status_code=status.HTTP_200_OK)
async def store(
    body: StoreRequest,
    redis: RedisBackend = Depends(get_redis_backend),
    supabase: SupabaseBackend = Depends(get_supabase_backend),
) -> dict:
    """Store a new cache entry in both Supabase and Redis.

    Inserts the embedding, raw query, and LLM response into the
    ``cache_entries`` table, then writes the same response to Redis
    keyed by the embedding hash.
    """
    supabase.store(body.query, body.response, body.embedding, body.api_key)
    redis.set(body.embedding, body.response)
    return {"status": "ok"}
