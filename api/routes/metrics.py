"""
Metrics route.

GET /metrics  header: X-API-Key  →  {hit_rate, tokens_saved, cost_saved, total_queries}

Reads aggregated counters from the api_keys table in Supabase.
Cost is estimated at $0.000002 per token saved (GPT-4o blended rate).
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import require_api_key
from api.db import get_supabase_client

router = APIRouter()

# Cost per token saved (USD). Based on GPT-4o blended input/output pricing.
_COST_PER_TOKEN_USD = 0.000002


class MetricsResponse(BaseModel):
    hit_rate: float
    tokens_saved: int
    cost_saved: float
    total_queries: int


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    api_key: str = Depends(require_api_key),
    client: Any = Depends(get_supabase_client),
) -> MetricsResponse:
    """Return aggregated cache metrics for the authenticated API key.

    Reads ``total_hits``, ``total_queries``, and ``tokens_saved`` from the
    ``api_keys`` table and derives ``hit_rate`` and ``cost_saved``.

    Returns:
        :class:`MetricsResponse` with computed statistics.

    Raises:
        HTTPException 404: If the API key has no record in the database.
    """
    result = (
        client.table("api_keys")
        .select("total_hits, total_queries, tokens_saved")
        .eq("api_key", api_key)
        .maybe_single()
        .execute()
    )

    if result.data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found.",
        )

    row = result.data
    total_queries: int = row.get("total_queries", 0) or 0
    total_hits: int = row.get("total_hits", 0) or 0
    tokens_saved: int = row.get("tokens_saved", 0) or 0

    hit_rate = total_hits / total_queries if total_queries > 0 else 0.0
    cost_saved = round(tokens_saved * _COST_PER_TOKEN_USD, 6)

    return MetricsResponse(
        hit_rate=hit_rate,
        tokens_saved=tokens_saved,
        cost_saved=cost_saved,
        total_queries=total_queries,
    )
