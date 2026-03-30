"""
FastAPI app served by `thriftlm serve`.

Endpoints:
    GET /         — dashboard HTML (thriftlm/static/dashboard.html) with
                    injected <meta> tags for api-url and api-key so the
                    dashboard auto-connects on load.
    GET /metrics  — validates X-API-Key, queries Supabase, returns JSON.
    GET /health   — liveness probe.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="ThriftLM serve", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"


def _expected_key() -> str:
    return os.environ.get("THRIFTLM_SERVE_API_KEY", "")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/metrics")
async def metrics(x_api_key: str = Header(..., alias="X-API-Key")):
    expected = _expected_key()
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL and SUPABASE_KEY must be set in your environment.",
        )

    try:
        from supabase import create_client
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="supabase package not installed. Run: pip install thriftlm[api]",
        )

    client = create_client(supabase_url, supabase_key)

    # ── aggregate metrics from api_keys table ──────────────────────────────
    agg = (
        client.table("api_keys")
        .select("total_queries,total_hits,tokens_saved")
        .eq("api_key", x_api_key)
        .maybe_single()
        .execute()
    )

    row = agg.data or {}
    total_queries: int = row.get("total_queries") or 0
    total_hits: int = row.get("total_hits") or 0
    tokens_saved: int = row.get("tokens_saved") or 0

    hit_rate: float = total_hits / total_queries if total_queries > 0 else 0.0
    cost_saved: float = round(tokens_saved / 1000 * 0.002, 4)

    # ── top 5 queries by hit count ─────────────────────────────────────────
    top_result = (
        client.table("cache_entries")
        .select("query,hit_count,last_hit_at")
        .eq("api_key", x_api_key)
        .order("hit_count", desc=True)
        .limit(5)
        .execute()
    )

    top_queries = []
    now = datetime.now(timezone.utc)
    for r in top_result.data or []:
        raw_ts = r.get("last_hit_at")
        if raw_ts:
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                secs = int((now - dt).total_seconds())
                if secs < 60:
                    last_hit = f"{secs}s ago"
                elif secs < 3600:
                    last_hit = f"{secs // 60}m ago"
                else:
                    last_hit = f"{secs // 3600}h ago"
            except Exception:
                last_hit = raw_ts
        else:
            last_hit = "—"

        top_queries.append(
            {"query": r.get("query", ""), "hits": r.get("hit_count", 0), "last_hit": last_hit}
        )

    return JSONResponse(
        {
            "hit_rate": round(hit_rate, 4),
            "total_queries": total_queries,
            "tokens_saved": tokens_saved,
            "cost_saved": cost_saved,
            "top_queries": top_queries,
        }
    )
