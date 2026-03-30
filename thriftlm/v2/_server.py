"""
FastAPI V2 endpoints — Phase 1.

Endpoints:
    POST /v2/plan/lookup   — task + context → FilledPlan or miss
    POST /v2/plan/store    — persist a plan template (hash re-verified server-side)
    GET  /v2/metrics       — minimal Phase 1 health payload
"""
from __future__ import annotations

import os
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from thriftlm.v2.adapter import (
    SlotFillError,
    SlotTypeError,
    TransformExecutionError,
    TransformNotFoundError,
    adapt_plan,
)
from thriftlm.embedder import Embedder
from thriftlm.v2.intent import canonicalize, compute_bucket_hash
from thriftlm.v2.plan_cache import PlanCache
from thriftlm.v2.schemas import IntentKey, PlanTemplate
from thriftlm.v2.validator import VALIDATOR_VERSION, validate_plan

app = FastAPI(title="ThriftLM V2", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Client / dependency helpers — called per-request, not at module level
# ---------------------------------------------------------------------------

def _make_supabase_client():
    """
    Instantiate and return a supabase-py client.

    Reads SUPABASE_URL and SUPABASE_KEY from the environment.
    Raises RuntimeError (→ HTTP 500) if either is missing.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url:
        raise RuntimeError("SUPABASE_URL env var is not set")
    if not key:
        raise RuntimeError("SUPABASE_KEY env var is not set")
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        raise RuntimeError("supabase package not installed — run: pip install thriftlm[api]")


class _InMemoryCanonCache:
    """
    Process-scoped canonicalization cache.

    Always consulted first — regardless of whether Redis is configured.
    Guarantees that within one server process the same task always resolves
    to the same bucket hash, even when Redis is unavailable or slow.
    No TTL; entries live for the lifetime of the process.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, task: str):
        result = self._store.get(task)
        if result is None:
            return None
        return {**result, "from_cache": True}

    def set(self, task: str, result) -> None:
        self._store[task] = result


_in_memory_canon_cache = _InMemoryCanonCache()


def _make_redis_cache():
    """Return a Redis-backed CanonicalizationCache, or None if REDIS_URL is unset."""
    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        return None
    from thriftlm.v2.canonicalization_cache import CanonicalizationCache
    return CanonicalizationCache(redis_url)


def _get_or_canonicalize(task: str):
    """
    Layered canonicalization lookup: in-memory → Redis → OpenAI.

    1. In-memory cache always checked first — guarantees same bucket within a
       server process lifetime even when Redis is down.
    2. Redis checked on in-memory miss — provides cross-process/cross-restart
       persistence in production.
    3. OpenAI called only on a full cache miss; result stored in both layers.

    Returns CanonicalizationResult, or None if OpenAI fails/times out.
    """
    result = _in_memory_canon_cache.get(task)
    if result is not None:
        return result

    redis_cache = _make_redis_cache()
    if redis_cache is not None:
        result = redis_cache.get(task)
        if result is not None:
            _in_memory_canon_cache.set(task, result)
            return result

    result = canonicalize(task)
    if result is not None:
        _in_memory_canon_cache.set(task, result)
        if redis_cache is not None:
            redis_cache.set(task, result)
    return result


def _build_structural_signature(plan: PlanTemplate) -> dict[str, Any]:
    """
    Derive structural_signature server-side from the template.

    Never trusts the caller to provide this — always computed fresh.
    """
    intent_key = plan["intent_key"]
    required_context_keys = sorted({
        slot["source"]
        for slot in plan.get("slots", [])
        if slot.get("required")
    })
    tool_families = sorted({
        step["tool_family"]
        for step in plan.get("steps", [])
        if step.get("tool_family")
    })
    has_side_effects = any(
        step.get("side_effect") is True
        for step in plan.get("steps", [])
    )
    return {
        "required_context_keys": required_context_keys,
        "tool_families": tool_families,
        "has_side_effects": has_side_effects,
        "format": intent_key.get("format"),
        "audience": intent_key.get("audience"),
        "step_count": len(plan.get("steps", [])),
    }


def _store_plan(
    sb,
    api_key: str,
    plan: PlanTemplate,
    intent_bucket_hash: str,
    structural_signature: dict[str, Any],
    embedding: list[float],
) -> str | None:
    """Insert the plan into the plans table and return the assigned id."""
    result = (
        sb.table("plans")
        .insert({
            "api_key": api_key,
            "intent_key_json": plan["intent_key"],
            "intent_bucket_hash": intent_bucket_hash,
            "description": plan["description"],
            "embedding": embedding,
            "template_json": dict(plan),
            "output_schema_json": plan.get("output_schema", {}),
            "structural_signature": structural_signature,
            "plan_version": plan.get("plan_version", "1"),
            "canonicalizer_version": plan.get("canonicalizer_version", ""),
            "extractor_version": plan.get("extractor_version", ""),
            "validator_version": plan.get("validator_version", ""),
            "is_valid": True,
        })
        .execute()
    )
    return result.data[0]["id"] if result.data else None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LookupRequest(BaseModel):
    api_key: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    runtime_caps: dict[str, Any] = Field(default_factory=dict)


class StoreRequest(BaseModel):
    api_key: str
    plan: dict[str, Any]   # PlanTemplate — validated structurally, not via TypedDict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v2/plan/lookup")
def lookup(req: LookupRequest):
    """
    Main hit/miss entry point.

    Flow:
      1. Check canonicalization cache (Redis) before calling OpenAI.
      2. On Redis miss, call canonicalize(); cache the result.
      3. Fetch + rerank plan candidates via PlanCache.
      4. For each candidate: adapt_plan → validate_plan.
      5. Return first passing candidate as a hit, or miss if all fail.
    """
    if not req.api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    if not req.task:
        raise HTTPException(status_code=400, detail="task is required")

    # Step 1 — canonicalize (layered: in-memory → Redis → OpenAI)
    canon_result = _get_or_canonicalize(req.task)
    if canon_result is None:
        return {"status": "miss", "reason": "canonicalization_failed"}

    # Step 2 — fetch + rerank
    try:
        sb = _make_supabase_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    plan_cache = PlanCache(supabase_client=sb, api_key=req.api_key)
    candidates = plan_cache.get(
        intent_bucket_hash=canon_result["intent_bucket_hash"],
        task=req.task,
        context=req.context,
        runtime_caps=req.runtime_caps,
    )

    # Step 3 — adapt + validate (first passing candidate wins)
    for candidate in candidates:
        plan = candidate["plan"]
        try:
            filled = adapt_plan(plan, req.context)
        except (SlotFillError, SlotTypeError, TransformNotFoundError, TransformExecutionError):
            continue

        validation = validate_plan(plan, filled, req.runtime_caps)
        if not validation["ok"]:
            continue

        return {
            "status": "hit",
            "filled_plan": filled,
            "canonicalization_result": canon_result,
            "matched_plan_id": plan["plan_id"],
            "score": candidate["final_score"],
        }

    return {
        "status": "miss",
        "reason": "no_valid_plan",
        "canonicalization_result": canon_result,
    }


_REQUIRED_PLAN_FIELDS: list[tuple[str, type]] = [
    ("intent_key", dict),
    ("intent_bucket_hash", str),
    ("description", str),
    ("steps", list),
    ("slots", list),
    ("output_schema", dict),
]


def _validate_plan_shape(plan: dict[str, Any]) -> None:
    """Raise HTTPException 400 if any required PlanTemplate field is missing or wrong type."""
    for field, expected_type in _REQUIRED_PLAN_FIELDS:
        if field not in plan or not isinstance(plan[field], expected_type):
            raise HTTPException(status_code=400, detail="invalid_plan_template")


@app.post("/v2/plan/store")
def store(req: StoreRequest):
    """
    Persist a plan template.

    Ordering (per spec):
      1. Validate plan shape.
      2. Verify hash — reject early before any expensive ops.
      3. Build structural_signature server-side.
      4. Compute embedding from description.
      5. Insert to DB.
    """
    if not req.api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    # Step 1 — validate required fields before touching intent_key
    _validate_plan_shape(req.plan)

    plan = cast(PlanTemplate, req.plan)

    # Step 2 — recompute + verify hash (never trust caller's value)
    try:
        intent_key = cast(IntentKey, plan["intent_key"])
        expected_hash = compute_bucket_hash(intent_key)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_intent_key")

    if expected_hash != plan.get("intent_bucket_hash"):
        raise HTTPException(status_code=400, detail="hash_mismatch")

    # Step 2 — structural_signature derived server-side
    sig = _build_structural_signature(plan)

    # Step 3 — embed description
    try:
        embedding = Embedder().embed(plan["description"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"embedding_failed: {exc}")

    # Step 4 — insert
    try:
        sb = _make_supabase_client()
        plan_id = _store_plan(
            sb=sb,
            api_key=req.api_key,
            plan=plan,
            intent_bucket_hash=expected_hash,
            structural_signature=sig,
            embedding=embedding,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"store_failed: {exc}")

    return {
        "status": "stored",
        "plan_id": plan_id or plan.get("plan_id"),
        "intent_bucket_hash": expected_hash,
    }


@app.get("/v2/metrics")
def metrics():
    """Minimal Phase 1 health payload."""
    return {
        "status": "ok",
        "version": VALIDATOR_VERSION,
        "phase": "phase_1",
    }
