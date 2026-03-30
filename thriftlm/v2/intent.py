from __future__ import annotations

import hashlib
import json
import os
from typing import Any


import httpx

from thriftlm.v2.schemas import CanonicalizationResult, IntentKey

CANONICALIZER_VERSION = "v0.4"

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_TIMEOUT = 10.0  # 10s — sized for OpenAI gpt-4o-mini (~2-3s p50)

_SYSTEM_PROMPT = """\
You are a task intent canonicalizer. Given a task description, extract a \
structured intent and return ONLY valid JSON with exactly these fields:

{
  "action":      <str — verb describing what to do, e.g. "summarize">,
  "target":      <str — object being acted on, e.g. "pull_requests">,
  "goal":        <str — desired outcome, e.g. "identify_blockers">,
  "time_scope":  <str | null — time window if mentioned, else null>,
  "domain":      <str | null — subject domain if specific, else null>,
  "format":      <str | null — output format if specified, else null>,
  "audience":    <str | null — intended audience if mentioned, else null>,
  "constraints": <list[str] | null — constraint strings if any, else null>,
  "tool_family": <str | null — tool ecosystem implied, e.g. "github", else null>,
  "confidence":  <float 0.0–1.0 — your confidence in this canonicalization>
}

Return only the JSON object. No explanation, no markdown."""


_OPTIONAL_STR_FIELDS = ("domain", "format", "audience", "tool_family")


def _normalize_intent_key(intent_key: IntentKey) -> IntentKey:
    """
    Return a new IntentKey with all values normalized:
    - string fields: lowercase + strip
    - optional fields (domain/format/audience/tool_family): omitted when None
    - constraints: sorted + each item lowercased + stripped; omitted when None or empty
    - time_scope: kept even when None (required field)
    """
    ts = intent_key["time_scope"]
    normalized: IntentKey = {
        "action":     intent_key["action"].lower().strip(),
        "target":     intent_key["target"].lower().strip(),
        "goal":       intent_key["goal"].lower().strip(),
        "time_scope": ts.lower().strip() if ts is not None else None,
    }
    for field in _OPTIONAL_STR_FIELDS:
        val = intent_key.get(field)  # type: ignore[literal-required]
        if val is not None:
            normalized[field] = val.lower().strip()  # type: ignore[literal-required]
    constraints = intent_key.get("constraints")
    if constraints:
        normalized["constraints"] = sorted(c.lower().strip() for c in constraints)
    return normalized


_HASH_FIELDS = ("action", "target", "goal", "time_scope")


def compute_bucket_hash(intent_key: IntentKey) -> str:
    """
    Deterministic 16-char hex hash based on the 4 stable core fields only:
    action, target, goal, time_scope.

    Optional fields (domain, format, audience, constraints, tool_family) are
    intentionally excluded — they are LLM-unstable across invocations and
    should not affect bucket identity.  They remain available on the IntentKey
    for scoring and metadata but must not change the bucket a plan is filed under.
    """
    normalized = _normalize_intent_key(intent_key)
    serializable = {
        k: normalized[k]  # type: ignore[literal-required]
        for k in _HASH_FIELDS
        if normalized.get(k) is not None  # type: ignore[literal-required]
    }
    serialized = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _call_openai(task: str) -> tuple[str, float] | None:
    """
    Call OpenAI and return (raw_json_text, confidence) or None on any failure.

    Reads OPENAI_API_KEY from env.
    Returns None if:
    - API key missing
    - timeout
    - HTTP error
    - bad response shape
    - JSON parse failure
    - confidence missing / invalid
    - confidence < 0.85
    - any exception
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    try:
        resp = httpx.post(
            _OPENAI_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw_text: str = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

    try:
        data: dict[str, Any] = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError):
        return None

    if confidence < 0.85:
        return None

    return raw_text, confidence


def canonicalize(task: str) -> CanonicalizationResult | None:
    """
    Canonicalize *task* into a structured IntentKey via OpenAI.

    Returns None on timeout, parse failure, exception, or confidence < 0.85.
    Does not perform caching — that is handled by canonicalization_cache.py.
    """
    result = _call_openai(task)
    if result is None:
        return None

    raw_text, confidence = result

    try:
        data: dict[str, Any] = json.loads(raw_text)
        raw_key: IntentKey = {
            "action":     str(data["action"]),
            "target":     str(data["target"]),
            "goal":       str(data["goal"]),
            "time_scope": data.get("time_scope"),
        }
        for field in _OPTIONAL_STR_FIELDS:
            val = data.get(field)
            if val is not None:
                raw_key[field] = str(val)  # type: ignore[literal-required]
        constraints = data.get("constraints")
        if isinstance(constraints, list) and constraints:
            raw_key["constraints"] = [str(c) for c in constraints]
    except (KeyError, TypeError):
        return None

    intent_key = _normalize_intent_key(raw_key)
    intent_bucket_hash = compute_bucket_hash(intent_key)

    return CanonicalizationResult(
        intent_key=intent_key,
        intent_bucket_hash=intent_bucket_hash,
        confidence=confidence,
        canonicalizer_version=CANONICALIZER_VERSION,
        raw_canonical_text=raw_text,
        from_cache=False,
    )
