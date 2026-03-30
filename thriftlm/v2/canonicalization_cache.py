from __future__ import annotations

import hashlib
import json
from typing import cast

from thriftlm.v2.schemas import CanonicalizationResult, IntentKey

_KEY_PREFIX = "tlm:v2:canon:"
_REQUIRED_FIELDS = (
    "intent_key",
    "intent_bucket_hash",
    "confidence",
    "canonicalizer_version",
    "raw_canonical_text",
)
_INTENT_KEY_REQUIRED = ("action", "target", "goal", "time_scope")
_INTENT_KEY_OPTIONAL_STR = ("domain", "format", "audience", "tool_family")


def _is_valid_intent_key(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for field in _INTENT_KEY_REQUIRED:
        if field not in value:
            return False
    if not isinstance(value["action"], str):
        return False
    if not isinstance(value["target"], str):
        return False
    if not isinstance(value["goal"], str):
        return False
    if value["time_scope"] is not None and not isinstance(value["time_scope"], str):
        return False
    for field in _INTENT_KEY_OPTIONAL_STR:
        val = value.get(field)
        if val is not None and not isinstance(val, str):
            return False
    constraints = value.get("constraints")
    if constraints is not None:
        if not isinstance(constraints, list):
            return False
        if not all(isinstance(c, str) for c in constraints):
            return False
    return True


class CanonicalizationCache:
    """
    Thin Redis cache for CanonicalizationResult objects.

    Sits in front of intent.py — caller checks here first, only calls
    canonicalize() on a miss, then stores the result.

    Args:
        redis_url:   Connection URL (redis:// or rediss:// for TLS).
        ttl_seconds: TTL for stored entries. Default 3600 (1 hour).
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 3600) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client = None  # lazy-loaded on first use

    def _get_client(self):
        if self._client is None:
            from redis import Redis
            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def make_key(self, task: str) -> str:
        """sha256(raw_task_text) prefixed with tlm:v2:canon: — no normalization."""
        digest = hashlib.sha256(task.encode()).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    def get(self, task: str) -> CanonicalizationResult | None:
        """
        Return cached CanonicalizationResult for *task*, or None on miss.

        Sets from_cache=True on the returned object regardless of stored value.
        Returns None on missing key, JSON parse failure, or missing required fields.
        """
        try:
            raw = self._get_client().get(self.make_key(task))
        except Exception:
            return None

        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        for field in _REQUIRED_FIELDS:
            if field not in data:
                return None

        if not _is_valid_intent_key(data["intent_key"]):
            return None
        if not isinstance(data["intent_bucket_hash"], str):
            return None
        if not isinstance(data["confidence"], (int, float)):
            return None
        if not isinstance(data["canonicalizer_version"], str):
            return None
        if not isinstance(data["raw_canonical_text"], str):
            return None

        return CanonicalizationResult(
            intent_key=cast(IntentKey, data["intent_key"]),
            intent_bucket_hash=data["intent_bucket_hash"],
            confidence=float(data["confidence"]),
            canonicalizer_version=data["canonicalizer_version"],
            raw_canonical_text=data["raw_canonical_text"],
            from_cache=True,
        )

    def set(self, task: str, result: CanonicalizationResult) -> None:
        """Store *result* under the key derived from *task* with the configured TTL."""
        try:
            payload = json.dumps({
                "intent_key":            dict(result["intent_key"]),
                "intent_bucket_hash":    result["intent_bucket_hash"],
                "confidence":            result["confidence"],
                "canonicalizer_version": result["canonicalizer_version"],
                "raw_canonical_text":    result["raw_canonical_text"],
                "from_cache":            result["from_cache"],
            })
            self._get_client().setex(self.make_key(task), self.ttl_seconds, payload)
        except Exception:
            pass  # cache write failures are non-fatal
