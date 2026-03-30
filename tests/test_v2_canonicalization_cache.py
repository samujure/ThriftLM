"""
Tests for thriftlm.v2.canonicalization_cache.CanonicalizationCache.

The Redis client is replaced with a MagicMock — no real network calls.
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from thriftlm.v2.canonicalization_cache import CanonicalizationCache
from thriftlm.v2.schemas import CanonicalizationResult, IntentKey

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

REDIS_URL = "redis://localhost:6379"

_INTENT_KEY: IntentKey = {
    "action":     "summarize",
    "target":     "pull_requests",
    "goal":       "identify_blockers",
    "time_scope": None,
    "tool_family": "github",
}

_RESULT: CanonicalizationResult = CanonicalizationResult(
    intent_key=_INTENT_KEY,
    intent_bucket_hash="abc123def456abcd",
    confidence=0.95,
    canonicalizer_version="v0.4",
    raw_canonical_text='{"action":"summarize","target":"pull_requests"}',
    from_cache=False,
)


def _cache_with_mock_client(stored: str | None = None, ttl: int = 3600):
    """Return a CanonicalizationCache whose Redis client is fully mocked."""
    cache = CanonicalizationCache(redis_url=REDIS_URL, ttl_seconds=ttl)
    mock_client = MagicMock()
    mock_client.get.return_value = stored
    cache._client = mock_client
    return cache, mock_client


# ---------------------------------------------------------------------------
# make_key
# ---------------------------------------------------------------------------

class TestMakeKey:
    def test_deterministic(self):
        cache = CanonicalizationCache(REDIS_URL)
        assert cache.make_key("hello") == cache.make_key("hello")

    def test_same_task_same_key(self):
        c1 = CanonicalizationCache(REDIS_URL)
        c2 = CanonicalizationCache(REDIS_URL)
        assert c1.make_key("summarize PRs") == c2.make_key("summarize PRs")

    def test_different_task_different_key(self):
        cache = CanonicalizationCache(REDIS_URL)
        assert cache.make_key("task A") != cache.make_key("task B")

    def test_prefix(self):
        cache = CanonicalizationCache(REDIS_URL)
        assert cache.make_key("x").startswith("tlm:v2:canon:")

    def test_key_is_sha256_of_raw_text(self):
        cache = CanonicalizationCache(REDIS_URL)
        task = "Summarize open pull requests"
        expected = "tlm:v2:canon:" + hashlib.sha256(task.encode()).hexdigest()
        assert cache.make_key(task) == expected

    def test_no_normalization(self):
        """Case differences must produce different keys (raw text, no lowercasing)."""
        cache = CanonicalizationCache(REDIS_URL)
        assert cache.make_key("Summarize PRs") != cache.make_key("summarize prs")


# ---------------------------------------------------------------------------
# get — miss paths
# ---------------------------------------------------------------------------

class TestGetMiss:
    def test_missing_key_returns_none(self):
        cache, _ = _cache_with_mock_client(stored=None)
        assert cache.get("any task") is None

    def test_malformed_json_returns_none(self):
        cache, _ = _cache_with_mock_client(stored="not valid json {")
        assert cache.get("any task") is None

    def test_non_dict_json_returns_none(self):
        cache, _ = _cache_with_mock_client(stored=json.dumps([1, 2, 3]))
        assert cache.get("any task") is None

    def test_missing_intent_key_field_returns_none(self):
        data = {
            "intent_bucket_hash":    "abc",
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_missing_intent_bucket_hash_returns_none(self):
        data = {
            "intent_key":            {},
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_missing_confidence_returns_none(self):
        data = {
            "intent_key":            {},
            "intent_bucket_hash":    "abc",
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_wrong_type_intent_bucket_hash_returns_none(self):
        data = {
            "intent_key":            {},
            "intent_bucket_hash":    12345,      # wrong type
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_wrong_type_intent_key_returns_none(self):
        data = {
            "intent_key":            "not a dict",   # wrong type
            "intent_bucket_hash":    "abc",
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_intent_key_wrong_field_type_returns_none(self):
        """action is int, not str — must be rejected."""
        data = {
            "intent_key":            {"action": 123, "target": "prs", "goal": "find", "time_scope": None},
            "intent_bucket_hash":    "abc",
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None

    def test_intent_key_constraints_non_string_members_returns_none(self):
        """constraints list contains ints, not strings — must be rejected."""
        data = {
            "intent_key":            {"action": "triage", "target": "issues", "goal": "classify",
                                      "time_scope": None, "constraints": [1, 2, 3]},
            "intent_bucket_hash":    "abc",
            "confidence":            0.9,
            "canonicalizer_version": "v0.4",
            "raw_canonical_text":    "{}",
        }
        cache, _ = _cache_with_mock_client(stored=json.dumps(data))
        assert cache.get("any task") is None


# ---------------------------------------------------------------------------
# get — hit path
# ---------------------------------------------------------------------------

class TestGetHit:
    def _stored_payload(self, from_cache: bool = False) -> str:
        return json.dumps({
            "intent_key":            dict(_INTENT_KEY),
            "intent_bucket_hash":    _RESULT["intent_bucket_hash"],
            "confidence":            _RESULT["confidence"],
            "canonicalizer_version": _RESULT["canonicalizer_version"],
            "raw_canonical_text":    _RESULT["raw_canonical_text"],
            "from_cache":            from_cache,
        })

    def test_returns_canonicalization_result(self):
        cache, _ = _cache_with_mock_client(stored=self._stored_payload())
        result = cache.get("summarize PRs")
        assert result is not None
        assert result["intent_bucket_hash"] == _RESULT["intent_bucket_hash"]
        assert result["confidence"] == _RESULT["confidence"]
        assert result["canonicalizer_version"] == _RESULT["canonicalizer_version"]

    def test_from_cache_always_true(self):
        """from_cache must be True even if stored value had False."""
        cache, _ = _cache_with_mock_client(stored=self._stored_payload(from_cache=False))
        result = cache.get("summarize PRs")
        assert result is not None
        assert result["from_cache"] is True

    def test_intent_key_fields_preserved(self):
        cache, _ = _cache_with_mock_client(stored=self._stored_payload())
        result = cache.get("summarize PRs")
        assert result is not None
        assert result["intent_key"]["action"] == "summarize"
        assert result["intent_key"]["tool_family"] == "github"


# ---------------------------------------------------------------------------
# set → get round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_set_then_get(self):
        cache = CanonicalizationCache(redis_url=REDIS_URL)
        mock_client = MagicMock()
        cache._client = mock_client

        task = "Summarize open PRs and find blockers"
        cache.set(task, _RESULT)

        # Capture what was stored
        assert mock_client.setex.called
        key_arg, ttl_arg, payload_arg = mock_client.setex.call_args.args

        assert key_arg == cache.make_key(task)
        assert ttl_arg == cache.ttl_seconds

        # Replay through get()
        mock_client.get.return_value = payload_arg
        result = cache.get(task)

        assert result is not None
        assert result["from_cache"] is True
        assert result["intent_bucket_hash"] == _RESULT["intent_bucket_hash"]
        assert result["confidence"] == _RESULT["confidence"]
        assert result["raw_canonical_text"] == _RESULT["raw_canonical_text"]

    def test_ttl_passed_to_redis(self):
        cache = CanonicalizationCache(redis_url=REDIS_URL, ttl_seconds=7200)
        mock_client = MagicMock()
        cache._client = mock_client

        cache.set("any task", _RESULT)
        _, ttl_arg, _ = mock_client.setex.call_args.args
        assert ttl_arg == 7200

    def test_default_ttl_is_3600(self):
        cache = CanonicalizationCache(redis_url=REDIS_URL)
        assert cache.ttl_seconds == 3600
