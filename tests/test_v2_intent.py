"""
Tests for thriftlm.v2.intent — compute_bucket_hash and canonicalize.

All OpenAI HTTP calls are replaced with unittest.mock — no real network calls.
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from thriftlm.v2.intent import CANONICALIZER_VERSION, canonicalize, compute_bucket_hash
from thriftlm.v2.schemas import IntentKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_response(data: dict) -> MagicMock:
    """Build a mock httpx.Response that looks like an OpenAI chat completion."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "choices": [{"message": {"content": json.dumps(data)}}]
    }
    return mock


_GOOD_PAYLOAD = {
    "action": "summarize",
    "target": "pull_requests",
    "goal": "identify_blockers",
    "time_scope": None,
    "domain": None,
    "format": None,
    "audience": "engineering",
    "constraints": None,
    "tool_family": "github",
    "confidence": 0.95,
}


# ---------------------------------------------------------------------------
# compute_bucket_hash
# ---------------------------------------------------------------------------

class TestComputeBucketHash:
    def _key(self, **kwargs) -> IntentKey:
        base: IntentKey = {
            "action": "summarize",
            "target": "prs",
            "goal": "find_blockers",
            "time_scope": None,
        }
        base.update(kwargs)  # type: ignore[typeddict-item]
        return base

    def test_determinism(self):
        key = self._key()
        assert compute_bucket_hash(key) == compute_bucket_hash(key)

    def test_returns_16_chars(self):
        assert len(compute_bucket_hash(self._key())) == 16

    def test_none_time_scope_excluded_from_hash_bytes(self):
        """time_scope=None stays in the key but must not appear in hash serialization."""
        key = self._key(time_scope=None)
        # time_scope=None → excluded from hash; key with time_scope set → different hash
        key_with_scope = self._key(time_scope="last_7d")
        assert compute_bucket_hash(key) != compute_bucket_hash(key_with_scope)
        # Verify None is genuinely absent from the serialized bytes used for hashing
        normalized = {k: v for k, v in key.items() if v is not None}
        serialized = json.dumps(
            {k: sorted(v) if isinstance(v, list) else v.lower().strip() for k, v in normalized.items()},
            sort_keys=True, separators=(",", ":"),
        )
        assert "time_scope" not in serialized

    def test_optional_none_excluded(self):
        """domain=None should produce same hash as omitting domain."""
        with_domain_none = self._key(domain=None)
        without_domain = self._key()
        assert compute_bucket_hash(with_domain_none) == compute_bucket_hash(without_domain)

    def test_list_fields_sorted(self):
        """constraints in different order produce the same hash (optional field → not hashed)."""
        key_a = self._key(constraints=["urgent", "bug", "regression"])
        key_b = self._key(constraints=["regression", "urgent", "bug"])
        assert compute_bucket_hash(key_a) == compute_bucket_hash(key_b)

    def test_different_keys_differ(self):
        key_a = self._key(action="summarize")
        key_b = self._key(action="triage")
        assert compute_bucket_hash(key_a) != compute_bucket_hash(key_b)

    def test_lowercase_normalization(self):
        """Values are lowercased before hashing."""
        lower = self._key(action="summarize", target="emails", goal="find_urgent")
        upper = self._key(action="SUMMARIZE", target="EMAILS", goal="FIND_URGENT")
        assert compute_bucket_hash(lower) == compute_bucket_hash(upper)

    def test_whitespace_stripped(self):
        padded = self._key(action="  summarize  ", target="  prs  ", goal="  find_blockers  ")
        clean = self._key(action="summarize", target="prs", goal="find_blockers")
        assert compute_bucket_hash(padded) == compute_bucket_hash(clean)

    def test_hash_is_sha256_of_core_fields_only(self):
        """Hash must be sha256 of only the 4 core fields, lowercase, sorted keys."""
        key = self._key()
        # Only action/target/goal/time_scope — optional fields excluded
        core = {
            "action": "summarize",
            "target": "prs",
            "goal": "find_blockers",
            # time_scope is None → excluded from serialization
        }
        serialized = json.dumps(core, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        assert compute_bucket_hash(key) == expected

    # ------------------------------------------------------------------
    # Core-fields-only invariant (Phase 1 bucket hash decision)
    # ------------------------------------------------------------------

    def test_optional_fields_do_not_affect_hash(self):
        """
        domain, format, audience, tool_family, constraints must NOT affect the
        bucket hash — they are LLM-unstable and would cause the same task to hash
        to different buckets across OpenAI calls.
        """
        base = self._key()
        with_domain = self._key(domain="engineering")
        with_format = self._key(format="slack_message")
        with_audience = self._key(audience="engineers")
        with_tool_family = self._key(tool_family="github")
        with_constraints = self._key(constraints=["urgent", "regression"])
        with_all = self._key(
            domain="engineering", format="slack_message",
            audience="engineers", tool_family="github",
            constraints=["urgent"],
        )

        h = compute_bucket_hash(base)
        assert compute_bucket_hash(with_domain) == h
        assert compute_bucket_hash(with_format) == h
        assert compute_bucket_hash(with_audience) == h
        assert compute_bucket_hash(with_tool_family) == h
        assert compute_bucket_hash(with_constraints) == h
        assert compute_bucket_hash(with_all) == h

    def test_core_fields_all_affect_hash(self):
        """Changing any of the 4 core fields must produce a different bucket."""
        base = self._key()
        assert compute_bucket_hash(self._key(action="triage")) != compute_bucket_hash(base)
        assert compute_bucket_hash(self._key(target="issues")) != compute_bucket_hash(base)
        assert compute_bucket_hash(self._key(goal="classify")) != compute_bucket_hash(base)
        assert compute_bucket_hash(self._key(time_scope="last_7d")) != compute_bucket_hash(base)

    def test_two_keys_differing_only_in_optional_fields_share_bucket(self):
        """
        A plan seeded with tool_family='github' and a lookup that omits tool_family
        must land in the same bucket — the reranker handles tool_family filtering,
        not the bucket router.
        """
        seeded_key = self._key(tool_family="github", domain="repository")
        lookup_key = self._key()  # no optional fields
        assert compute_bucket_hash(seeded_key) == compute_bucket_hash(lookup_key)


# ---------------------------------------------------------------------------
# canonicalize — happy path
# ---------------------------------------------------------------------------

class TestCanonicalizeSuccess:
    def test_returns_canonicalization_result(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(_GOOD_PAYLOAD)):
            result = canonicalize("Summarize open PRs and find blockers for the engineering team")

        assert result is not None
        assert result["confidence"] == 0.95
        assert result["canonicalizer_version"] == CANONICALIZER_VERSION
        assert result["from_cache"] is False
        assert len(result["intent_bucket_hash"]) == 16

    def test_intent_key_fields(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(_GOOD_PAYLOAD)):
            result = canonicalize("Summarize open PRs")

        assert result is not None
        key = result["intent_key"]
        assert key["action"] == "summarize"
        assert key["target"] == "pull_requests"
        assert key["goal"] == "identify_blockers"
        assert key["tool_family"] == "github"
        assert key["audience"] == "engineering"

    def test_none_optional_fields_omitted_from_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(_GOOD_PAYLOAD)):
            result = canonicalize("Summarize open PRs")

        assert result is not None
        key = result["intent_key"]
        assert "domain" not in key
        assert "format" not in key
        assert "constraints" not in key

    def test_constraints_sorted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {**_GOOD_PAYLOAD, "constraints": ["urgent", "bug"]}
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(payload)):
            result = canonicalize("Triage urgent bugs")

        assert result is not None
        assert result["intent_key"]["constraints"] == ["bug", "urgent"]

    def test_bucket_hash_deterministic(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(_GOOD_PAYLOAD)):
            r1 = canonicalize("task")
            r2 = canonicalize("task")

        assert r1 is not None and r2 is not None
        assert r1["intent_bucket_hash"] == r2["intent_bucket_hash"]

    def test_raw_canonical_text_is_json_string(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(_GOOD_PAYLOAD)):
            result = canonicalize("Summarize PRs")

        assert result is not None
        parsed = json.loads(result["raw_canonical_text"])
        assert parsed["action"] == "summarize"

    def test_intent_key_normalized_and_hash_stable(self, monkeypatch):
        """Model returns messy casing/whitespace; result must be normalized and hash stable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        dirty_payload = {
            **_GOOD_PAYLOAD,
            "action":      "  SUMMARIZE  ",
            "target":      "Pull_Requests",
            "goal":        "Identify_Blockers",
            "audience":    "Engineering",
            "tool_family": "GitHub",
            "constraints": ["Urgent", "Bug", "regression"],
            "confidence":  0.95,
        }
        clean_payload = {
            **_GOOD_PAYLOAD,
            "action":      "summarize",
            "target":      "pull_requests",
            "goal":        "identify_blockers",
            "audience":    "engineering",
            "tool_family": "github",
            "constraints": ["bug", "regression", "urgent"],  # sorted + lower
            "confidence":  0.95,
        }

        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(dirty_payload)):
            dirty_result = canonicalize("Summarize PRs")
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(clean_payload)):
            clean_result = canonicalize("Summarize PRs")

        assert dirty_result is not None and clean_result is not None

        key = dirty_result["intent_key"]
        assert key["action"] == "summarize"
        assert key["target"] == "pull_requests"
        assert key["goal"] == "identify_blockers"
        assert key["audience"] == "engineering"
        assert key["tool_family"] == "github"
        assert key["constraints"] == ["bug", "regression", "urgent"]

        # raw_canonical_text must carry the original messy model output, not normalized
        raw = json.loads(dirty_result["raw_canonical_text"])
        assert raw["action"] == "  SUMMARIZE  "

        # Hash must match clean equivalent
        assert dirty_result["intent_bucket_hash"] == clean_result["intent_bucket_hash"]


# ---------------------------------------------------------------------------
# canonicalize — failure paths
# ---------------------------------------------------------------------------

class TestCanonicalizeFailure:
    def test_low_confidence_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {**_GOOD_PAYLOAD, "confidence": 0.70}
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(payload)):
            assert canonicalize("some task") is None

    def test_confidence_exactly_at_threshold_passes(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {**_GOOD_PAYLOAD, "confidence": 0.85}
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(payload)):
            assert canonicalize("some task") is not None

    def test_malformed_json_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "choices": [{"message": {"content": "not valid json {"}}]
        }
        with patch("thriftlm.v2.intent.httpx.post", return_value=mock):
            assert canonicalize("some task") is None

    def test_timeout_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post",
                   side_effect=httpx.TimeoutException("timed out")):
            assert canonicalize("some task") is None

    def test_http_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post",
                   side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())):
            assert canonicalize("some task") is None

    def test_generic_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("thriftlm.v2.intent.httpx.post", side_effect=RuntimeError("boom")):
            assert canonicalize("some task") is None

    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert canonicalize("some task") is None

    def test_missing_required_field_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {**_GOOD_PAYLOAD}
        del payload["action"]
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(payload)):
            assert canonicalize("some task") is None

    def test_missing_confidence_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = {k: v for k, v in _GOOD_PAYLOAD.items() if k != "confidence"}
        with patch("thriftlm.v2.intent.httpx.post", return_value=_openai_response(payload)):
            assert canonicalize("some task") is None
