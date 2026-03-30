"""
Tests for thriftlm.v2.adapters — BasePlanCache and ThriftLMPlanCache.

No real HTTP.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from thriftlm.v2.adapters import BasePlanCache, ThriftLMPlanCache

_BASE_URL = "http://localhost:8000"
_API_KEY = "tlm_test"

_HIT_RESPONSE = {
    "status": "hit",
    "matched_plan_id": "plan-1",
    "score": 0.93,
    "filled_plan": {},
    "canonicalization_result": {},
}

_STORE_RESPONSE = {
    "status": "stored",
    "plan_id": "plan-1",
    "intent_bucket_hash": "abc123def456abcd",
}

_PLAN = {"plan_id": "plan-1", "intent_bucket_hash": "abc123def456abcd",
         "description": "test", "intent_key": {}, "steps": [], "slots": [],
         "output_schema": {}, "optional_outputs": []}


def _mock_response(status_code: int, json_body: dict) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_body
    mock.text = str(json_body)
    return mock


# ---------------------------------------------------------------------------
# BasePlanCache
# ---------------------------------------------------------------------------

class TestBasePlanCache:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BasePlanCache()  # type: ignore[abstract]

    def test_concrete_subclass_requires_both_methods(self):
        class Partial(BasePlanCache):
            def lookup(self, task, context, runtime_caps):
                return {}
        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        class Full(BasePlanCache):
            def lookup(self, task, context, runtime_caps):
                return {}
            def store(self, plan):
                return {}
        obj = Full()
        assert isinstance(obj, BasePlanCache)


# ---------------------------------------------------------------------------
# ThriftLMPlanCache — lookup
# ---------------------------------------------------------------------------

class TestThriftLMPlanCacheLookup:
    def test_sends_correct_payload(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)
        mock_resp = _mock_response(200, _HIT_RESPONSE)

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            cache.lookup(
                task="summarize PRs",
                context={"repo": "org/repo"},
                runtime_caps={"tool_families": ["github"]},
            )

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["api_key"] == _API_KEY
        assert payload["task"] == "summarize PRs"
        assert payload["context"] == {"repo": "org/repo"}
        assert payload["runtime_caps"] == {"tool_families": ["github"]}

    def test_posts_to_correct_url(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)
        mock_resp = _mock_response(200, _HIT_RESPONSE)

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            cache.lookup("task", {}, {})

        url = mock_post.call_args[0][0]
        assert url == f"{_BASE_URL}/v2/plan/lookup"

    def test_returns_json_dict(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(200, _HIT_RESPONSE)):
            result = cache.lookup("task", {}, {})

        assert result == _HIT_RESPONSE

    def test_non_200_raises_runtime_error(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(400, {"detail": "bad_request"})):
            with pytest.raises(RuntimeError, match="400"):
                cache.lookup("task", {}, {})

    def test_500_raises_runtime_error(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(500, {"detail": "server_error"})):
            with pytest.raises(RuntimeError, match="500"):
                cache.lookup("task", {}, {})

    def test_timeout_raises_runtime_error(self):
        import httpx as _httpx
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", side_effect=_httpx.TimeoutException("timed out")):
            with pytest.raises(RuntimeError, match="timed out"):
                cache.lookup("task", {}, {})

    def test_uses_configured_timeout(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL, timeout=2.5)

        with patch("httpx.post", return_value=_mock_response(200, _HIT_RESPONSE)) as mock_post:
            cache.lookup("task", {}, {})

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 2.5

    def test_trailing_slash_stripped_from_base_url(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL + "/")

        with patch("httpx.post", return_value=_mock_response(200, _HIT_RESPONSE)) as mock_post:
            cache.lookup("task", {}, {})

        url = mock_post.call_args[0][0]
        assert not url.startswith(f"{_BASE_URL}//")


# ---------------------------------------------------------------------------
# ThriftLMPlanCache — store
# ---------------------------------------------------------------------------

class TestThriftLMPlanCacheStore:
    def test_sends_correct_payload(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(200, _STORE_RESPONSE)) as mock_post:
            cache.store(_PLAN)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["api_key"] == _API_KEY
        assert payload["plan"] == _PLAN

    def test_posts_to_correct_url(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(200, _STORE_RESPONSE)) as mock_post:
            cache.store(_PLAN)

        url = mock_post.call_args[0][0]
        assert url == f"{_BASE_URL}/v2/plan/store"

    def test_returns_json_dict(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(200, _STORE_RESPONSE)):
            result = cache.store(_PLAN)

        assert result == _STORE_RESPONSE

    def test_non_200_raises_runtime_error(self):
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", return_value=_mock_response(400, {"detail": "hash_mismatch"})):
            with pytest.raises(RuntimeError, match="400"):
                cache.store(_PLAN)

    def test_timeout_raises_runtime_error(self):
        import httpx as _httpx
        cache = ThriftLMPlanCache(api_key=_API_KEY, base_url=_BASE_URL)

        with patch("httpx.post", side_effect=_httpx.TimeoutException("timed out")):
            with pytest.raises(RuntimeError, match="timed out"):
                cache.store(_PLAN)
