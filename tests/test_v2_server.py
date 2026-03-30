"""
Tests for thriftlm.v2._server — Phase 1 endpoints.

No real Redis, Supabase, OpenAI, or sentence-transformers.
All external dependencies are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from thriftlm.v2._server import (
    app,
    _build_structural_signature,
    _InMemoryCanonCache,
    _in_memory_canon_cache,
)
from thriftlm.v2.schemas import (
    CanonicalizationResult,
    FilledPlan,
    IntentKey,
    PlanTemplate,
    ValidationResult,
)
from thriftlm.v2.validator import VALIDATOR_VERSION

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_API_KEY = "tlm_test"
_BUCKET = "abc123def456abcd"

_INTENT_KEY: IntentKey = {
    "action": "summarize",
    "target": "pull_requests",
    "goal": "identify_blockers",
    "time_scope": None,
}

_PLAN: PlanTemplate = {
    "plan_id": "plan-1",
    "intent_key": _INTENT_KEY,
    "intent_bucket_hash": _BUCKET,
    "description": "fetch open pull requests and summarize blockers",
    "steps": [
        {"step_id": "1", "op": "fetch_prs", "inputs": {"repo": "org/repo"}, "outputs": {"prs": "list"}},
        {"step_id": "2", "op": "summarize", "inputs": {"prs": "{prs}"}, "outputs": {"summary": "str"}},
    ],
    "slots": [
        {"name": "repo", "source": "repo", "required": True,
         "type_hint": "str", "transform": None, "transform_args": None, "default": None},
    ],
    "output_schema": {"summary": "str"},
    "optional_outputs": [],
    "plan_version": "1",
    "canonicalizer_version": "v0.4",
    "extractor_version": "v0.1",
    "validator_version": VALIDATOR_VERSION,
    "created_at": "2026-01-01T00:00:00Z",
}

_FILLED: FilledPlan = {
    "plan_id": "plan-1",
    "intent_bucket_hash": _BUCKET,
    "filled_slots": {"repo": "org/repo"},
    "steps": [
        {"step_id": "1", "op": "fetch_prs", "inputs": {"repo": "org/repo"}, "outputs": {"prs": "list"}},
        {"step_id": "2", "op": "summarize", "inputs": {"prs": "{prs}"}, "outputs": {"summary": "str"}},
    ],
    "output_schema": {"summary": "str"},
}

_CANON: CanonicalizationResult = {
    "intent_key": _INTENT_KEY,
    "intent_bucket_hash": _BUCKET,
    "confidence": 0.95,
    "canonicalizer_version": "v0.4",
    "raw_canonical_text": "{}",
    "from_cache": False,
}

_VALIDATION_OK: ValidationResult = {
    "ok": True,
    "failed_stage": None,
    "reason": None,
    "validator_version": VALIDATOR_VERSION,
}

_VALIDATION_FAIL: ValidationResult = {
    "ok": False,
    "failed_stage": "1",
    "reason": "missing_required_slot:repo",
    "validator_version": VALIDATOR_VERSION,
}

_SCORED_CANDIDATE = {
    "plan": _PLAN,
    "semantic_similarity": 0.95,
    "structural_score": 0.90,
    "final_score": 0.935,
}

_CONTEXT = {"repo": "org/repo"}
_CAPS = {"tool_families": ["github"], "allow_side_effects": False}


def _lookup_body(**overrides):
    body = {
        "api_key": _API_KEY,
        "task": "summarize open PRs for org/repo",
        "context": _CONTEXT,
        "runtime_caps": _CAPS,
    }
    body.update(overrides)
    return body


def _store_body(**overrides):
    body = {"api_key": _API_KEY, "plan": dict(_PLAN)}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Helpers for patching the server's private helpers
# ---------------------------------------------------------------------------

def _patch_get_or_canon(return_value=None):
    """Patch _get_or_canonicalize to return a fixed CanonicalizationResult (or None)."""
    return patch("thriftlm.v2._server._get_or_canonicalize", return_value=return_value)


def _patch_sb(insert_id="inserted-plan-id"):
    mock = MagicMock()
    mock.table.return_value.insert.return_value.execute.return_value.data = [{"id": insert_id}]
    return mock


# ---------------------------------------------------------------------------
# POST /v2/plan/lookup
# ---------------------------------------------------------------------------

class TestLookupHitCanonCacheHit:
    """Canon cache returns a hit — no OpenAI call needed."""

    def test_returns_hit(self):
        with (
            _patch_get_or_canon({**_CANON, "from_cache": True}),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", return_value=_FILLED),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_OK),
        ):
            MockPC.return_value.get.return_value = [_SCORED_CANDIDATE]
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "hit"
        assert data["matched_plan_id"] == "plan-1"
        assert data["score"] == pytest.approx(0.935)

    def test_canonicalize_not_called_on_cache_hit(self):
        with (
            _patch_get_or_canon({**_CANON, "from_cache": True}),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", return_value=_FILLED),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_OK),
            patch("thriftlm.v2._server.canonicalize") as mock_canon,
        ):
            MockPC.return_value.get.return_value = [_SCORED_CANDIDATE]
            client.post("/v2/plan/lookup", json=_lookup_body())

        mock_canon.assert_not_called()


class TestLookupHitCanonCacheMiss:
    """Canon cache misses — server calls canonicalize() and caches the result."""

    def test_hit_after_canonicalize(self):
        with (
            _patch_get_or_canon(_CANON),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", return_value=_FILLED),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_OK),
        ):
            MockPC.return_value.get.return_value = [_SCORED_CANDIDATE]
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        assert resp.json()["status"] == "hit"

    def test_canon_result_stored_in_in_memory_cache(self):
        """After an OpenAI call, result is stored in _in_memory_canon_cache."""
        task = "unique-task-for-storage-test"
        _in_memory_canon_cache._store.pop(task, None)  # ensure clean state

        with (
            patch("thriftlm.v2._server.canonicalize", return_value=_CANON),
            patch("thriftlm.v2._server._make_redis_cache", return_value=None),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", return_value=_FILLED),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_OK),
        ):
            MockPC.return_value.get.return_value = [_SCORED_CANDIDATE]
            client.post("/v2/plan/lookup", json={**_lookup_body(), "task": task})

        assert _in_memory_canon_cache._store.get(task) is not None
        _in_memory_canon_cache._store.pop(task, None)  # cleanup


class TestLookupCanonicalizationFailed:
    def test_returns_miss_canonicalization_failed(self):
        with _patch_get_or_canon(None):
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "miss"
        assert data["reason"] == "canonicalization_failed"


class TestLookupSkipsBadCandidate:
    """Adapter exception on first candidate → server tries next."""

    def test_skips_to_second_candidate(self):
        from thriftlm.v2.adapter import SlotFillError

        good_plan = {**_PLAN, "plan_id": "plan-2"}
        good_candidate = {**_SCORED_CANDIDATE, "plan": good_plan, "final_score": 0.85}
        candidates = [_SCORED_CANDIDATE, good_candidate]
        filled_good = {**_FILLED, "plan_id": "plan-2"}

        with (
            _patch_get_or_canon(_CANON),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", side_effect=[SlotFillError("missing"), filled_good]),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_OK),
        ):
            MockPC.return_value.get.return_value = candidates
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "hit"
        assert data["matched_plan_id"] == "plan-2"

    def test_validation_fail_skips_to_next(self):
        good_plan = {**_PLAN, "plan_id": "plan-2"}
        good_candidate = {**_SCORED_CANDIDATE, "plan": good_plan, "final_score": 0.85}
        candidates = [_SCORED_CANDIDATE, good_candidate]
        filled_good = {**_FILLED, "plan_id": "plan-2"}

        with (
            _patch_get_or_canon(_CANON),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", side_effect=[_FILLED, filled_good]),
            patch("thriftlm.v2._server.validate_plan", side_effect=[_VALIDATION_FAIL, _VALIDATION_OK]),
        ):
            MockPC.return_value.get.return_value = candidates
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        assert resp.json()["matched_plan_id"] == "plan-2"


class TestLookupNoValidPlan:
    def test_all_candidates_fail_returns_miss(self):
        with (
            _patch_get_or_canon(_CANON),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
            patch("thriftlm.v2._server.adapt_plan", return_value=_FILLED),
            patch("thriftlm.v2._server.validate_plan", return_value=_VALIDATION_FAIL),
        ):
            MockPC.return_value.get.return_value = [_SCORED_CANDIDATE]
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "miss"
        assert data["reason"] == "no_valid_plan"
        assert data["canonicalization_result"]["intent_bucket_hash"] == _BUCKET

    def test_empty_candidates_returns_miss(self):
        with (
            _patch_get_or_canon(_CANON),
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.PlanCache") as MockPC,
        ):
            MockPC.return_value.get.return_value = []
            resp = client.post("/v2/plan/lookup", json=_lookup_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "miss"
        assert data["reason"] == "no_valid_plan"


class TestLookupValidation:
    def test_missing_api_key_returns_400(self):
        resp = client.post("/v2/plan/lookup", json={"task": "x", "context": {}, "runtime_caps": {}})
        # pydantic will reject missing required field
        assert resp.status_code == 422

    def test_missing_task_returns_400(self):
        with _patch_get_or_canon(None):
            resp = client.post("/v2/plan/lookup", json={"api_key": _API_KEY, "task": "", "context": {}, "runtime_caps": {}})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# _InMemoryCanonCache — unit tests
# ---------------------------------------------------------------------------

class TestInMemoryCanonCache:
    def setup_method(self):
        self.cache = _InMemoryCanonCache()

    def test_get_returns_none_on_empty(self):
        assert self.cache.get("anything") is None

    def test_set_then_get_returns_result(self):
        self.cache.set("task-a", _CANON)
        result = self.cache.get("task-a")
        assert result is not None
        assert result["intent_bucket_hash"] == _BUCKET

    def test_get_sets_from_cache_true(self):
        self.cache.set("task-a", {**_CANON, "from_cache": False})
        result = self.cache.get("task-a")
        assert result["from_cache"] is True

    def test_stored_result_not_mutated_by_get(self):
        """get() returns a copy with from_cache=True; stored original is unchanged."""
        self.cache.set("task-a", {**_CANON, "from_cache": False})
        self.cache.get("task-a")
        assert self.cache._store["task-a"]["from_cache"] is False

    def test_different_tasks_independent(self):
        self.cache.set("task-a", {**_CANON, "intent_bucket_hash": "aaa"})
        self.cache.set("task-b", {**_CANON, "intent_bucket_hash": "bbb"})
        assert self.cache.get("task-a")["intent_bucket_hash"] == "aaa"
        assert self.cache.get("task-b")["intent_bucket_hash"] == "bbb"

    def test_overwrite_replaces_entry(self):
        self.cache.set("task-a", {**_CANON, "intent_bucket_hash": "old"})
        self.cache.set("task-a", {**_CANON, "intent_bucket_hash": "new"})
        assert self.cache.get("task-a")["intent_bucket_hash"] == "new"


# ---------------------------------------------------------------------------
# _get_or_canonicalize — layered cache behavior
# ---------------------------------------------------------------------------

class TestGetOrCanonicalize:
    from thriftlm.v2._server import _get_or_canonicalize as _fn

    def setup_method(self):
        _in_memory_canon_cache._store.clear()

    def test_returns_none_when_canonicalize_fails(self):
        with (
            patch("thriftlm.v2._server._make_redis_cache", return_value=None),
            patch("thriftlm.v2._server.canonicalize", return_value=None),
        ):
            from thriftlm.v2._server import _get_or_canonicalize
            result = _get_or_canonicalize("some task")
        assert result is None

    def test_calls_canonicalize_on_full_miss(self):
        with (
            patch("thriftlm.v2._server._make_redis_cache", return_value=None),
            patch("thriftlm.v2._server.canonicalize", return_value=_CANON) as mock_canon,
        ):
            from thriftlm.v2._server import _get_or_canonicalize
            _get_or_canonicalize("fresh task")
        mock_canon.assert_called_once_with("fresh task")

    def test_stores_in_memory_after_openai_call(self):
        task = "store-after-openai"
        with (
            patch("thriftlm.v2._server._make_redis_cache", return_value=None),
            patch("thriftlm.v2._server.canonicalize", return_value=_CANON),
        ):
            from thriftlm.v2._server import _get_or_canonicalize
            _get_or_canonicalize(task)
        assert _in_memory_canon_cache._store.get(task) is not None

    def test_in_memory_hit_skips_canonicalize(self):
        task = "cached-task"
        _in_memory_canon_cache.set(task, _CANON)
        with (
            patch("thriftlm.v2._server._make_redis_cache", return_value=None),
            patch("thriftlm.v2._server.canonicalize") as mock_canon,
        ):
            from thriftlm.v2._server import _get_or_canonicalize
            result = _get_or_canonicalize(task)
        mock_canon.assert_not_called()
        assert result is not None

    def test_in_memory_hit_returns_from_cache_true(self):
        task = "cached-task-fc"
        _in_memory_canon_cache.set(task, {**_CANON, "from_cache": False})
        with patch("thriftlm.v2._server._make_redis_cache", return_value=None):
            from thriftlm.v2._server import _get_or_canonicalize
            result = _get_or_canonicalize(task)
        assert result["from_cache"] is True

    def test_redis_hit_stores_in_memory_and_skips_openai(self):
        task = "redis-hit-task"
        mock_redis = MagicMock()
        mock_redis.get.return_value = {**_CANON, "from_cache": True}

        with (
            patch("thriftlm.v2._server._make_redis_cache", return_value=mock_redis),
            patch("thriftlm.v2._server.canonicalize") as mock_canon,
        ):
            from thriftlm.v2._server import _get_or_canonicalize
            result = _get_or_canonicalize(task)

        mock_canon.assert_not_called()
        assert result is not None
        assert _in_memory_canon_cache._store.get(task) is not None


# ---------------------------------------------------------------------------
# POST /v2/plan/store
# ---------------------------------------------------------------------------

class TestStorePlanValidation:
    def test_missing_intent_key_returns_400(self):
        bad_plan = {k: v for k, v in _PLAN.items() if k != "intent_key"}
        resp = client.post("/v2/plan/store", json={"api_key": _API_KEY, "plan": bad_plan})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid_plan_template"

    def test_steps_not_list_returns_400(self):
        bad_plan = {**_PLAN, "steps": "not-a-list"}
        resp = client.post("/v2/plan/store", json={"api_key": _API_KEY, "plan": bad_plan})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid_plan_template"

    def test_output_schema_not_dict_returns_400(self):
        bad_plan = {**_PLAN, "output_schema": ["not", "a", "dict"]}
        resp = client.post("/v2/plan/store", json={"api_key": _API_KEY, "plan": bad_plan})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid_plan_template"

    def test_valid_shape_passes_validation(self):
        with (
            patch("thriftlm.v2._server._make_supabase_client", return_value=_patch_sb()),
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
            patch("thriftlm.v2._server.Embedder") as MockEmb,
        ):
            MockEmb.return_value.embed.return_value = [0.0] * 384
            resp = client.post("/v2/plan/store", json=_store_body())
        assert resp.status_code == 200


class TestStoreSuccess:
    def test_returns_stored(self):
        mock_sb = _patch_sb("new-plan-id")

        with (
            patch("thriftlm.v2._server._make_supabase_client", return_value=mock_sb),
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
            patch("thriftlm.v2._server.Embedder") as MockEmb,
        ):
            MockEmb.return_value.embed.return_value = [0.1] * 384
            resp = client.post("/v2/plan/store", json=_store_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["plan_id"] == "new-plan-id"
        assert data["intent_bucket_hash"] == _BUCKET

    def test_inserts_once(self):
        mock_sb = _patch_sb()

        with (
            patch("thriftlm.v2._server._make_supabase_client", return_value=mock_sb),
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
            patch("thriftlm.v2._server.Embedder") as MockEmb,
        ):
            MockEmb.return_value.embed.return_value = [0.0] * 384
            client.post("/v2/plan/store", json=_store_body())

        mock_sb.table.return_value.insert.assert_called_once()


class TestStoreHashMismatch:
    def test_returns_400_on_mismatch(self):
        wrong_hash_plan = {**_PLAN, "intent_bucket_hash": "wronghash0000000"}

        with (
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
        ):
            resp = client.post("/v2/plan/store", json={"api_key": _API_KEY, "plan": wrong_hash_plan})

        assert resp.status_code == 400
        assert resp.json()["detail"] == "hash_mismatch"

    def test_embedding_not_computed_on_hash_mismatch(self):
        wrong_hash_plan = {**_PLAN, "intent_bucket_hash": "wronghash0000000"}

        with (
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
            patch("thriftlm.v2._server.Embedder") as MockEmb,
        ):
            client.post("/v2/plan/store", json={"api_key": _API_KEY, "plan": wrong_hash_plan})

        MockEmb.assert_not_called()


class TestStoreStructuralSignature:
    def test_sig_derived_server_side(self):
        """Structural signature written to DB must be server-computed, not caller-supplied."""
        mock_sb = _patch_sb()
        captured = {}

        def capture_insert(data):
            captured["sig"] = data["structural_signature"]
            return mock_sb.table.return_value.insert.return_value

        mock_sb.table.return_value.insert.side_effect = capture_insert

        with (
            patch("thriftlm.v2._server._make_supabase_client", return_value=mock_sb),
            patch("thriftlm.v2._server.compute_bucket_hash", return_value=_BUCKET),
            patch("thriftlm.v2._server.Embedder") as MockEmb,
        ):
            MockEmb.return_value.embed.return_value = [0.0] * 384
            client.post("/v2/plan/store", json=_store_body())

        assert "required_context_keys" in captured["sig"]
        assert "tool_families" in captured["sig"]
        assert "has_side_effects" in captured["sig"]
        assert "step_count" in captured["sig"]

    def test_required_slots_in_sig(self):
        sig = _build_structural_signature(_PLAN)
        assert "repo" in sig["required_context_keys"]

    def test_required_context_keys_sorted_and_deduped(self):
        plan_dupes = {
            **_PLAN,
            "slots": [
                {"name": "repo", "source": "repo", "required": True,
                 "type_hint": "str", "transform": None, "transform_args": None, "default": None},
                {"name": "repo2", "source": "repo", "required": True,  # duplicate source
                 "type_hint": "str", "transform": None, "transform_args": None, "default": None},
                {"name": "aaa", "source": "aaa", "required": True,
                 "type_hint": "str", "transform": None, "transform_args": None, "default": None},
            ],
        }
        sig = _build_structural_signature(plan_dupes)
        assert sig["required_context_keys"] == sorted(set(sig["required_context_keys"]))
        assert sig["required_context_keys"].count("repo") == 1

    def test_optional_slot_not_in_sig(self):
        plan_with_opt = {
            **_PLAN,
            "slots": [
                {"name": "repo", "source": "repo", "required": True,
                 "type_hint": "str", "transform": None, "transform_args": None, "default": None},
                {"name": "scope", "source": "scope", "required": False,
                 "type_hint": "str", "transform": None, "transform_args": None, "default": "7d"},
            ],
        }
        sig = _build_structural_signature(plan_with_opt)
        assert "repo" in sig["required_context_keys"]
        assert "scope" not in sig["required_context_keys"]

    def test_tool_families_sorted_unique(self):
        plan_with_tools = {
            **_PLAN,
            "steps": [
                {"step_id": "1", "op": "x", "inputs": {}, "outputs": {}, "tool_family": "github"},
                {"step_id": "2", "op": "y", "inputs": {}, "outputs": {}, "tool_family": "github"},
                {"step_id": "3", "op": "z", "inputs": {}, "outputs": {}, "tool_family": "jira"},
            ],
        }
        sig = _build_structural_signature(plan_with_tools)
        assert sig["tool_families"] == ["github", "jira"]

    def test_side_effect_detected(self):
        plan_with_fx = {
            **_PLAN,
            "steps": [
                {"step_id": "1", "op": "post", "inputs": {}, "outputs": {}, "side_effect": True},
            ],
        }
        sig = _build_structural_signature(plan_with_fx)
        assert sig["has_side_effects"] is True

    def test_no_side_effect(self):
        sig = _build_structural_signature(_PLAN)
        assert sig["has_side_effects"] is False

    def test_step_count(self):
        sig = _build_structural_signature(_PLAN)
        assert sig["step_count"] == 2


# ---------------------------------------------------------------------------
# GET /v2/metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_returns_ok(self):
        resp = client.get("/v2/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["phase"] == "phase_1"
        assert data["version"] == VALIDATOR_VERSION
