"""
Tests for thriftlm.v2.plan_cache.PlanCache.

No real Supabase, Redis, or network calls. All external dependencies mocked.
"""
from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from thriftlm.v2.plan_cache import (
    PlanCache,
    _cosine_similarity,
    _format_audience_score,
    _parse_embedding,
    _side_effect_compat,
    _slot_overlap_score,
    _tool_family_match_score,
)
from thriftlm.v2.schemas import PlanTemplate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUCKET = "abc123def456abcd"
_API_KEY = "tlm_test"

_TEMPLATE: PlanTemplate = {
    "plan_id":               "plan-1",
    "intent_key":            {"action": "summarize", "target": "prs", "goal": "find_blockers", "time_scope": None},
    "intent_bucket_hash":    _BUCKET,
    "description":           "fetch open pull requests and summarize blockers",
    "steps":                 [],
    "slots":                 [{"name": "repo", "source": "repo", "required": True,
                               "type_hint": "str", "transform": None, "transform_args": None, "default": None}],
    "output_schema":         {"summary": "string"},
    "optional_outputs":      [],
    "plan_version":          "1",
    "canonicalizer_version": "v0.4",
    "extractor_version":     "v0.1",
    "validator_version":     "v0.1",
    "created_at":            "2026-01-01T00:00:00Z",
}

_SIG = {
    "required_context_keys": ["repo"],
    "tool_families":         ["github"],
    "has_side_effects":      False,
    "format":                None,
    "audience":              None,
    "step_count":            2,
}

_CAPS = {"tool_families": ["github"], "allow_side_effects": False}
_CONTEXT = {"repo": "org/repo"}

# A unit vector pointing in one dimension — easy to reason about similarity.
_VEC_A = [1.0] + [0.0] * 383
_VEC_B = [0.0, 1.0] + [0.0] * 382  # orthogonal to A, similarity = 0


def _make_row(
    vec=None,
    template=None,
    sig=None,
    as_json_strings: bool = False,
) -> dict:
    vec = vec if vec is not None else _VEC_A
    template = template if template is not None else _TEMPLATE
    sig = sig if sig is not None else _SIG
    return {
        "id":                    "row-1",
        "description":           template["description"],
        "embedding":             json.dumps(vec) if as_json_strings else vec,
        "template_json":         json.dumps(template) if as_json_strings else template,
        "structural_signature":  json.dumps(sig) if as_json_strings else sig,
        "intent_bucket_hash":    _BUCKET,
        "plan_version":          "1",
        "canonicalizer_version": "v0.4",
        "extractor_version":     "v0.1",
        "validator_version":     "v0.1",
        "created_at":            "2026-01-01T00:00:00Z",
    }


def _make_cache(rows, threshold=0.0, top_k=5):
    """Return a PlanCache whose Supabase client returns *rows* and embedder is mocked."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value \
        .eq.return_value.eq.return_value.execute.return_value.data = rows

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = _VEC_A  # task vector == _VEC_A

    return PlanCache(
        supabase_client=mock_sb,
        api_key=_API_KEY,
        plan_threshold=threshold,
        top_k=top_k,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(_cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert _cosine_similarity(v1, v2) < 0

    def test_zero_vector_no_crash(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 0.0]
        result = _cosine_similarity(v1, v2)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# _parse_embedding
# ---------------------------------------------------------------------------

class TestParseEmbedding:
    def test_list_passthrough(self):
        vec = [1.0, 2.0, 3.0]
        assert _parse_embedding(vec) == [1.0, 2.0, 3.0]

    def test_json_string_parsed(self):
        vec = [0.1, 0.2, 0.3]
        assert _parse_embedding(json.dumps(vec)) == pytest.approx(vec)

    def test_invalid_json_string_returns_none(self):
        assert _parse_embedding("not json {") is None

    def test_non_numeric_list_returns_none(self):
        assert _parse_embedding(["a", "b"]) is None

    def test_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_empty_list_returns_none(self):
        assert _parse_embedding([]) is None


# ---------------------------------------------------------------------------
# Subscore helpers
# ---------------------------------------------------------------------------

class TestSlotOverlapScore:
    def test_empty_required_keys_returns_1(self):
        assert _slot_overlap_score({"required_context_keys": []}, {}) == 1.0

    def test_missing_required_keys_key_returns_1(self):
        assert _slot_overlap_score({}, {"repo": "x"}) == 1.0

    def test_full_overlap(self):
        sig = {"required_context_keys": ["repo", "time_scope"]}
        ctx = {"repo": "x", "time_scope": "7d", "extra": "y"}
        assert _slot_overlap_score(sig, ctx) == 1.0

    def test_partial_overlap(self):
        sig = {"required_context_keys": ["repo", "time_scope"]}
        ctx = {"repo": "x"}
        assert abs(_slot_overlap_score(sig, ctx) - 0.5) < 1e-9

    def test_no_overlap(self):
        sig = {"required_context_keys": ["repo"]}
        assert _slot_overlap_score(sig, {}) == 0.0


class TestToolFamilyMatchScore:
    def test_empty_plan_families_returns_1(self):
        assert _tool_family_match_score({"tool_families": []}, {"tool_families": ["github"]}) == 1.0

    def test_overlap_returns_1(self):
        sig = {"tool_families": ["github", "jira"]}
        assert _tool_family_match_score(sig, {"tool_families": ["github"]}) == 1.0

    def test_no_overlap_returns_0(self):
        sig = {"tool_families": ["jira"]}
        assert _tool_family_match_score(sig, {"tool_families": ["github"]}) == 0.0

    def test_missing_runtime_families_returns_0(self):
        sig = {"tool_families": ["github"]}
        assert _tool_family_match_score(sig, {}) == 0.0


class TestFormatAudienceScore:
    def test_both_null_returns_half(self):
        sig = {"format": None, "audience": None}
        assert _format_audience_score(sig, {}) == pytest.approx(0.5)

    def test_match_returns_1(self):
        sig = {"format": "slack_message", "audience": "engineering"}
        caps = {"format": "slack_message", "audience": "engineering"}
        assert _format_audience_score(sig, caps) == pytest.approx(1.0)

    def test_mismatch_returns_0(self):
        sig = {"format": "slack_message", "audience": "engineering"}
        caps = {"format": "email", "audience": "management"}
        assert _format_audience_score(sig, caps) == pytest.approx(0.0)

    def test_plan_null_runtime_present_returns_half(self):
        sig = {"format": None, "audience": None}
        caps = {"format": "slack_message", "audience": "engineering"}
        assert _format_audience_score(sig, caps) == pytest.approx(0.5)

    def test_plan_set_runtime_missing_returns_half(self):
        sig = {"format": "slack_message", "audience": "engineering"}
        assert _format_audience_score(sig, {}) == pytest.approx(0.5)


class TestSideEffectCompat:
    def test_no_side_effects_returns_1(self):
        assert _side_effect_compat({"has_side_effects": False}, {}) == 1.0

    def test_side_effects_allowed_returns_1(self):
        sig = {"has_side_effects": True}
        assert _side_effect_compat(sig, {"allow_side_effects": True}) == 1.0

    def test_side_effects_not_allowed_returns_0(self):
        sig = {"has_side_effects": True}
        assert _side_effect_compat(sig, {"allow_side_effects": False}) == 0.0

    def test_side_effects_default_not_allowed(self):
        sig = {"has_side_effects": True}
        assert _side_effect_compat(sig, {}) == 0.0


# ---------------------------------------------------------------------------
# PlanCache.get
# ---------------------------------------------------------------------------

class TestPlanCacheGet:
    def test_empty_bucket_returns_empty(self):
        cache = _make_cache(rows=[])
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert result == []

    def test_supabase_exception_returns_empty(self):
        mock_sb = MagicMock()
        mock_sb.table.side_effect = RuntimeError("DB down")
        cache = PlanCache(mock_sb, _API_KEY, embedder=MagicMock())
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_malformed_embedding_row_skipped(self):
        row = _make_row()
        row["embedding"] = "not a vector"
        cache = _make_cache(rows=[row])
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_malformed_template_json_row_skipped(self):
        row = _make_row()
        row["template_json"] = "{broken"
        cache = _make_cache(rows=[row])
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_malformed_structural_signature_row_skipped(self):
        row = _make_row()
        row["structural_signature"] = "{broken"
        cache = _make_cache(rows=[row])
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_returns_scored_plan(self):
        row = _make_row()  # stored vec == task vec == _VEC_A → semantic = 1.0
        cache = _make_cache(rows=[row], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert len(result) == 1
        sp = result[0]
        assert abs(sp["semantic_similarity"] - 1.0) < 1e-6
        assert 0.0 <= sp["structural_score"] <= 1.0
        assert sp["final_score"] == pytest.approx(0.7 * sp["semantic_similarity"] + 0.3 * sp["structural_score"])

    def test_threshold_filters_low_scores(self):
        # store a vec orthogonal to task vec → semantic = 0, final ≈ 0.3*struct
        row = _make_row(vec=_VEC_B)
        cache = _make_cache(rows=[row], threshold=0.99)
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_top_k_respected(self):
        rows = [_make_row() for _ in range(10)]
        cache = _make_cache(rows=rows, threshold=0.0, top_k=3)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert len(result) <= 3

    def test_results_sorted_descending(self):
        # Two rows: one with identical vec (sem=1.0) and one orthogonal (sem=0.0)
        row_high = _make_row(vec=_VEC_A)
        row_low = _make_row(vec=_VEC_B)
        cache = _make_cache(rows=[row_low, row_high], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert result[0]["final_score"] >= result[-1]["final_score"]

    def test_embedding_as_json_string_parsed(self):
        row = _make_row(as_json_strings=True)
        cache = _make_cache(rows=[row], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert len(result) == 1
        assert abs(result[0]["semantic_similarity"] - 1.0) < 1e-6

    def test_template_json_as_string_parsed(self):
        row = _make_row(as_json_strings=True)
        cache = _make_cache(rows=[row], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        assert result[0]["plan"]["plan_id"] == "plan-1"

    def test_good_rows_survive_bad_rows(self):
        bad_row = _make_row()
        bad_row["embedding"] = "bad"
        good_row = _make_row()
        cache = _make_cache(rows=[bad_row, good_row], threshold=0.0)
        assert len(cache.get(_BUCKET, "task", _CONTEXT, _CAPS)) == 1

    def test_slot_overlap_empty_required_keys_scores_1(self):
        sig = {**_SIG, "required_context_keys": []}
        row = _make_row(sig=sig)
        cache = _make_cache(rows=[row], threshold=0.0)
        result = cache.get(_BUCKET, "task", {}, _CAPS)  # empty context
        assert len(result) == 1
        # slot_overlap = 1.0 since required_context_keys is empty

    def test_tool_family_mismatch_reduces_score(self):
        sig = {**_SIG, "tool_families": ["jira"]}
        row_mismatch = _make_row(sig=sig)
        row_match = _make_row()  # tool_families = ["github"]
        cache = _make_cache(rows=[row_mismatch, row_match], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, _CAPS)
        scores = {r["plan"]["plan_id"]: r["structural_score"] for r in result}
        # Both have same plan_id "plan-1", compare via structural_score difference
        assert result[0]["structural_score"] >= result[-1]["structural_score"]

    def test_sig_required_context_keys_as_string_skips_row(self):
        """required_context_keys='repo' (string, not list) must cause row to be skipped."""
        sig = {**_SIG, "required_context_keys": "repo"}
        row = _make_row(sig=sig)
        cache = _make_cache(rows=[row], threshold=0.0)
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_sig_tool_families_as_string_skips_row(self):
        """tool_families='github' (string, not list) must cause row to be skipped."""
        sig = {**_SIG, "tool_families": "github"}
        row = _make_row(sig=sig)
        cache = _make_cache(rows=[row], threshold=0.0)
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_sig_has_side_effects_wrong_type_skips_row(self):
        """has_side_effects='false' (string, not bool) must cause row to be skipped."""
        sig = {**_SIG, "has_side_effects": "false"}
        row = _make_row(sig=sig)
        cache = _make_cache(rows=[row], threshold=0.0)
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_embedding_length_mismatch_skips_row(self):
        """Stored embedding with wrong dimensionality must be skipped."""
        short_vec = [1.0, 0.0, 0.0]   # task vec is 384-dim; this is 3-dim
        row = _make_row(vec=short_vec)
        cache = _make_cache(rows=[row], threshold=0.0)
        assert cache.get(_BUCKET, "task", _CONTEXT, _CAPS) == []

    def test_side_effects_blocked_when_not_allowed(self):
        sig = {**_SIG, "has_side_effects": True}
        row = _make_row(sig=sig)
        caps = {**_CAPS, "allow_side_effects": False}
        cache = _make_cache(rows=[row], threshold=0.0)
        result = cache.get(_BUCKET, "task", _CONTEXT, caps)
        assert len(result) == 1
        # side_effect_compat = 0.0 → structural_score is lower
        assert result[0]["structural_score"] < 1.0
