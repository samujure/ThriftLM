"""
Tests for thriftlm.v2.adapter — adapt_plan, TransformRegistry, built-in transforms.

No network, no Supabase, no Redis, no OpenAI.
"""
from __future__ import annotations

from typing import Any

import pytest

from thriftlm.v2.adapter import (
    SlotFillError,
    SlotTypeError,
    TransformExecutionError,
    TransformNotFoundError,
    TransformRegistry,
    adapt_plan,
    _filter_open,
    _sort_by_date_desc,
    _top_n,
    _strip_html,
    _group_by_status,
    _truncate,
    _to_slack_bullets,
    _matches_type_hint,
)
from thriftlm.v2.schemas import PlanTemplate, SlotSpec, PlanStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slot(
    name: str,
    source: str,
    type_hint: str = "str",
    required: bool = True,
    transform: str | None = None,
    transform_args: dict | None = None,
    default: Any = None,
) -> SlotSpec:
    return SlotSpec(
        name=name,
        source=source,
        type_hint=type_hint,
        required=required,
        transform=transform,
        transform_args=transform_args,
        default=default,
    )


def _step(step_id: str, inputs: dict) -> PlanStep:
    return PlanStep(step_id=step_id, op="noop", inputs=inputs, outputs={})


def _plan(slots: list[SlotSpec], steps: list[PlanStep] | None = None) -> PlanTemplate:
    return PlanTemplate(
        plan_id="test-plan",
        intent_key={"action": "test", "target": "x", "goal": "y", "time_scope": None},
        intent_bucket_hash="abc123",
        description="test plan",
        steps=steps or [],
        slots=slots,
        output_schema={},
        optional_outputs=[],
        plan_version="1",
        canonicalizer_version="v0.4",
        extractor_version="v0.1",
        validator_version="v0.1",
        created_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# _matches_type_hint
# ---------------------------------------------------------------------------

class TestMatchesTypeHint:
    def test_str(self):
        assert _matches_type_hint("hello", "str")
        assert not _matches_type_hint(1, "str")

    def test_int(self):
        assert _matches_type_hint(42, "int")
        assert not _matches_type_hint(True, "int")   # bool excluded
        assert not _matches_type_hint(3.14, "int")

    def test_float(self):
        assert _matches_type_hint(3.14, "float")
        assert _matches_type_hint(1, "float")         # int is float-compatible
        assert not _matches_type_hint(True, "float")  # bool excluded

    def test_bool(self):
        assert _matches_type_hint(True, "bool")
        assert not _matches_type_hint(1, "bool")

    def test_dict(self):
        assert _matches_type_hint({}, "dict")
        assert not _matches_type_hint([], "dict")

    def test_list(self):
        assert _matches_type_hint([], "list")
        assert not _matches_type_hint({}, "list")

    def test_list_str(self):
        assert _matches_type_hint(["a", "b"], "list[str]")
        assert not _matches_type_hint(["a", 1], "list[str]")

    def test_list_dict(self):
        assert _matches_type_hint([{"k": "v"}], "list[dict]")
        assert not _matches_type_hint(["a"], "list[dict]")

    def test_dict_str_list_dict(self):
        assert _matches_type_hint({"open": [{"id": 1}]}, "dict[str, list[dict]]")
        assert not _matches_type_hint({"open": "bad"}, "dict[str, list[dict]]")

    def test_unknown_hint_passes(self):
        assert _matches_type_hint("anything", "list[pr]")


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------

class TestBuiltinTransforms:
    def test_filter_open(self):
        items = [{"status": "open", "id": 1}, {"status": "closed", "id": 2}]
        result = _filter_open(items, {})
        assert result == [{"status": "open", "id": 1}]

    def test_sort_by_date_desc(self):
        items = [{"date": "2026-01-01"}, {"date": "2026-03-01"}, {"date": "2026-02-01"}]
        result = _sort_by_date_desc(items, {"field": "date"})
        assert [i["date"] for i in result] == ["2026-03-01", "2026-02-01", "2026-01-01"]

    def test_top_n(self):
        assert _top_n([1, 2, 3, 4, 5], {"n": 3}) == [1, 2, 3]
        assert _top_n([1, 2], {"n": 10}) == [1, 2]

    def test_strip_html(self):
        assert _strip_html("<b>hello</b> <i>world</i>", {}) == "hello world"
        assert _strip_html("no tags", {}) == "no tags"

    def test_group_by_status(self):
        items = [{"state": "open"}, {"state": "closed"}, {"state": "open"}]
        result = _group_by_status(items, {"field": "state"})
        assert len(result["open"]) == 2
        assert len(result["closed"]) == 1

    def test_group_by_status_missing_field_uses_unknown(self):
        items = [{"state": "open"}, {"other": "x"}, {"state": None}]
        result = _group_by_status(items, {"field": "state"})
        assert len(result["open"]) == 1
        assert len(result["unknown"]) == 2  # missing key and None both → "unknown"
        assert None not in result

    def test_truncate(self):
        assert _truncate("hello world", {"max_chars": 5}) == "hello"
        assert _truncate("hi", {"max_chars": 100}) == "hi"

    def test_to_slack_bullets_list_str(self):
        result = _to_slack_bullets(["fix login", "add tests"], {})
        assert result == "- fix login\n- add tests"

    def test_to_slack_bullets_custom_prefix(self):
        result = _to_slack_bullets(["a", "b"], {"prefix": "*"})
        assert result == "* a\n* b"

    def test_to_slack_bullets_list_dict_uses_str(self):
        items = [{"id": 1, "title": "fix bug"}]
        result = _to_slack_bullets(items, {})
        assert result == f"- {str(items[0])}"

    def test_to_slack_bullets_empty(self):
        assert _to_slack_bullets([], {}) == ""


# ---------------------------------------------------------------------------
# TransformRegistry
# ---------------------------------------------------------------------------

class TestTransformRegistry:
    def test_register_and_apply(self):
        reg = TransformRegistry()
        reg.register("double", lambda v, _: v * 2)
        assert reg.apply("double", 5, {}) == 10

    def test_unknown_transform_raises(self):
        reg = TransformRegistry()
        with pytest.raises(TransformNotFoundError):
            reg.apply("nonexistent", "x", {})

    def test_runtime_error_wrapped_as_execution_error(self):
        reg = TransformRegistry()
        reg.register("exploder", lambda v, _: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(TransformExecutionError, match="exploder.*boom"):
            reg.apply("exploder", "value", {})

    def test_get_unknown_raises(self):
        reg = TransformRegistry()
        with pytest.raises(TransformNotFoundError):
            reg.get("nonexistent")


# ---------------------------------------------------------------------------
# adapt_plan — slot resolution
# ---------------------------------------------------------------------------

class TestAdaptPlanSlots:
    def test_required_slot_resolved(self):
        plan = _plan([_slot("repo", "repo", "str")])
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        assert result["filled_slots"]["repo"] == "org/myrepo"

    def test_required_slot_missing_raises(self):
        plan = _plan([_slot("repo", "repo", "str", required=True)])
        with pytest.raises(SlotFillError):
            adapt_plan(plan, {})

    def test_optional_slot_missing_uses_default(self):
        plan = _plan([_slot("time_scope", "time_scope", "str", required=False, default="last_7d")])
        result = adapt_plan(plan, {})
        assert result["filled_slots"]["time_scope"] == "last_7d"

    def test_optional_slot_present_overrides_default(self):
        plan = _plan([_slot("time_scope", "time_scope", "str", required=False, default="last_7d")])
        result = adapt_plan(plan, {"time_scope": "last_30d"})
        assert result["filled_slots"]["time_scope"] == "last_30d"

    def test_wrong_type_raises_slot_type_error(self):
        plan = _plan([_slot("count", "count", "int")])
        with pytest.raises(SlotTypeError):
            adapt_plan(plan, {"count": "not-an-int"})

    def test_multiple_slots_all_resolved(self):
        plan = _plan([
            _slot("repo", "repo", "str"),
            _slot("time_scope", "time_scope", "str", required=False, default="last_7d"),
        ])
        result = adapt_plan(plan, {"repo": "org/repo"})
        assert result["filled_slots"]["repo"] == "org/repo"
        assert result["filled_slots"]["time_scope"] == "last_7d"


# ---------------------------------------------------------------------------
# adapt_plan — transforms
# ---------------------------------------------------------------------------

class TestAdaptPlanTransforms:
    def test_transform_applied(self):
        plan = _plan([_slot("items", "items", "list", transform="top_n", transform_args={"n": 2})])
        result = adapt_plan(plan, {"items": [1, 2, 3, 4, 5]})
        assert result["filled_slots"]["items"] == [1, 2]

    def test_unknown_transform_raises(self):
        plan = _plan([_slot("x", "x", "str", transform="does_not_exist")])
        with pytest.raises(TransformNotFoundError):
            adapt_plan(plan, {"x": "value"})

    def test_wrong_type_after_transform_raises(self):
        # top_n returns a list; type_hint is "str" → should raise SlotTypeError
        plan = _plan([_slot("items", "items", "str", transform="top_n", transform_args={"n": 2})])
        with pytest.raises(SlotTypeError):
            adapt_plan(plan, {"items": [1, 2, 3]})

    def test_filter_open_transform(self):
        items = [{"status": "open"}, {"status": "closed"}]
        plan = _plan([_slot("prs", "prs", "list", transform="filter_open")])
        result = adapt_plan(plan, {"prs": items})
        assert result["filled_slots"]["prs"] == [{"status": "open"}]

    def test_strip_html_transform(self):
        plan = _plan([_slot("body", "body", "str", transform="strip_html")])
        result = adapt_plan(plan, {"body": "<p>hello</p>"})
        assert result["filled_slots"]["body"] == "hello"

    def test_custom_registry_used(self):
        reg = TransformRegistry()
        reg.register("shout", lambda v, _: v.upper())
        plan = _plan([_slot("msg", "msg", "str", transform="shout")])
        result = adapt_plan(plan, {"msg": "hello"}, registry=reg)
        assert result["filled_slots"]["msg"] == "HELLO"


# ---------------------------------------------------------------------------
# adapt_plan — placeholder substitution
# ---------------------------------------------------------------------------

class TestAdaptPlanSubstitution:
    def test_exact_placeholder_replaced(self):
        steps = [_step("1", {"repo": "{repo}"})]
        plan = _plan([_slot("repo", "repo", "str")], steps=steps)
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        assert result["steps"][0]["inputs"]["repo"] == "org/myrepo"

    def test_partial_string_not_substituted(self):
        steps = [_step("1", {"url": "https://github.com/{repo}"})]
        plan = _plan([_slot("repo", "repo", "str")], steps=steps)
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        assert result["steps"][0]["inputs"]["url"] == "https://github.com/{repo}"

    def test_non_slot_reference_untouched(self):
        # "{prs}" appears in step but is NOT a slot — leave as-is
        steps = [_step("1", {"data": "{prs}"})]
        plan = _plan([_slot("repo", "repo", "str")], steps=steps)
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        assert result["steps"][0]["inputs"]["data"] == "{prs}"

    def test_prior_step_reference_untouched(self):
        # step references output of a prior step, not a slot
        steps = [
            _step("1", {"repo": "{repo}"}),
            _step("2", {"prs": "{prs}"}),   # {prs} is a step output, not a slot
        ]
        plan = _plan([_slot("repo", "repo", "str")], steps=steps)
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        assert result["steps"][1]["inputs"]["prs"] == "{prs}"

    def test_non_string_value_substituted(self):
        # slot holds a list; placeholder in step gets the list object
        steps = [_step("1", {"items": "{items}"})]
        plan = _plan([_slot("items", "items", "list")], steps=steps)
        result = adapt_plan(plan, {"items": [1, 2, 3]})
        assert result["steps"][0]["inputs"]["items"] == [1, 2, 3]

    def test_nested_placeholder_in_dict_input_untouched(self):
        """Nested placeholder inside a dict value is intentionally not substituted."""
        steps = [_step("1", {"config": {"key": "{repo}"}})]
        plan = _plan([_slot("repo", "repo", "str")], steps=steps)
        result = adapt_plan(plan, {"repo": "org/myrepo"})
        # The outer value is a dict, not a bare "{repo}" string — left unchanged
        assert result["steps"][0]["inputs"]["config"] == {"key": "{repo}"}

    def test_no_steps_returns_empty_steps(self):
        plan = _plan([_slot("repo", "repo", "str")])
        result = adapt_plan(plan, {"repo": "x"})
        assert result["steps"] == []

    def test_returned_shape(self):
        plan = _plan([_slot("repo", "repo", "str")])
        result = adapt_plan(plan, {"repo": "x"})
        assert result["plan_id"] == "test-plan"
        assert result["intent_bucket_hash"] == "abc123"
        assert "filled_slots" in result
        assert "steps" in result
        assert "output_schema" in result
