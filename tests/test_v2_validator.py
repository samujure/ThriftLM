"""
Tests for thriftlm.v2.validator.validate_plan — 7-stage ordered pipeline.

No network, no Redis, no Supabase, no OpenAI.
"""
from __future__ import annotations

from typing import Any

import pytest

from thriftlm.v2.validator import VALIDATOR_VERSION, validate_plan

from thriftlm.v2.schemas import FilledPlan, PlanStep, PlanTemplate, SlotSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slot(name: str, type_hint: str = "str", required: bool = True) -> SlotSpec:
    return SlotSpec(
        name=name, source=name, type_hint=type_hint,
        required=required, transform=None, transform_args=None, default=None,
    )


def _step(
    step_id: str,
    inputs: dict | None = None,
    outputs: dict | None = None,
    tool_family: str | None = None,
    side_effect: bool = False,
) -> PlanStep:
    s = PlanStep(step_id=step_id, op="noop", inputs=inputs or {}, outputs=outputs or {})
    if tool_family is not None:
        s["tool_family"] = tool_family
    if side_effect:
        s["side_effect"] = True
    return s


def _plan(
    slots: list[SlotSpec] | None = None,
    steps: list[PlanStep] | None = None,
    output_schema: dict | None = None,
    optional_outputs: list[str] | None = None,
) -> PlanTemplate:
    return PlanTemplate(
        plan_id="test-plan",
        intent_key={"action": "test", "target": "x", "goal": "y", "time_scope": None},
        intent_bucket_hash="abc123",
        description="test",
        steps=steps or [],
        slots=slots or [],
        output_schema=output_schema or {},
        optional_outputs=optional_outputs or [],
        plan_version="1",
        canonicalizer_version="v0.4",
        extractor_version="v0.1",
        validator_version=VALIDATOR_VERSION,
        created_at="2026-01-01T00:00:00Z",
    )


def _filled(
    filled_slots: dict | None = None,
    steps: list[PlanStep] | None = None,
    output_schema: dict | None = None,
) -> FilledPlan:
    return FilledPlan(
        plan_id="test-plan",
        intent_bucket_hash="abc123",
        filled_slots=filled_slots or {},
        steps=steps or [],
        output_schema=output_schema or {},
    )


_CAPS = {"tool_families": ["github"], "allow_side_effects": False}
_CAPS_SIDE_EFFECTS = {"tool_families": ["github"], "allow_side_effects": True}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_all_stages_pass(self):
        plan = _plan(
            slots=[_slot("repo", "str")],
            steps=[_step("1", inputs={"repo": "org/repo"}, outputs={"prs": "list"})],
            output_schema={"prs": "list"},
        )
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "org/repo"}, outputs={"prs": "list"})],
            output_schema={"prs": "list"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True
        assert result["failed_stage"] is None
        assert result["reason"] is None
        assert result["validator_version"] == VALIDATOR_VERSION

    def test_no_slots_no_steps(self):
        plan = _plan()
        filled = _filled()
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 1 — missing required slot
# ---------------------------------------------------------------------------

class TestStage1:
    def test_missing_required_slot(self):
        plan = _plan(slots=[_slot("repo", required=True)])
        filled = _filled(filled_slots={})  # repo missing
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "1"
        assert "repo" in result["reason"]

    def test_optional_slot_missing_passes(self):
        plan = _plan(slots=[_slot("time_scope", required=False)])
        filled = _filled(filled_slots={})
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_required_slot_present_passes(self):
        plan = _plan(slots=[_slot("repo", required=True)])
        filled = _filled(filled_slots={"repo": "org/repo"})
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 2 — slot type mismatch
# ---------------------------------------------------------------------------

class TestStage2:
    def test_wrong_slot_type(self):
        plan = _plan(slots=[_slot("count", "int")])
        filled = _filled(filled_slots={"count": "not-an-int"})
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "2"
        assert "count" in result["reason"]

    def test_correct_slot_type_passes(self):
        plan = _plan(slots=[_slot("count", "int")])
        filled = _filled(filled_slots={"count": 42})
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_list_type_passes(self):
        plan = _plan(slots=[_slot("items", "list")])
        filled = _filled(filled_slots={"items": [1, 2, 3]})
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 3 — unsatisfied step input
# ---------------------------------------------------------------------------

class TestStage3:
    def test_unresolved_placeholder_fails(self):
        """Step input is still "{missing}" — not in slots or prior outputs."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"data": "{missing_key}"})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "3"

    def test_resolved_slot_in_input_passes(self):
        """Adapter substituted the slot — value is now a concrete string, not a placeholder."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "org/repo"}, outputs={"prs": "list"})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_prior_step_output_reference_passes(self):
        """Step 2 inputs reference step 1's output key as a placeholder — that's valid."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[
                _step("1", inputs={"repo": "org/repo"}, outputs={"prs": "list"}),
                _step("2", inputs={"prs": "{prs}"}, outputs={"summary": "str"}),
            ],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_future_step_output_reference_fails(self):
        """A step cannot reference an output that is only produced later."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[
                _step("1", inputs={"prs": "{prs}"}),  # prs not produced yet
                _step("2", inputs={"repo": "org/repo"}, outputs={"prs": "list"}),
            ],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "3"
        assert result["reason"].startswith("unsatisfied_step_input:")

    def test_slot_placeholder_still_present_fails(self):
        """Adapter must substitute top-level slot placeholders. If '{repo}' survives
        into the filled plan while 'repo' is in filled_slots, stage 3 rejects it as
        an unsubstituted slot placeholder."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "{repo}"})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "3"
        assert result["reason"].startswith("unsubstituted_slot_placeholder:")


# ---------------------------------------------------------------------------
# Stage 4 — missing tool family
# ---------------------------------------------------------------------------

class TestStage4:
    def test_missing_tool_family(self):
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "org/repo"}, tool_family="jira")],
        )
        caps = {"tool_families": ["github"], "allow_side_effects": False}
        result = validate_plan(plan, filled, caps)
        assert result["ok"] is False
        assert result["failed_stage"] == "4"
        assert "jira" in result["reason"]

    def test_matching_tool_family_passes(self):
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "org/repo"}, tool_family="github")],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_no_tool_family_on_step_passes(self):
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"repo": "org/repo"})],  # no tool_family
        )
        result = validate_plan(plan, filled, {})  # empty caps
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 5 — unresolved nested placeholder
# ---------------------------------------------------------------------------

class TestStage5:
    def test_nested_placeholder_in_dict_fails(self):
        """Adapter does shallow substitution only; nested {repo} inside a dict must be caught."""
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"config": {"url": "{repo}"}})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "5"
        assert result["reason"] == "unresolved_placeholder"

    def test_nested_placeholder_in_list_fails(self):
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"items": ["{repo}", "literal"]})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "5"

    def test_resolved_nested_value_passes(self):
        plan = _plan(slots=[_slot("repo")])
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[_step("1", inputs={"config": {"url": "org/repo"}})],
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_placeholder_in_non_input_step_field_caught(self):
        """Placeholder-shaped string in a non-inputs step field must be caught by stage 5."""
        plan = _plan(slots=[_slot("repo")])
        # outputs values are type hints in practice; using a placeholder here to
        # prove stage 5 walks non-inputs step fields
        step = _step("1", inputs={"repo": "org/repo"}, outputs={"summary": "{unresolved}"})
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[step],
            output_schema={"summary": "str"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "5"
        assert result["reason"] == "unresolved_placeholder"

    def test_valid_prior_step_reference_not_failed_by_stage5(self):
        """Top-level input {prs} that stage 3 accepts as a valid step-output ref
        must not be double-failed by stage 5."""
        plan = _plan(slots=[_slot("repo")], output_schema={"summary": "str"})
        filled = _filled(
            filled_slots={"repo": "org/repo"},
            steps=[
                _step("1", inputs={"repo": "org/repo"}, outputs={"prs": "list"}),
                _step("2", inputs={"prs": "{prs}"}, outputs={"summary": "str"}),
            ],
            output_schema={"summary": "str"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 6 — missing output producer
# ---------------------------------------------------------------------------

class TestStage6:
    def test_required_output_no_producer(self):
        plan = _plan(
            output_schema={"summary": "str"},
            optional_outputs=[],
        )
        filled = _filled(
            steps=[_step("1", outputs={"other": "str"})],  # "summary" never produced
            output_schema={"summary": "str"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "6"
        assert "summary" in result["reason"]

    def test_output_produced_passes(self):
        plan = _plan(output_schema={"summary": "str"})
        filled = _filled(
            steps=[_step("1", outputs={"summary": "str"})],
            output_schema={"summary": "str"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True

    def test_output_schema_mismatch_fails(self):
        """filled_plan output_schema diverging from plan output_schema is rejected."""
        plan = _plan(output_schema={"summary": "str"})
        filled = _filled(
            steps=[_step("1", outputs={"summary": "str"})],
            output_schema={"summary": "str", "extra": "str"},  # mutated
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is False
        assert result["failed_stage"] == "6"
        assert result["reason"] == "output_schema_mismatch"

    def test_optional_output_not_produced_passes(self):
        """optional_outputs are excluded from stage 6."""
        plan = _plan(
            output_schema={"summary": "str", "debug": "str"},
            optional_outputs=["debug"],
        )
        filled = _filled(
            steps=[_step("1", outputs={"summary": "str"})],  # "debug" not produced
            output_schema={"summary": "str", "debug": "str"},
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage 7 — side effects blocked
# ---------------------------------------------------------------------------

class TestStage7:
    def test_side_effect_blocked(self):
        plan = _plan()
        filled = _filled(
            steps=[_step("1", side_effect=True)],
        )
        caps = {"allow_side_effects": False}
        result = validate_plan(plan, filled, caps)
        assert result["ok"] is False
        assert result["failed_stage"] == "7"
        assert result["reason"] == "side_effects_not_allowed"

    def test_side_effect_allowed(self):
        plan = _plan()
        filled = _filled(steps=[_step("1", side_effect=True)])
        result = validate_plan(plan, filled, _CAPS_SIDE_EFFECTS)
        assert result["ok"] is True

    def test_no_side_effects_passes_without_permission(self):
        plan = _plan()
        filled = _filled(steps=[_step("1", side_effect=False)])
        result = validate_plan(plan, filled, {"allow_side_effects": False})
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Stage ordering — stops at first failure
# ---------------------------------------------------------------------------

class TestStageOrdering:
    def test_stops_at_stage_1_not_stage_2(self):
        """Both stage 1 (missing required slot) and stage 2 (type mismatch on other slot)
        would fail, but only stage 1 is reported."""
        plan = _plan(slots=[
            _slot("repo", "str", required=True),
            _slot("count", "int", required=False),
        ])
        filled = _filled(filled_slots={"count": "wrong-type"})  # repo missing, count wrong type
        result = validate_plan(plan, filled, _CAPS)
        assert result["failed_stage"] == "1"

    def test_stops_at_stage_2_not_stage_3(self):
        plan = _plan(slots=[_slot("count", "int")])
        filled = _filled(
            filled_slots={"count": "not-an-int"},
            steps=[_step("1", inputs={"x": "{missing}"})],  # stage 3 would also fail
        )
        result = validate_plan(plan, filled, _CAPS)
        assert result["failed_stage"] == "2"
