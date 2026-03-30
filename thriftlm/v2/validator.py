from __future__ import annotations

import re
from typing import Any

from thriftlm.v2.adapter import _matches_type_hint
from thriftlm.v2.schemas import FilledPlan, PlanTemplate, ValidationResult

VALIDATOR_VERSION = "v0.4"


# Shared placeholder pattern — used in stage 3 and stage 5.
_PLACEHOLDER_RE = re.compile(r'^\{[^{}]+\}$')


def _fail(stage: str, reason: str) -> ValidationResult:
    return ValidationResult(
        ok=False,
        failed_stage=stage,
        reason=reason,
        validator_version=VALIDATOR_VERSION,
    )


def _walk_values(obj: Any):
    """Yield every leaf string reachable inside dicts, lists, and plain values."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_values(item)


def _walk_step_stage5(step: PlanStep):
    """
    Yield leaf strings from a step for stage 5 placeholder checking.

    Walks the entire step dict recursively with one exception:
    top-level string values in step["inputs"] are skipped — those are owned
    by stage 3, which already validated exact-placeholder references there.
    All other step fields (op, tool_family, outputs, etc.) are walked fully.
    Nested dict/list values inside step["inputs"] are still descended into.

    Dict keys are never yielded — only values.
    """
    for field, val in step.items():
        if field == "inputs" and isinstance(val, dict):
            # Skip top-level string input values; descend into containers only.
            for input_val in val.values():
                if isinstance(input_val, (dict, list)):
                    yield from _walk_values(input_val)
        else:
            yield from _walk_values(val)


def validate_plan(
    plan: PlanTemplate,
    filled_plan: FilledPlan,
    runtime_caps: dict,
) -> ValidationResult:
    """
    Run the 7-stage ordered validation pipeline.

    Stops at the first failing stage and returns a ValidationResult with
    ok=False, failed_stage, and a short machine-readable reason string.

    Args:
        plan:         Original PlanTemplate (needed for slot specs, optional_outputs).
        filled_plan:  FilledPlan produced by adapter.adapt_plan().
        runtime_caps: Runtime capabilities dict (tool_families, allow_side_effects).
    """
    filled_slots = filled_plan["filled_slots"]
    steps = filled_plan["steps"]

    # ------------------------------------------------------------------
    # Stage 1 — all required slots are resolved
    # ------------------------------------------------------------------
    for slot in plan["slots"]:
        if slot["required"] and slot["name"] not in filled_slots:
            return _fail("1", f"missing_required_slot:{slot['name']}")

    # ------------------------------------------------------------------
    # Stage 2 — slot values conform to declared type_hints
    # ------------------------------------------------------------------
    for slot in plan["slots"]:
        name = slot["name"]
        if name in filled_slots:
            if not _matches_type_hint(filled_slots[name], slot["type_hint"]):
                return _fail("2", f"slot_type_mismatch:{name}")

    # ------------------------------------------------------------------
    # Stage 3 — all step inputs satisfied (no unresolved placeholders,
    #           no references to unavailable slots or outputs)
    # ------------------------------------------------------------------
    available_slots: set[str] = set(filled_slots.keys())
    produced_outputs: set[str] = set()

    for step in steps:
        step_id = step.get("step_id", "?")
        for input_name, value in (step.get("inputs") or {}).items():
            if isinstance(value, str) and _PLACEHOLDER_RE.match(value):
                ref_name = value[1:-1]
                if ref_name in produced_outputs:
                    pass  # valid prior-step output reference — executor resolves this
                elif ref_name in filled_slots:
                    # Slot was filled but adapter left the placeholder unsubstituted
                    return _fail("3", f"unsubstituted_slot_placeholder:{step_id}:{input_name}")
                else:
                    # Name is unknown — neither a slot nor a prior step output
                    return _fail("3", f"unsatisfied_step_input:{step_id}:{input_name}")
        # All outputs from this step become available to subsequent steps
        produced_outputs.update((step.get("outputs") or {}).keys())

    # ------------------------------------------------------------------
    # Stage 4 — referenced tool_families present in runtime_caps
    # ------------------------------------------------------------------
    runtime_families: set[str] = set(runtime_caps.get("tool_families") or [])
    for step in steps:
        tf = step.get("tool_family")
        if tf and tf not in runtime_families:
            return _fail("4", f"missing_tool_family:{tf}")

    # ------------------------------------------------------------------
    # Stage 5 — no unresolved slot-shaped placeholders remain anywhere
    #           in the filled steps (validator as final safety net)
    #
    # Walks every step fully via _walk_step_stage5, which skips only the
    # top-level string values in step["inputs"] (already owned by stage 3)
    # but descends into nested containers under inputs and inspects all
    # other step fields recursively.
    # ------------------------------------------------------------------
    for step in steps:
        for value in _walk_step_stage5(step):
            if _PLACEHOLDER_RE.match(value):
                return _fail("5", "unresolved_placeholder")

    # ------------------------------------------------------------------
    # Stage 6 — every non-optional output_schema field has a producer
    #
    # Authoritative source is plan["output_schema"] — the original template.
    # filled_plan["output_schema"] is caller-supplied and could be mutated;
    # a mismatch is itself a validation failure.
    # ------------------------------------------------------------------
    all_produced: set[str] = set()
    for step in steps:
        all_produced.update((step.get("outputs") or {}).keys())

    if filled_plan["output_schema"] != plan["output_schema"]:
        return _fail("6", "output_schema_mismatch")

    optional = set(plan.get("optional_outputs") or [])
    for field in plan["output_schema"]:
        if field not in optional and field not in all_produced:
            return _fail("6", f"missing_output_producer:{field}")

    # ------------------------------------------------------------------
    # Stage 7 — side-effecting steps permitted by runtime_caps
    # ------------------------------------------------------------------
    allow_side_effects: bool = runtime_caps.get("allow_side_effects", False)
    if not allow_side_effects:
        for step in steps:
            if step.get("side_effect") is True:
                return _fail("7", "side_effects_not_allowed")

    return ValidationResult(
        ok=True,
        failed_stage=None,
        reason=None,
        validator_version=VALIDATOR_VERSION,
    )
