from __future__ import annotations

import re
from typing import Any

from thriftlm.v2.schemas import FilledPlan, PlanTemplate, PlanStep


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SlotFillError(Exception):
    """Required slot could not be resolved from context."""


class SlotTypeError(Exception):
    """Resolved/transformed slot value does not match the declared type_hint."""


class TransformNotFoundError(Exception):
    """Named transform is not registered."""


class TransformExecutionError(Exception):
    """Registered transform exists but failed during execution."""


# ---------------------------------------------------------------------------
# Type hint checking
# ---------------------------------------------------------------------------

def _matches_type_hint(value: Any, type_hint: str) -> bool:
    if type_hint == "str":
        return isinstance(value, str)
    if type_hint == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_hint == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_hint == "bool":
        return isinstance(value, bool)
    if type_hint == "dict":
        return isinstance(value, dict)
    if type_hint == "list":
        return isinstance(value, list)
    if type_hint == "list[str]":
        return isinstance(value, list) and all(isinstance(i, str) for i in value)
    if type_hint == "list[dict]":
        return isinstance(value, list) and all(isinstance(i, dict) for i in value)
    if type_hint == "dict[str, list[dict]]":
        if not isinstance(value, dict):
            return False
        return all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(i, dict) for i in v)
            for k, v in value.items()
        )
    # Unknown type hint — pass through rather than false-reject
    return True


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

class TransformRegistry:
    def __init__(self) -> None:
        self._transforms: dict[str, Any] = {}

    def register(self, name: str, fn) -> None:
        self._transforms[name] = fn

    def get(self, name: str):
        if name not in self._transforms:
            raise TransformNotFoundError(f"Transform not found: {name!r}")
        return self._transforms[name]

    def apply(self, name: str, value: Any, args: dict[str, Any] | None) -> Any:
        fn = self.get(name)  # raises TransformNotFoundError if missing
        try:
            return fn(value, args or {})
        except (TransformNotFoundError, SlotFillError, SlotTypeError):
            raise  # let our own exceptions propagate unchanged
        except Exception as exc:
            raise TransformExecutionError(
                f"Transform {name!r} failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------

def _filter_open(value: Any, args: dict) -> list:
    return [item for item in value if item.get("status") == "open"]


def _sort_by_date_desc(value: Any, args: dict) -> list:
    field = args["field"]
    return sorted(value, key=lambda item: item.get(field) or "", reverse=True)


def _top_n(value: Any, args: dict) -> list:
    return value[: args["n"]]


def _strip_html(value: Any, args: dict) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _group_by_status(value: Any, args: dict) -> dict:
    field = args["field"]
    result: dict[str, list] = {}
    for item in value:
        key = item.get(field) or "unknown"  # None or missing → "unknown"
        result.setdefault(key, []).append(item)
    return result


def _truncate(value: Any, args: dict) -> str:
    return value[: args["max_chars"]]


def _to_slack_bullets(value: Any, args: dict) -> str:
    prefix = args.get("prefix", "-")
    lines = []
    for item in value:
        if isinstance(item, dict):
            lines.append(f"{prefix} {str(item)}")
        else:
            lines.append(f"{prefix} {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default registry singleton (module-level, but not coupling — callers can
# instantiate their own registry if needed)
# ---------------------------------------------------------------------------

_default_registry = TransformRegistry()
_default_registry.register("filter_open", _filter_open)
_default_registry.register("sort_by_date_desc", _sort_by_date_desc)
_default_registry.register("top_n", _top_n)
_default_registry.register("strip_html", _strip_html)
_default_registry.register("group_by_status", _group_by_status)
_default_registry.register("truncate", _truncate)
_default_registry.register("to_slack_bullets", _to_slack_bullets)


# ---------------------------------------------------------------------------
# Placeholder substitution
# ---------------------------------------------------------------------------

def _substitute_inputs(
    steps: list[PlanStep],
    filled_slots: dict[str, Any],
) -> list[PlanStep]:
    """
    Replace exact full-string placeholders of the form "{slot_name}" in
    step inputs with the corresponding filled slot value.

    Rules:
    - Only replaces when the entire value string is exactly "{slot_name}".
    - Partial strings like "repo={repo}" are left unchanged.
    - Prior-step references (e.g. "{prs}") that are not in filled_slots
      are left unchanged — they resolve at execution time, not here.
    - Unknown slot names are left unchanged — no KeyError.
    - Substitution is shallow (top-level step inputs only). Nested
      placeholders inside dict or list values are intentionally NOT
      traversed or substituted. Phase 1 does not support recursive
      template expansion.
    """
    adapted: list[PlanStep] = []
    for step in steps:
        inputs = dict(step.get("inputs") or {})
        new_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if (
                isinstance(v, str)
                and v.startswith("{")
                and v.endswith("}")
                and len(v) > 2
            ):
                slot_name = v[1:-1]
                if slot_name in filled_slots:
                    new_inputs[k] = filled_slots[slot_name]
                else:
                    new_inputs[k] = v  # unknown name — leave unchanged
            else:
                new_inputs[k] = v
        adapted.append({**step, "inputs": new_inputs})  # type: ignore[misc]
    return adapted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def adapt_plan(
    plan: PlanTemplate,
    context: dict,
    registry: TransformRegistry | None = None,
) -> FilledPlan:
    """
    Resolve all SlotSpecs from *context*, apply transforms, type-check, and
    return a FilledPlan with substituted step inputs.

    Raises:
        SlotFillError: required slot missing from context with no default.
        SlotTypeError: resolved value does not match declared type_hint.
        TransformNotFoundError: slot references a transform not in the registry.
        TransformExecutionError: transform exists but raised at runtime.
    """
    reg = registry if registry is not None else _default_registry
    filled_slots: dict[str, Any] = {}

    for slot in plan["slots"]:
        source = slot["source"]
        name = slot["name"]
        type_hint = slot["type_hint"]

        if source in context:
            value = context[source]
        elif slot["required"] and slot["default"] is None:
            raise SlotFillError(
                f"Required slot {name!r} (source={source!r}) not found in context"
            )
        else:
            value = slot["default"]

        if slot.get("transform") is not None:
            value = reg.apply(slot["transform"], value, slot.get("transform_args"))

        if not _matches_type_hint(value, type_hint):
            raise SlotTypeError(
                f"Slot {name!r} expected type {type_hint!r}, "
                f"got {type(value).__name__!r}"
            )

        filled_slots[name] = value

    adapted_steps = _substitute_inputs(plan["steps"], filled_slots)

    return FilledPlan(
        plan_id=plan["plan_id"],
        intent_bucket_hash=plan["intent_bucket_hash"],
        filled_slots=filled_slots,
        steps=adapted_steps,
        output_schema=plan["output_schema"],
    )
