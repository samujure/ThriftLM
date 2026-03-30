from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from thriftlm.v2.schemas import ExtractionResult, PlanStep, PlanTemplate, SlotSpec

EXTRACTOR_VERSION = "v0.4"

# ---------------------------------------------------------------------------
# PII scrubbing patterns — applied in order
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"[A-Za-z0-9_\-]{40,}"),
        "[REDACTED_KEY]",
    ),
]


def _scrub_pii(value: str) -> str:
    """Apply PII redaction patterns in order to a single string value."""
    for pattern, replacement in _PII_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

# String values in step outputs that are recognized type-hint tokens are
# preserved as-is.  Anything else (real runtime strings like "done", "hello")
# is mapped through _infer_type_hint().  "string" is included because traces
# commonly use it as an alias for "str".
_KNOWN_TYPE_HINTS = {"str", "int", "float", "bool", "list", "dict", "any", "none", "string"}


def _infer_type_hint(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


# ---------------------------------------------------------------------------
# Public API — is_extractable_trace
# ---------------------------------------------------------------------------

def is_extractable_trace(trace: dict) -> tuple[bool, str | None]:
    """
    Returns (True, None) if trace is extractable.
    Returns (False, reason) if not.

    Checks (in order):
    - trace must be a dict → else ("malformed_trace")
    - trace must have "steps" key → else ("missing_steps")
    - trace["steps"] must be a non-empty list → else ("empty_steps")
    """
    if not isinstance(trace, dict):
        return False, "malformed_trace"
    if "steps" not in trace:
        return False, "missing_steps"
    steps = trace["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        return False, "empty_steps"
    return True, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context_key_map(
    context: dict,
    canonicalization_result: dict,
) -> dict[str, Any]:
    """
    Build a map from key_name → actual_runtime_value.

    Sources:
    - canonicalization_result["intent_key"] (only keys with non-None string values)
    - context (key → value), which overrides intent_key on collision
    """
    key_map: dict[str, Any] = {}

    intent_key: dict = canonicalization_result.get("intent_key") or {}
    for k, v in intent_key.items():
        if isinstance(v, str):
            key_map[k] = v

    # context takes precedence over intent_key on the same key name
    for k, v in context.items():
        key_map[k] = v

    return key_map


def _build_value_to_keys(key_map: dict[str, Any]) -> dict[str, list[str]]:
    """
    Invert the context key map for string values only:
    runtime_value → sorted list of key names.

    Sorting gives deterministic alphabetical fallback when no input-key match exists.
    """
    value_to_keys: dict[str, list[str]] = {}
    for k, v in key_map.items():
        if isinstance(v, str):
            value_to_keys.setdefault(v, []).append(k)
    for lst in value_to_keys.values():
        lst.sort()
    return value_to_keys


# Called by the store path (_server.py /v2/plan/store), not by extract_plan_template().
# When persisting an ExtractionResult, callers must call:
#   build_structural_signature(template["steps"], [s["name"] for s in template["slots"]], canonicalization_result)
# and include the result in the DB insert.
def build_structural_signature(
    abstracted_steps: list[dict],
    slot_names: list[str],
    canonicalization_result: dict,
) -> dict[str, Any]:
    """
    Build a structural_signature dict compatible with plan_cache.py scoring.

    Shape:
    {
      "required_context_keys": sorted unique slot names,
      "tool_families": sorted unique non-None tool_family values across steps,
      "has_side_effects": True if any step.side_effect == True,
      "format": intent_key.get("format"),
      "audience": intent_key.get("audience"),
      "step_count": len(abstracted_steps),
    }
    """
    intent_key: dict = canonicalization_result.get("intent_key") or {}

    tool_families: list[str] = sorted({
        step.get("tool_family")
        for step in abstracted_steps
        if step.get("tool_family") is not None
    })

    has_side_effects: bool = any(
        step.get("side_effect") is True for step in abstracted_steps
    )

    return {
        "required_context_keys": sorted(set(slot_names)),
        "tool_families": tool_families,
        "has_side_effects": has_side_effects,
        "format": intent_key.get("format"),
        "audience": intent_key.get("audience"),
        "step_count": len(abstracted_steps),
    }


# ---------------------------------------------------------------------------
# Public API — extract_plan_template
# ---------------------------------------------------------------------------

def extract_plan_template(
    task: str,
    context: dict,
    execution_trace: dict,
    planner_output: dict | None,
    canonicalization_result: dict,
) -> ExtractionResult:
    """
    Generalize a completed execution trace into a reusable PlanTemplate.
    Pure heuristic — no LLM, no network, no Redis, no Supabase.
    planner_output is accepted but not used in v0.1.
    """
    # ------------------------------------------------------------------
    # Guard: validate canonicalization_result has required keys
    # ------------------------------------------------------------------
    if (
        not isinstance(canonicalization_result, dict)
        or "intent_key" not in canonicalization_result
        or "intent_bucket_hash" not in canonicalization_result
    ):
        return ExtractionResult(
            ok=False,
            template=None,
            extraction_confidence=0.0,
            generalization_notes="invalid_canonicalization_result",
            refusal_reason="invalid_canonicalization_result",
        )

    # ------------------------------------------------------------------
    # Refusal 1: is_extractable_trace
    # ------------------------------------------------------------------
    extractable, reason = is_extractable_trace(execution_trace)
    if not extractable:
        return ExtractionResult(
            ok=False,
            template=None,
            extraction_confidence=0.0,
            generalization_notes=reason,
            refusal_reason=reason,
        )

    # ------------------------------------------------------------------
    # Build context key map and its value→keys inversion
    # ------------------------------------------------------------------
    key_map = _build_context_key_map(context, canonicalization_result)
    value_to_keys = _build_value_to_keys(key_map)

    # ------------------------------------------------------------------
    # Walk steps: abstract string input values → placeholders
    #
    # Only replace exact top-level string input values (v0.1 spec).
    # Non-string values are kept as-is silently.
    #
    # Tie-breaking rule for duplicate context values:
    # Among candidate key names that share the same runtime value, prefer
    # the one whose name also appears as a key in the same step's inputs
    # dict (position-sensitive co-occurrence).  If none match, use the
    # first name in alphabetical order (guaranteed by _build_value_to_keys).
    # ------------------------------------------------------------------
    raw_steps: list[dict] = execution_trace["steps"]
    abstracted_steps: list[PlanStep] = []
    slots_by_name: dict[str, SlotSpec] = {}  # preserves insertion (first-appearance) order
    slot_order: list[str] = []

    total_step_inputs: int = 0
    matched_input_count: int = 0
    unique_slot_names: set[str] = set()

    for step in raw_steps:
        new_step = dict(step)
        inputs: dict[str, Any] = dict(step.get("inputs") or {})
        new_inputs: dict[str, Any] = {}

        for input_key, input_val in inputs.items():
            total_step_inputs += 1

            if not isinstance(input_val, str):
                # Non-string values: keep as-is, no abstraction in v0.1
                new_inputs[input_key] = input_val
                continue

            candidate_keys = value_to_keys.get(input_val)
            if candidate_keys:
                # Prefer a candidate whose name also appears as a key in this
                # step's inputs; otherwise fall back to alphabetical order.
                chosen: str | None = None
                for ck in candidate_keys:
                    if ck in inputs:
                        chosen = ck
                        break
                if chosen is None:
                    chosen = candidate_keys[0]  # already alphabetically sorted

                new_inputs[input_key] = f"{{{chosen}}}"
                matched_input_count += 1
                unique_slot_names.add(chosen)

                if chosen not in slots_by_name:
                    slot_order.append(chosen)
                    runtime_val = key_map[chosen]
                    slots_by_name[chosen] = SlotSpec(
                        name=chosen,
                        source=chosen,
                        type_hint=_infer_type_hint(runtime_val),
                        required=True,
                        transform=None,
                        transform_args=None,
                        default=None,
                    )
            else:
                # Hardcoded literal — apply PII scrubbing before storing
                new_inputs[input_key] = _scrub_pii(input_val)

        new_step["inputs"] = new_inputs
        abstracted_steps.append(new_step)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Compute extraction_confidence
    # slots_found = unique slot names extracted (not raw matched input count)
    # TODO: confidence = unique_slots / total_inputs is intentionally strict.
    # A trace using the same slot across many steps (e.g. repo x4) scores low
    # even if the abstraction is solid. Consider matched_input_count / total_step_inputs
    # or a hybrid score in a future revision.
    # ------------------------------------------------------------------
    slots_found: int = len(unique_slot_names)
    extraction_confidence: float = slots_found / max(1, total_step_inputs)

    # ------------------------------------------------------------------
    # Refusal 2: all steps are side-effecting and no slots were extracted
    # ------------------------------------------------------------------
    all_have_side_effects = all(
        step.get("side_effect") is True for step in raw_steps
    )
    if all_have_side_effects and slots_found == 0:
        return ExtractionResult(
            ok=False,
            template=None,
            extraction_confidence=extraction_confidence,
            generalization_notes="all_side_effects",
            refusal_reason="all_side_effects",
        )

    # ------------------------------------------------------------------
    # Refusal 3: low confidence
    # ------------------------------------------------------------------
    if extraction_confidence < 0.5:
        return ExtractionResult(
            ok=False,
            template=None,
            extraction_confidence=extraction_confidence,
            generalization_notes="low_confidence",
            refusal_reason="low_confidence",
        )

    # ------------------------------------------------------------------
    # PII scrub SlotSpec defaults (v0.1: all defaults are None, but
    # guard is here for correctness if that ever changes)
    # ------------------------------------------------------------------
    for slot in slots_by_name.values():
        if slot["default"] is not None and isinstance(slot["default"], str):
            slot["default"] = _scrub_pii(slot["default"])

    # ------------------------------------------------------------------
    # Infer output_schema from LAST step outputs only
    # ------------------------------------------------------------------
    last_step = abstracted_steps[-1]
    last_outputs: dict = last_step.get("outputs") or {}
    output_schema: dict[str, str] = {}
    for out_key, out_val in last_outputs.items():
        if (
            isinstance(out_val, str)
            and out_val.lower().split("[")[0].strip() in _KNOWN_TYPE_HINTS
        ):
            # Recognized type-hint token (e.g. "str", "list[pr]", "string") — keep as-is
            output_schema[out_key] = out_val
        else:
            output_schema[out_key] = _infer_type_hint(out_val)

    # ------------------------------------------------------------------
    # Build PlanTemplate
    # ------------------------------------------------------------------
    template = PlanTemplate(
        plan_id=str(uuid4()),
        intent_key=canonicalization_result["intent_key"],
        intent_bucket_hash=canonicalization_result["intent_bucket_hash"],
        description=str(canonicalization_result.get("raw_canonical_text", ""))[:200],
        steps=abstracted_steps,
        slots=[slots_by_name[name] for name in slot_order],
        output_schema=output_schema,
        optional_outputs=[],
        plan_version="1",
        canonicalizer_version=canonicalization_result.get("canonicalizer_version", "unknown"),
        extractor_version=EXTRACTOR_VERSION,
        validator_version="unset",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return ExtractionResult(
        ok=True,
        template=template,
        extraction_confidence=extraction_confidence,
        generalization_notes=(
            f"{slots_found} unique slots extracted from {matched_input_count} matched inputs"
            f" ({total_step_inputs} total)"
        ),
        refusal_reason=None,
    )
