"""
Tests for thriftlm/v2/extractor.py — Phase 2 unit tests.

No network, no Redis, no Supabase, no OpenAI.
All inputs are plain dict literals.
"""
from __future__ import annotations

import pytest

from thriftlm.v2.extractor import (
    EXTRACTOR_VERSION,
    build_structural_signature,
    extract_plan_template,
    is_extractable_trace,
)


# ---------------------------------------------------------------------------
# Fixtures / shared builders
# ---------------------------------------------------------------------------

def _canon(
    action: str = "summarize",
    target: str = "pull_requests",
    goal: str = "identify_blockers",
    bucket: str = "abc123deadbeef12",
    raw_text: str = "Summarize pull requests to identify blockers",
    format: str | None = None,
    audience: str | None = None,
    tool_family: str | None = "github",
) -> dict:
    intent_key: dict = {
        "action": action,
        "target": target,
        "goal": goal,
        "time_scope": None,
        "tool_family": tool_family,
    }
    if format is not None:
        intent_key["format"] = format
    if audience is not None:
        intent_key["audience"] = audience
    return {
        "intent_key": intent_key,
        "intent_bucket_hash": bucket,
        "confidence": 0.92,
        "canonicalizer_version": "v0.4",
        "raw_canonical_text": raw_text,
        "from_cache": False,
    }


def _two_step_trace(repo: str = "org/my-repo", account: str = "acc-42") -> dict:
    return {
        "steps": [
            {
                "step_id": "1",
                "op": "fetch_pull_requests",
                "inputs": {"repo": repo},
                "outputs": {"prs": "list[pr]"},
            },
            {
                "step_id": "2",
                "op": "produce_summary",
                "inputs": {"account_id": account, "data": "static_value"},
                "outputs": {"summary": "string"},
            },
        ]
    }


# ---------------------------------------------------------------------------
# is_extractable_trace
# ---------------------------------------------------------------------------

class TestIsExtractableTrace:
    def test_valid_trace_returns_true(self):
        trace = {"steps": [{"step_id": "1", "op": "fetch", "inputs": {}, "outputs": {}}]}
        ok, reason = is_extractable_trace(trace)
        assert ok is True
        assert reason is None

    def test_non_dict_returns_malformed(self):
        ok, reason = is_extractable_trace("not a dict")  # type: ignore[arg-type]
        assert ok is False
        assert reason == "malformed_trace"

    def test_none_returns_malformed(self):
        ok, reason = is_extractable_trace(None)  # type: ignore[arg-type]
        assert ok is False
        assert reason == "malformed_trace"

    def test_missing_steps_key(self):
        ok, reason = is_extractable_trace({"ops": []})
        assert ok is False
        assert reason == "missing_steps"

    def test_empty_steps_list(self):
        ok, reason = is_extractable_trace({"steps": []})
        assert ok is False
        assert reason == "empty_steps"

    def test_steps_not_a_list(self):
        ok, reason = is_extractable_trace({"steps": "not-a-list"})
        assert ok is False
        assert reason == "empty_steps"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_two_steps_two_slots(self):
        context = {"repo": "org/my-repo", "account_id": "acc-42"}
        trace = _two_step_trace()
        canon = _canon()

        result = extract_plan_template(
            task="summarize pull requests",
            context=context,
            execution_trace=trace,
            planner_output=None,
            canonicalization_result=canon,
        )

        assert result["ok"] is True
        assert result["refusal_reason"] is None
        tmpl = result["template"]
        assert tmpl is not None
        assert tmpl["extractor_version"] == EXTRACTOR_VERSION
        assert tmpl["plan_version"] == "1"
        assert tmpl["validator_version"] == "unset"
        assert tmpl["intent_bucket_hash"] == canon["intent_bucket_hash"]

        slot_names = {s["name"] for s in tmpl["slots"]}
        assert "repo" in slot_names
        assert "account_id" in slot_names
        assert len(tmpl["slots"]) == 2

        # Step inputs should be placeholders
        step1_inputs = tmpl["steps"][0]["inputs"]
        assert step1_inputs["repo"] == "{repo}"

        step2_inputs = tmpl["steps"][1]["inputs"]
        assert step2_inputs["account_id"] == "{account_id}"
        # "static_value" is a hardcoded literal — not a slot
        assert step2_inputs["data"] == "static_value"

    def test_slot_specs_have_correct_fields(self):
        context = {"repo": "org/my-repo", "account_id": "acc-42"}
        trace = _two_step_trace()
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"]
        for slot in result["template"]["slots"]:
            assert slot["required"] is True
            assert slot["transform"] is None
            assert slot["transform_args"] is None
            assert slot["default"] is None
            assert slot["source"] == slot["name"]

    def test_output_schema_from_last_step_only(self):
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "fetch",
                    "inputs": {"repo": "org/r"},
                    "outputs": {"prs": "list[pr]", "should_not_appear": "str"},
                },
                {
                    "step_id": "2",
                    "op": "summarize",
                    "inputs": {"repo": "org/r"},
                    "outputs": {"summary": "string"},
                },
            ]
        }
        context = {"repo": "org/r"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"]
        # Only last step's outputs appear
        assert "summary" in result["template"]["output_schema"]
        assert "prs" not in result["template"]["output_schema"]
        assert "should_not_appear" not in result["template"]["output_schema"]

    def test_slot_order_follows_first_appearance(self):
        trace = {
            "steps": [
                {"step_id": "1", "op": "a", "inputs": {"x": "val_x"}, "outputs": {}},
                {"step_id": "2", "op": "b", "inputs": {"y": "val_y"}, "outputs": {"out": "str"}},
            ]
        }
        context = {"x": "val_x", "y": "val_y"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"]
        slot_names = [s["name"] for s in result["template"]["slots"]]
        assert slot_names == ["x", "y"]

    def test_description_truncated_to_200_chars(self):
        long_text = "x" * 300
        canon = _canon(raw_text=long_text)
        trace = {
            "steps": [
                {"step_id": "1", "op": "op", "inputs": {"k": "v"}, "outputs": {"r": "str"}}
            ]
        }
        context = {"k": "v"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=canon,
        )
        assert result["ok"]
        assert len(result["template"]["description"]) == 200

    def test_generalization_notes_format(self):
        context = {"repo": "org/my-repo", "account_id": "acc-42"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=_two_step_trace(), planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"]
        notes = result["generalization_notes"]
        assert notes is not None
        assert "2 unique slots extracted from 2 matched inputs (3 total)" in notes


# ---------------------------------------------------------------------------
# Refusal cases
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_missing_steps_key_in_extract(self):
        result = extract_plan_template(
            task="t", context={},
            execution_trace={"ops": []}, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "missing_steps"

    def test_empty_steps_refusal(self):
        result = extract_plan_template(
            task="t", context={},
            execution_trace={"steps": []}, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "empty_steps"

    def test_all_side_effects_no_slots(self):
        # All steps side_effect=True, context provides nothing matching inputs
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "send_email",
                    "inputs": {"to": "nobody@example.com"},  # matches no context key value
                    "outputs": {},
                    "side_effect": True,
                },
                {
                    "step_id": "2",
                    "op": "post_slack",
                    "inputs": {"channel": "#general"},
                    "outputs": {"result": "str"},
                    "side_effect": True,
                },
            ]
        }
        # Context is non-empty but values don't match trace inputs
        context = {"repo": "org/r", "account_id": "acc-1"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "all_side_effects"

    def test_low_confidence_four_inputs_zero_matches(self):
        # 4 input values, context is non-empty but none of its values match
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "do_a",
                    "inputs": {"a": "literal_a", "b": "literal_b"},
                    "outputs": {},
                },
                {
                    "step_id": "2",
                    "op": "do_b",
                    "inputs": {"c": "literal_c", "d": "literal_d"},
                    "outputs": {"out": "str"},
                },
            ]
        }
        context = {"repo": "completely_different", "account_id": "also_different"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "low_confidence"
        assert result["extraction_confidence"] == 0.0

    def test_invalid_canonicalization_result_missing_keys(self):
        result = extract_plan_template(
            task="t", context={"repo": "r"},
            execution_trace=_two_step_trace(), planner_output=None,
            canonicalization_result={"confidence": 0.9},  # missing intent_key + hash
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "invalid_canonicalization_result"

    def test_invalid_canonicalization_result_not_dict(self):
        result = extract_plan_template(
            task="t", context={},
            execution_trace=_two_step_trace(), planner_output=None,
            canonicalization_result=None,  # type: ignore[arg-type]
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "invalid_canonicalization_result"

    def test_invalid_canon_missing_bucket_hash(self):
        result = extract_plan_template(
            task="t", context={},
            execution_trace=_two_step_trace(), planner_output=None,
            canonicalization_result={
                "intent_key": {"action": "a", "target": "b", "goal": "c", "time_scope": None}
                # missing intent_bucket_hash
            },
        )
        assert result["ok"] is False
        assert result["refusal_reason"] == "invalid_canonicalization_result"


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------

class TestPIIScrubbing:
    def test_email_in_literal_is_redacted(self):
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "send",
                    "inputs": {"to": "alice@example.com"},  # no matching context value
                    "outputs": {"ok": "bool"},
                },
            ]
        }
        context = {"repo": "org/r"}  # value "org/r" != "alice@example.com"
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        # 1 input, 0 slots → confidence = 0.0 < 0.5 → low_confidence refusal
        # But we can still inspect via extraction_confidence; check the scrub
        # by using a context that provides a match so we get ok=True on other inputs.
        # Easier: use a 2-input trace where 1 matches (confidence >= 0.5).
        trace2 = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "send",
                    "inputs": {
                        "to": "alice@example.com",
                        "repo": "org/r",
                    },
                    "outputs": {"done": "bool"},
                }
            ]
        }
        result2 = extract_plan_template(
            task="t", context={"repo": "org/r"},
            execution_trace=trace2, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result2["ok"] is True
        step_inputs = result2["template"]["steps"][0]["inputs"]
        assert step_inputs["to"] == "[REDACTED_EMAIL]"
        assert step_inputs["repo"] == "{repo}"

    def test_api_key_in_literal_is_redacted(self):
        secret = "A" * 40  # 40 alphanumeric chars — matches the API key pattern
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "call_api",
                    "inputs": {
                        "token": secret,
                        "repo": "org/r",
                    },
                    "outputs": {"result": "str"},
                }
            ]
        }
        result = extract_plan_template(
            task="t", context={"repo": "org/r"},
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        step_inputs = result["template"]["steps"][0]["inputs"]
        assert step_inputs["token"] == "[REDACTED_KEY]"
        assert step_inputs["repo"] == "{repo}"


# ---------------------------------------------------------------------------
# structural_signature
# ---------------------------------------------------------------------------

class TestStructuralSignature:
    def _steps_with_tool_family(self) -> list[dict]:
        return [
            {"step_id": "1", "op": "a", "tool_family": "github", "side_effect": False,
             "inputs": {}, "outputs": {}},
            {"step_id": "2", "op": "b", "tool_family": "slack", "side_effect": True,
             "inputs": {}, "outputs": {}},
            {"step_id": "3", "op": "c", "tool_family": "github",
             "inputs": {}, "outputs": {"out": "str"}},
        ]

    def test_tool_families_sorted_and_unique(self):
        steps = self._steps_with_tool_family()
        sig = build_structural_signature(steps, ["repo", "account_id"], _canon())
        assert sig["tool_families"] == ["github", "slack"]

    def test_required_context_keys_sorted(self):
        steps = self._steps_with_tool_family()
        sig = build_structural_signature(steps, ["repo", "account_id"], _canon())
        assert sig["required_context_keys"] == ["account_id", "repo"]

    def test_has_side_effects_true_when_any_step_side_effectful(self):
        steps = self._steps_with_tool_family()
        sig = build_structural_signature(steps, [], _canon())
        assert sig["has_side_effects"] is True

    def test_has_side_effects_false_when_no_side_effects(self):
        steps = [
            {"step_id": "1", "op": "a", "side_effect": False, "inputs": {}, "outputs": {}},
        ]
        sig = build_structural_signature(steps, [], _canon())
        assert sig["has_side_effects"] is False

    def test_has_side_effects_false_when_field_absent(self):
        steps = [{"step_id": "1", "op": "a", "inputs": {}, "outputs": {}}]
        sig = build_structural_signature(steps, [], _canon())
        assert sig["has_side_effects"] is False

    def test_step_count(self):
        steps = self._steps_with_tool_family()
        sig = build_structural_signature(steps, [], _canon())
        assert sig["step_count"] == 3

    def test_format_and_audience_from_intent_key(self):
        canon = _canon(format="slack_message", audience="engineering")
        steps = [{"step_id": "1", "op": "a", "inputs": {}, "outputs": {}}]
        sig = build_structural_signature(steps, [], canon)
        assert sig["format"] == "slack_message"
        assert sig["audience"] == "engineering"

    def test_format_none_when_not_in_intent_key(self):
        canon = _canon()  # no format/audience set
        steps = [{"step_id": "1", "op": "a", "inputs": {}, "outputs": {}}]
        sig = build_structural_signature(steps, [], canon)
        assert sig["format"] is None
        assert sig["audience"] is None


# ---------------------------------------------------------------------------
# extraction_confidence
# ---------------------------------------------------------------------------

class TestExtractionConfidence:
    def test_all_inputs_matched(self):
        trace = {
            "steps": [
                {"step_id": "1", "op": "a", "inputs": {"repo": "org/r"}, "outputs": {}},
                {"step_id": "2", "op": "b", "inputs": {"account": "acc-1"}, "outputs": {"out": "str"}},
            ]
        }
        context = {"repo": "org/r", "account": "acc-1"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["extraction_confidence"] == 1.0

    def test_partial_match_confidence(self):
        # 4 inputs, 2 matched → confidence = 0.5
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "a",
                    "inputs": {
                        "repo": "org/r",   # matches context
                        "lit1": "literal1",  # no match
                        "lit2": "literal2",  # no match
                        "account": "acc-1",  # matches context
                    },
                    "outputs": {"out": "str"},
                }
            ]
        }
        context = {"repo": "org/r", "account": "acc-1"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["extraction_confidence"] == pytest.approx(0.5)

    def test_confidence_exactly_at_threshold(self):
        # 2 inputs, 1 matched → confidence = 0.5 (exactly at threshold — passes)
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "a",
                    "inputs": {"repo": "org/r", "lit": "literal"},
                    "outputs": {"out": "str"},
                }
            ]
        }
        context = {"repo": "org/r"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["extraction_confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Duplicate context key values
# ---------------------------------------------------------------------------

class TestDuplicateContextKeyValues:
    def test_prefers_key_that_appears_in_step_inputs(self):
        # Both "repo" and "project" hold the same value.
        # Input key name is "repo" — so "repo" should win.
        context = {"repo": "shared-value", "project": "shared-value"}
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "fetch",
                    "inputs": {"repo": "shared-value"},
                    "outputs": {"data": "str"},
                }
            ]
        }
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["template"]["steps"][0]["inputs"]["repo"] == "{repo}"
        slot_names = [s["name"] for s in result["template"]["slots"]]
        assert slot_names == ["repo"]

    def test_alphabetical_fallback_when_no_input_key_match(self):
        # Both "project" and "repo" hold the same value.
        # Input key is "data" — neither "project" nor "repo" matches "data".
        # Alphabetical order: "project" < "repo" → choose "project".
        context = {"project": "shared-value", "repo": "shared-value"}
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "fetch",
                    "inputs": {"data": "shared-value"},
                    "outputs": {"out": "str"},
                }
            ]
        }
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["template"]["steps"][0]["inputs"]["data"] == "{project}"
        slot_names = [s["name"] for s in result["template"]["slots"]]
        assert slot_names == ["project"]


# ---------------------------------------------------------------------------
# Non-string inputs (must be kept as-is silently)
# ---------------------------------------------------------------------------

class TestNonStringInputs:
    def test_non_string_inputs_preserved(self):
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "process",
                    "inputs": {
                        "count": 42,        # int — no abstraction
                        "flag": True,       # bool — no abstraction
                        "repo": "org/r",    # str — can be abstracted
                    },
                    "outputs": {"out": "str"},
                }
            ]
        }
        context = {"repo": "org/r"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        # 3 total inputs, 1 str match → confidence = 1/3 < 0.5 → low_confidence
        # The non-string inputs count toward total_step_inputs but don't become slots.
        assert result["ok"] is False
        assert result["refusal_reason"] == "low_confidence"
        assert result["extraction_confidence"] == pytest.approx(1 / 3)

    def test_non_string_inputs_contribute_to_total(self):
        # 2 str inputs (1 matched) + 0 non-str = 2 total, 1 slot → confidence 0.5
        trace = {
            "steps": [
                {
                    "step_id": "1",
                    "op": "process",
                    "inputs": {"repo": "org/r", "lit": "hardcoded"},
                    "outputs": {"out": "str"},
                }
            ]
        }
        context = {"repo": "org/r"}
        result = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        assert result["ok"] is True
        assert result["extraction_confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# planner_output is ignored (v0.1)
# ---------------------------------------------------------------------------

class TestPlannerOutputIgnored:
    def test_planner_output_none_does_not_affect_result(self):
        context = {"repo": "org/r"}
        trace = {
            "steps": [
                {"step_id": "1", "op": "fetch", "inputs": {"repo": "org/r"},
                 "outputs": {"data": "str"}}
            ]
        }
        r1 = extract_plan_template(
            task="t", context=context,
            execution_trace=trace, planner_output=None,
            canonicalization_result=_canon(),
        )
        r2 = extract_plan_template(
            task="t", context=context,
            execution_trace=trace,
            planner_output={"trace": [], "steps": [], "cost": 0.03},
            canonicalization_result=_canon(),
        )
        assert r1["ok"] == r2["ok"]
        assert r1["extraction_confidence"] == r2["extraction_confidence"]
        assert r1["refusal_reason"] == r2["refusal_reason"]

    def test_plan_id_is_unique_across_calls(self):
        context = {"repo": "org/r"}
        trace = {
            "steps": [
                {"step_id": "1", "op": "fetch", "inputs": {"repo": "org/r"},
                 "outputs": {"data": "str"}}
            ]
        }
        r1 = extract_plan_template(
            task="t", context=context, execution_trace=trace,
            planner_output=None, canonicalization_result=_canon(),
        )
        r2 = extract_plan_template(
            task="t", context=context, execution_trace=trace,
            planner_output=None, canonicalization_result=_canon(),
        )
        assert r1["template"]["plan_id"] != r2["template"]["plan_id"]
