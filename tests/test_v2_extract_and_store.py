"""
Tests for scripts/extract_and_store.py.

No real network — ThriftLMPlanCache.store is mocked throughout.
All inputs are plain dict literals.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow importing the script as a module from the scripts/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract_and_store import extract_and_store  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _canon() -> dict:
    return {
        "intent_key": {
            "action": "summarize",
            "target": "pull_requests",
            "goal": "identify_blockers",
            "time_scope": None,
            "tool_family": "github",
        },
        "intent_bucket_hash": "abc123deadbeef12",
        "confidence": 0.92,
        "canonicalizer_version": "v0.4",
        "raw_canonical_text": "Summarize pull requests to identify blockers",
        "from_cache": False,
    }


def _good_trace() -> dict:
    return {
        "steps": [
            {
                "step_id": "1",
                "op": "fetch_prs",
                "tool_family": "github",
                "inputs": {"repo": "org/myrepo"},
                "outputs": {"prs": "list[pr]"},
            },
            {
                "step_id": "2",
                "op": "produce_summary",
                "inputs": {"repo": "org/myrepo"},
                "outputs": {"summary": "string"},
            },
        ]
    }


def _bad_trace() -> dict:
    # Missing "steps" key → is_extractable_trace returns (False, "missing_steps")
    return {"ops": []}


_CONTEXT = {"repo": "org/myrepo"}
_API_KEY = "tlm_test"
_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helper: call extract_and_store with store mocked
# ---------------------------------------------------------------------------

def _run(
    trace: dict = None,
    context: dict = None,
    store_return: dict = None,
    store_raises: Exception = None,
    planner_output: dict | None = None,
) -> tuple[dict, MagicMock]:
    """
    Run extract_and_store with ThriftLMPlanCache.store mocked.
    Returns (result, mock_store_method).
    """
    if trace is None:
        trace = _good_trace()
    if context is None:
        context = _CONTEXT
    if store_return is None and store_raises is None:
        store_return = {"plan_id": "plan-uuid-1234"}

    mock_store = MagicMock()
    if store_raises is not None:
        mock_store.side_effect = store_raises
    else:
        mock_store.return_value = store_return

    with patch("extract_and_store.ThriftLMPlanCache") as MockClient:
        MockClient.return_value.store = mock_store
        result = extract_and_store(
            task="summarize open PRs for org/myrepo",
            context=context,
            execution_trace=trace,
            canonicalization_result=_canon(),
            api_key=_API_KEY,
            base_url=_BASE_URL,
            planner_output=planner_output,
        )

    return result, mock_store


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_status_stored(self):
        result, _ = _run()
        assert result["status"] == "stored"

    def test_plan_id_returned(self):
        result, _ = _run(store_return={"plan_id": "plan-abc"})
        assert result["plan_id"] == "plan-abc"

    def test_extraction_confidence_positive(self):
        result, _ = _run()
        assert result["extraction_confidence"] > 0.0

    def test_refusal_reason_none_on_success(self):
        result, _ = _run()
        assert result["refusal_reason"] is None

    def test_error_none_on_success(self):
        result, _ = _run()
        assert result["error"] is None

    def test_generalization_notes_populated(self):
        result, _ = _run()
        assert result["generalization_notes"] is not None

    def test_store_called_once(self):
        _, mock_store = _run()
        mock_store.assert_called_once()

    def test_client_constructed_with_correct_args(self):
        with patch("extract_and_store.ThriftLMPlanCache") as MockClient:
            MockClient.return_value.store = MagicMock(return_value={"plan_id": "x"})
            extract_and_store(
                task="t", context=_CONTEXT,
                execution_trace=_good_trace(),
                canonicalization_result=_canon(),
                api_key="tlm_key", base_url="http://srv", timeout=3.0,
            )
        MockClient.assert_called_once_with(
            api_key="tlm_key", base_url="http://srv", timeout=3.0
        )


# ---------------------------------------------------------------------------
# Extraction refused — store must never be called
# ---------------------------------------------------------------------------

class TestExtractionRefused:
    def test_status_refused_on_bad_trace(self):
        result, mock_store = _run(trace=_bad_trace())
        assert result["status"] == "refused"

    def test_refusal_reason_set(self):
        result, _ = _run(trace=_bad_trace())
        assert result["refusal_reason"] == "missing_steps"

    def test_store_not_called_on_refusal(self):
        _, mock_store = _run(trace=_bad_trace())
        mock_store.assert_not_called()

    def test_plan_id_none_on_refusal(self):
        result, _ = _run(trace=_bad_trace())
        assert result["plan_id"] is None

    def test_error_none_on_refusal(self):
        result, _ = _run(trace=_bad_trace())
        assert result["error"] is None

    def test_confidence_zero_on_bad_trace(self):
        result, _ = _run(trace=_bad_trace())
        assert result["extraction_confidence"] == 0.0

    def test_refused_on_empty_steps(self):
        result, mock_store = _run(trace={"steps": []})
        assert result["status"] == "refused"
        assert result["refusal_reason"] == "empty_steps"
        mock_store.assert_not_called()

    def test_refused_on_low_confidence(self):
        # 4 inputs, no context matches → confidence = 0.0 → low_confidence
        trace = {
            "steps": [
                {"step_id": "1", "op": "a",
                 "inputs": {"x": "lit1", "y": "lit2", "z": "lit3", "w": "lit4"},
                 "outputs": {"out": "str"}},
            ]
        }
        result, mock_store = _run(trace=trace, context={"unrelated_key": "different_value"})
        assert result["status"] == "refused"
        assert result["refusal_reason"] == "low_confidence"
        mock_store.assert_not_called()


# ---------------------------------------------------------------------------
# Store fails — extraction succeeded but server returned an error
# ---------------------------------------------------------------------------

class TestStoreFailed:
    def test_status_store_failed(self):
        result, _ = _run(store_raises=RuntimeError("store timed out"))
        assert result["status"] == "store_failed"

    def test_error_message_propagated(self):
        result, _ = _run(store_raises=RuntimeError("store timed out"))
        assert "store timed out" in result["error"]

    def test_plan_id_none_on_store_failure(self):
        result, _ = _run(store_raises=RuntimeError("500"))
        assert result["plan_id"] is None

    def test_refusal_reason_none_on_store_failure(self):
        result, _ = _run(store_raises=RuntimeError("500"))
        assert result["refusal_reason"] is None

    def test_confidence_still_populated_on_store_failure(self):
        result, _ = _run(store_raises=RuntimeError("500"))
        assert result["extraction_confidence"] > 0.0

    def test_generalization_notes_still_populated_on_store_failure(self):
        result, _ = _run(store_raises=RuntimeError("500"))
        assert result["generalization_notes"] is not None


# ---------------------------------------------------------------------------
# structural_signature attached before store
# ---------------------------------------------------------------------------

class TestStructuralSignatureAttached:
    def test_structural_signature_in_plan_passed_to_store(self):
        _, mock_store = _run()
        call_args = mock_store.call_args
        plan_dict = call_args[0][0]  # first positional arg to store()
        assert "structural_signature" in plan_dict

    def test_structural_signature_has_required_context_keys(self):
        _, mock_store = _run()
        plan_dict = mock_store.call_args[0][0]
        sig = plan_dict["structural_signature"]
        assert "required_context_keys" in sig
        assert "repo" in sig["required_context_keys"]

    def test_structural_signature_has_tool_families(self):
        _, mock_store = _run()
        plan_dict = mock_store.call_args[0][0]
        sig = plan_dict["structural_signature"]
        assert "tool_families" in sig
        assert "github" in sig["tool_families"]

    def test_structural_signature_has_step_count(self):
        _, mock_store = _run()
        plan_dict = mock_store.call_args[0][0]
        sig = plan_dict["structural_signature"]
        assert sig["step_count"] == 2

    def test_only_one_extra_field_added(self):
        """structural_signature is the only key added beyond the template fields."""
        from thriftlm.v2.extractor import extract_plan_template

        extraction = extract_plan_template(
            task="t", context=_CONTEXT,
            execution_trace=_good_trace(), planner_output=None,
            canonicalization_result=_canon(),
        )
        template_keys = set(extraction["template"].keys())

        _, mock_store = _run()
        plan_dict = mock_store.call_args[0][0]
        store_keys = set(plan_dict.keys())

        assert store_keys - template_keys == {"structural_signature"}


# ---------------------------------------------------------------------------
# planner_output=None does not affect result
# ---------------------------------------------------------------------------

class TestPlannerOutputIgnored:
    def test_none_and_non_none_produce_same_status(self):
        result_none, _ = _run(planner_output=None)
        result_with, _ = _run(planner_output={"trace": [], "cost": 0.02})
        assert result_none["status"] == result_with["status"]

    def test_none_and_non_none_produce_same_confidence(self):
        result_none, _ = _run(planner_output=None)
        result_with, _ = _run(planner_output={"trace": [], "cost": 0.02})
        assert result_none["extraction_confidence"] == result_with["extraction_confidence"]
