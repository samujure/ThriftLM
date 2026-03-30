"""
Phase 2 integration helper: extract a reusable plan template from a successful
execution trace and store it via the V2 server.

Usage:
    python scripts/extract_and_store.py \
        --api-key tlm_test \
        --base-url http://localhost:8000 \
        --task "summarize open PRs for org/myrepo" \
        --context '{"repo": "org/myrepo"}' \
        --trace path/to/trace.json \
        --canon path/to/canon.json

Optional:
    --planner-output path/to/planner_output.json
    --timeout 5.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from thriftlm.v2.extractor import build_structural_signature, extract_plan_template
from thriftlm.v2.adapters.generic import ThriftLMPlanCache


def extract_and_store(
    task: str,
    context: dict,
    execution_trace: dict,
    canonicalization_result: dict,
    api_key: str,
    base_url: str,
    planner_output: dict | None = None,
    timeout: float = 5.0,
) -> dict:
    """
    Extract a reusable plan template from a successful execution trace
    and store it via the V2 server.

    Returns a result dict:
    {
        "status": "stored" | "refused" | "store_failed",
        "plan_id": str | None,
        "extraction_confidence": float,
        "refusal_reason": str | None,
        "generalization_notes": str | None,
        "error": str | None,
    }
    """
    # ------------------------------------------------------------------
    # Step 1 — extract
    # ------------------------------------------------------------------
    result = extract_plan_template(
        task, context, execution_trace, planner_output, canonicalization_result
    )

    if not result["ok"]:
        return {
            "status": "refused",
            "plan_id": None,
            "extraction_confidence": result["extraction_confidence"],
            "refusal_reason": result["refusal_reason"],
            "generalization_notes": result["generalization_notes"],
            "error": None,
        }

    # ------------------------------------------------------------------
    # Step 2 — build structural_signature
    # ------------------------------------------------------------------
    template = result["template"]
    sig = build_structural_signature(
        template["steps"],
        [s["name"] for s in template["slots"]],
        canonicalization_result,
    )

    # ------------------------------------------------------------------
    # Step 3 — attach structural_signature to plan dict before store
    # ------------------------------------------------------------------
    plan_dict = dict(template)
    plan_dict["structural_signature"] = sig

    # ------------------------------------------------------------------
    # Step 4 — store via ThriftLMPlanCache
    # ------------------------------------------------------------------
    client = ThriftLMPlanCache(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        store_response = client.store(plan_dict)
        plan_id = store_response.get("plan_id")
        return {
            "status": "stored",
            "plan_id": plan_id,
            "extraction_confidence": result["extraction_confidence"],
            "refusal_reason": None,
            "generalization_notes": result["generalization_notes"],
            "error": None,
        }
    except RuntimeError as e:
        return {
            "status": "store_failed",
            "plan_id": None,
            "extraction_confidence": result["extraction_confidence"],
            "refusal_reason": None,
            "generalization_notes": result["generalization_notes"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _load_json_file(path: str, label: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to load {label} from {path!r}: {e}"}))
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a plan template from a trace and store it via ThriftLM V2."
    )
    parser.add_argument("--api-key", required=True, help="Developer API key (tlm_xxx)")
    parser.add_argument("--base-url", required=True, help="V2 server base URL")
    parser.add_argument("--task", required=True, help="Task string")
    parser.add_argument("--context", default="{}", help="Context as JSON string")
    parser.add_argument("--trace", required=True, help="Path to execution_trace JSON file")
    parser.add_argument("--canon", required=True, help="Path to canonicalization_result JSON file")
    parser.add_argument("--planner-output", default=None, help="Path to planner_output JSON file (optional)")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    try:
        context = json.loads(args.context)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"failed to parse --context JSON: {e}"}))
        sys.exit(1)

    execution_trace = _load_json_file(args.trace, "--trace")
    canonicalization_result = _load_json_file(args.canon, "--canon")

    planner_output = None
    if args.planner_output:
        planner_output = _load_json_file(args.planner_output, "--planner-output")

    outcome = extract_and_store(
        task=args.task,
        context=context,
        execution_trace=execution_trace,
        canonicalization_result=canonicalization_result,
        api_key=args.api_key,
        base_url=args.base_url,
        planner_output=planner_output,
        timeout=args.timeout,
    )
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
