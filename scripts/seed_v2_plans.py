"""
Seed the Supabase `plans` table with the 3 canonical Phase 1 templates.

Usage:
    python scripts/seed_v2_plans.py --api-key tlm_xxx

Reads SUPABASE_URL and SUPABASE_KEY from the environment (or a .env file).
Skips any plan whose (api_key, intent_bucket_hash, description) already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from thriftlm.embedder import Embedder
from thriftlm.v2.intent import compute_bucket_hash
from thriftlm.v2.schemas import IntentKey, PlanTemplate

CANONICALIZER_VERSION = "v0.4"
VALIDATOR_VERSION = "v0.4"

# ---------------------------------------------------------------------------
# Seed plan definitions
# ---------------------------------------------------------------------------

_SEEDS: list[dict] = [
    {
        "intent_key": {
            "action": "summarize",
            "target": "pull_requests",
            "goal": "identify_blockers",
            "time_scope": None,
            "tool_family": "github",
        },
        "description": "Summarize open pull requests and identify blockers",
        "slots": [
            {
                "name": "repo", "source": "repo", "type_hint": "str",
                "required": True, "transform": None, "transform_args": None, "default": None,
            }
        ],
        "steps": [
            {
                "step_id": "1", "op": "fetch_prs", "tool_family": "github",
                "inputs": {"repo": "{repo}"}, "outputs": {"prs": "list"},
            },
            {
                "step_id": "2", "op": "summarize",
                "inputs": {"prs": "{prs}"}, "outputs": {"summary": "str"},
            },
        ],
        "output_schema": {"summary": "str"},
        "optional_outputs": [],
    },
    {
        "intent_key": {
            "action": "summarize",
            "target": "messages",
            "goal": "daily_digest",
            "time_scope": "today",
            "tool_family": "slack",
        },
        "description": "Summarize Slack channel messages into a daily digest",
        "slots": [
            {
                "name": "channel", "source": "channel", "type_hint": "str",
                "required": True, "transform": None, "transform_args": None, "default": None,
            }
        ],
        "steps": [
            {
                "step_id": "1", "op": "fetch_messages", "tool_family": "slack",
                "inputs": {"channel": "{channel}"}, "outputs": {"messages": "list"},
            },
            {
                "step_id": "2", "op": "summarize",
                "inputs": {"messages": "{messages}"}, "outputs": {"digest": "str"},
            },
        ],
        "output_schema": {"digest": "str"},
        "optional_outputs": [],
    },
    {
        "intent_key": {
            "action": "triage",
            "target": "issues",
            "goal": "prioritize_backlog",
            "time_scope": None,
            "tool_family": "github",
        },
        "description": "Triage and prioritize open GitHub issues by severity",
        "slots": [
            {
                "name": "repo", "source": "repo", "type_hint": "str",
                "required": True, "transform": None, "transform_args": None, "default": None,
            }
        ],
        "steps": [
            {
                "step_id": "1", "op": "fetch_issues", "tool_family": "github",
                "inputs": {"repo": "{repo}"}, "outputs": {"issues": "list"},
            },
            {
                "step_id": "2", "op": "triage",
                "inputs": {"issues": "{issues}"}, "outputs": {"prioritized": "list"},
            },
        ],
        "output_schema": {"prioritized": "list"},
        "optional_outputs": [],
    },
]


# ---------------------------------------------------------------------------
# structural_signature — mirrors _server.py exactly
# ---------------------------------------------------------------------------

def _build_structural_signature(plan: dict) -> dict:
    intent_key = plan["intent_key"]
    required_context_keys = sorted({
        slot["source"]
        for slot in plan.get("slots", [])
        if slot.get("required")
    })
    tool_families = sorted({
        step["tool_family"]
        for step in plan.get("steps", [])
        if step.get("tool_family")
    })
    has_side_effects = any(
        step.get("side_effect") is True
        for step in plan.get("steps", [])
    )
    return {
        "required_context_keys": required_context_keys,
        "tool_families": tool_families,
        "has_side_effects": has_side_effects,
        "format": intent_key.get("format"),
        "audience": intent_key.get("audience"),
        "step_count": len(plan.get("steps", [])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_supabase():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        sys.exit("Error: supabase package not installed. Run: pip install thriftlm[api]")


def seed(api_key: str) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    sb = _make_supabase()
    embedder = Embedder()
    created_at = datetime.now(timezone.utc).isoformat()

    for seed_def in _SEEDS:
        intent_key: IntentKey = seed_def["intent_key"]  # type: ignore[assignment]
        bucket_hash = compute_bucket_hash(intent_key)
        description = seed_def["description"]

        # Skip if already seeded (same tenant + bucket + description)
        existing = (
            sb.table("plans")
            .select("id")
            .eq("api_key", api_key)
            .eq("intent_bucket_hash", bucket_hash)
            .eq("description", description)
            .execute()
        )
        if existing.data:
            print(f"  SKIP  {description!r}  (bucket={bucket_hash})")
            continue

        # Build the full PlanTemplate
        import uuid
        plan_id = str(uuid.uuid4())
        plan: PlanTemplate = {
            "plan_id": plan_id,
            "intent_key": intent_key,
            "intent_bucket_hash": bucket_hash,
            "description": description,
            "steps": seed_def["steps"],
            "slots": seed_def["slots"],
            "output_schema": seed_def["output_schema"],
            "optional_outputs": seed_def["optional_outputs"],
            "plan_version": "1",
            "canonicalizer_version": CANONICALIZER_VERSION,
            "extractor_version": "seed",
            "validator_version": VALIDATOR_VERSION,
            "created_at": created_at,
        }

        sig = _build_structural_signature(plan)
        embedding = embedder.embed(description)

        result = (
            sb.table("plans")
            .insert({
                "api_key": api_key,
                "intent_key_json": intent_key,
                "intent_bucket_hash": bucket_hash,
                "description": description,
                "embedding": embedding,
                "template_json": dict(plan),
                "output_schema_json": plan["output_schema"],
                "structural_signature": sig,
                "plan_version": plan["plan_version"],
                "canonicalizer_version": plan["canonicalizer_version"],
                "extractor_version": plan["extractor_version"],
                "validator_version": plan["validator_version"],
                "is_valid": True,
            })
            .execute()
        )
        inserted_id = result.data[0]["id"] if result.data else plan_id
        print(f"  INSERT  {description!r}")
        print(f"          plan_id={inserted_id}  bucket={bucket_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ThriftLM V2 plan templates.")
    parser.add_argument("--api-key", required=True, help="Developer API key (tlm_xxx)")
    args = parser.parse_args()

    print(f"Seeding plans for api_key={args.api_key!r} ...\n")
    seed(args.api_key)
    print("\nDone.")


if __name__ == "__main__":
    main()
