"""
DEV TOOL — not part of the production path.

Debug script: trace the adapt+validate pipeline for a specific bucket hash.

Usage:
    python scripts/debug_v2_lookup.py --api-key tlm_xxx --bucket bb1fc6e10e2fde8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from thriftlm.v2.adapter import (
    SlotFillError, SlotTypeError, TransformNotFoundError, TransformExecutionError,
    adapt_plan,
)
from thriftlm.v2.plan_cache import PlanCache
from thriftlm.v2.validator import validate_plan


def _make_supabase():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set")
    from supabase import create_client
    return create_client(url, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--bucket", required=True, help="intent_bucket_hash")
    parser.add_argument("--task", default="summarize open PRs for org/myrepo")
    parser.add_argument("--context", default='{"repo": "org/myrepo"}')
    parser.add_argument("--runtime-caps",
                        default='{"tool_families": ["github"], "allow_side_effects": false}')
    args = parser.parse_args()

    context = json.loads(args.context)
    runtime_caps = json.loads(args.runtime_caps)

    sb = _make_supabase()

    # 1. Raw DB fetch — bypass PlanCache scoring to see what's stored
    print(f"\n=== Raw DB fetch for bucket={args.bucket!r} api_key={args.api_key!r} ===")
    rows = (
        sb.table("plans")
        .select("id, description, embedding, template_json, structural_signature, intent_bucket_hash")
        .eq("api_key", args.api_key)
        .eq("intent_bucket_hash", args.bucket)
        .eq("is_valid", True)
        .execute()
        .data
    )
    print(f"  Rows found: {len(rows)}")
    for i, row in enumerate(rows):
        print(f"\n  Row {i}: id={row['id']!r}  description={row['description']!r}")
        sig = row.get("structural_signature")
        print(f"          structural_signature={sig!r}")
        emb = row.get("embedding")
        if emb:
            emb_list = json.loads(emb) if isinstance(emb, str) else emb
            print(f"          embedding: {len(emb_list)} dims")
        else:
            print("          embedding: MISSING")
        tmpl = row.get("template_json")
        if isinstance(tmpl, str):
            tmpl = json.loads(tmpl)
        if tmpl:
            print(f"          plan_id={tmpl.get('plan_id')!r}")
            print(f"          slots={tmpl.get('slots')!r}")
            print(f"          steps={json.dumps(tmpl.get('steps'), indent=4)}")
            print(f"          output_schema={tmpl.get('output_schema')!r}")
        else:
            print("          template_json: MISSING/INVALID")

    if not rows:
        print("\nNo rows found — plan was never seeded under this bucket+api_key combo.")
        return

    # 2. PlanCache scoring
    print(f"\n=== PlanCache.get() scoring (task={args.task!r}) ===")
    plan_cache = PlanCache(supabase_client=sb, api_key=args.api_key, plan_threshold=0.0)  # 0.0 to see all
    candidates = plan_cache.get(
        intent_bucket_hash=args.bucket,
        task=args.task,
        context=context,
        runtime_caps=runtime_caps,
    )
    print(f"  Candidates returned (threshold=0.0): {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"\n  Candidate {i}: final_score={c['final_score']:.4f}  "
              f"sem={c['semantic_similarity']:.4f}  struct={c['structural_score']:.4f}")
        print(f"    plan_id={c['plan']['plan_id']!r}")

    # 3. adapt_plan + validate_plan for each candidate
    print(f"\n=== adapt_plan + validate_plan (context={context}, runtime_caps={runtime_caps}) ===")
    for i, c in enumerate(candidates):
        plan = c["plan"]
        print(f"\n  Candidate {i} (plan_id={plan.get('plan_id')!r}):")

        # adapt
        try:
            filled = adapt_plan(plan, context)
            print(f"    adapt_plan: OK  filled_slots={filled['filled_slots']!r}")
        except SlotFillError as e:
            print(f"    adapt_plan: FAIL SlotFillError: {e}")
            continue
        except SlotTypeError as e:
            print(f"    adapt_plan: FAIL SlotTypeError: {e}")
            continue
        except TransformNotFoundError as e:
            print(f"    adapt_plan: FAIL TransformNotFoundError: {e}")
            continue
        except TransformExecutionError as e:
            print(f"    adapt_plan: FAIL TransformExecutionError: {e}")
            continue

        # validate
        result = validate_plan(plan, filled, runtime_caps)
        if result["ok"]:
            print(f"    validate_plan: PASS — this should have been a HIT!")
        else:
            print(f"    validate_plan: FAIL stage={result['failed_stage']!r} reason={result['reason']!r}")
            print(f"      filled steps: {json.dumps(filled['steps'], indent=6)}")
            print(f"      plan output_schema: {plan.get('output_schema')!r}")
            print(f"      filled output_schema: {filled.get('output_schema')!r}")


if __name__ == "__main__":
    main()
