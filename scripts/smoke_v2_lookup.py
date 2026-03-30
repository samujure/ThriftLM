"""
Quick smoke test for the V2 lookup path.

Usage:
    python scripts/smoke_v2_lookup.py \\
        --api-key tlm_xxx \\
        --base-url http://localhost:8000 \\
        --task "summarize open PRs for org/myrepo" \\
        --context '{"repo": "org/myrepo"}' \\
        --runtime-caps '{"tool_families": ["github"], "allow_side_effects": false}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thriftlm.v2.adapters.generic import ThriftLMPlanCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V2 lookup endpoint.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", default="{}", help="JSON string")
    parser.add_argument("--runtime-caps", default="{}", help="JSON string")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    try:
        context = json.loads(args.context)
        runtime_caps = json.loads(args.runtime_caps)
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: invalid JSON argument — {exc}")

    cache = ThriftLMPlanCache(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    try:
        result = cache.lookup(task=args.task, context=context, runtime_caps=runtime_caps)
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
