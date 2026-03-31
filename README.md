<div align="center">

# ThriftLM

**Stop paying for the same LLM call twice just because users phrased it differently.**

`0.2.0` adds plan caching on top of response caching — repeated agent workflows now skip the planner entirely.

[![PyPI version](https://badge.fury.io/py/thriftlm.svg)](https://pypi.org/project/thriftlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-thriftlm-pink.svg)](https://samujure.github.io/ThriftLM/)

```bash
pip install thriftlm
```

</div>

---

## What ThriftLM is

ThriftLM is a two-layer caching system for LLM applications.

**V1 — response cache** (`thriftlm==0.1.x`, stable)
Same query → same answer. Intercepts repeated or semantically similar LLM calls before they reach your provider. Three-tier stack: Redis exact hash → local numpy cosine index → Supabase pgvector HNSW.

**V2 — plan cache** (`thriftlm==0.2.0`, new)
Same job → same execution plan, filled with fresh context. Intercepts agent tasks before planning. If a semantically similar task was planned before, V2 returns a validated, slot-filled `FilledPlan` — no planner call, no LLM call. If it misses, your planner runs and the result can be stored for next time.

**Both layers can run together.** V1 sits underneath V2 to cache repeated leaf LLM calls inside agent steps.

---

## What's new in 0.2.0

- **Plan-level cache** (`thriftlm.v2`) — reuse reasoning skeletons across task families, not just identical queries
- **Intent canonicalization** — tasks are routed to deterministic buckets via a structured `IntentKey` (gpt-4o-mini, cached 1h in Redis)
- **Composite candidate reranking** — `0.7 × semantic_similarity + 0.3 × structural_score` over fetched candidates
- **Slot filling + 7-stage validation** — plans are filled with caller context and validated before being returned; bad fills are silently discarded
- **Automatic plan extraction** — after a planner runs on a miss, `extract_plan_template()` generalizes the trace into a reusable template (deterministic, no LLM)
- **`scripts/`** — seed, smoke-test, and extract-and-store helpers for developer workflow
- **364 tests passing**

---

## Architecture

### V1 — response cache

```
query
  │
  ▼
┌─────────────────┐   HIT → return (~0.5ms)
│  Redis          │   exact embedding hash
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐   HIT → Supabase PK fetch → return (~50ms)
│  Local Numpy    │   cosine similarity matmul
│  Index          │
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐
│  Your LLM fn    │   llm_fn() called here
└────────┬────────┘
         │
         ▼
   PII scrub (Presidio, responses only) → store → return
```

### V2 — plan cache

**Lookup path:**
```
task + context + runtime_caps
  │
  ▼
canonicalize(task)          → IntentKey + intent_bucket_hash
  └── Redis 1h TTL          (no second OpenAI call on repeat tasks)
  │
  ▼
bucket fetch (Supabase)     → candidates matching intent_bucket_hash
  │
  ▼
composite rerank            → 0.7 × sem_sim + 0.3 × structural_score
  │
  ▼
adapt_plan()                → fill SlotSpecs from context + transforms
  │
  ▼
validate_plan()             → 7-stage pipeline; discard + try next on fail
  │
  ├── HIT  → return FilledPlan (planner never ran)
  └── MISS → return miss signal → caller runs planner
```

**Miss → extract → store path:**
```
caller planner runs → execution trace
  │
  ▼
extract_plan_template()     → generalize trace to PlanTemplate (deterministic)
  │
  ▼
POST /v2/plan/store         → server verifies bucket hash → stores in Supabase
  │
  ▼
next similar task hits the plan cache
```

---

## Installation

```bash
pip install thriftlm
```

**Prerequisites:**
- Python 3.10+
- Supabase project with pgvector (`supabase/setup.sql` to provision tables)
- Redis (local or [Upstash](https://upstash.com))
- `OPENAI_API_KEY` for V2 canonicalization (gpt-4o-mini)

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
```

---

## V1 Quickstart — response cache

```python
from thriftlm import SemanticCache
import openai

cache = SemanticCache(threshold=0.85, api_key="your-key")

def call_llm(query: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    )
    return resp.choices[0].message.content

# Cache check + LLM fallback in one call
response = cache.get_or_call("Explain semantic caching", call_llm)

# Near-duplicate → instant cache hit, no LLM called
response2 = cache.get_or_call("What is semantic caching?", call_llm)
```

That's the entire integration. No architecture changes — wrap the existing LLM call.

---

## V2 Quickstart — plan cache

Start the V2 server:

```bash
py -m uvicorn thriftlm.v2._server:app --port 8000
```

Use `ThriftLMPlanCache` in your agent:

```python
from thriftlm.v2.adapters.generic import ThriftLMPlanCache

cache = ThriftLMPlanCache(
    api_key="tlm_xxx",
    base_url="http://localhost:8000",
    timeout=30.0,   # first call does OpenAI canonicalization, allow extra time
)

task = "summarize open PRs for org/myrepo"
context = {"repo": "org/myrepo"}
runtime_caps = {"tool_families": ["github"], "allow_side_effects": False}

result = cache.lookup(task=task, context=context, runtime_caps=runtime_caps)

if result["status"] == "hit":
    # Planner skipped entirely — use the validated, slot-filled plan
    filled_plan = result["filled_plan"]
    executor.run(filled_plan, context)

else:
    # Cache miss — run your planner, then store the trace for next time
    planner_output = my_planner(task, context)
    executor.run(planner_output, context)

    # Optional: extract and store for future reuse
    from scripts.extract_and_store import extract_and_store
    extract_and_store(
        task=task,
        context=context,
        execution_trace=planner_output["trace"],
        canonicalization_result=result.get("canonicalization_result"),
        api_key="tlm_xxx",
        base_url="http://localhost:8000",
    )
```

On the second call with a semantically similar task (same intent, different `repo`), V2 returns a hit — slots are filled with the new context, validation passes, planner never runs.

---

## Core V2 concepts

**IntentKey** — structured decomposition of a task into `action`, `target`, `goal`, `time_scope`, and optional metadata fields (`domain`, `format`, `audience`, `tool_family`). Produced by the canonicalizer (gpt-4o-mini at temperature=0).

**intent_bucket_hash** — 16-char SHA-256 of the 4 core fields only (`action`, `target`, `goal`, `time_scope`). Optional fields are excluded from the hash intentionally: LLMs vary them across invocations even for the same task. The hash is the routing key for plan lookup.

**PlanTemplate** — a stored execution skeleton: ordered steps with typed I/O, `SlotSpec` declarations for caller-supplied values, output schema, and version metadata. Retrieved from Supabase by bucket hash.

**FilledPlan** — a `PlanTemplate` with all `SlotSpec` values resolved from the caller's `context`. Step inputs referencing `{slot_name}` are substituted. Prior-step output references (`{prs}`, `{grouped}`) are left for the executor.

**Structural scoring** — composite score used to rank candidates within a bucket:
```
final_score = 0.7 × semantic_similarity + 0.3 × structural_score

structural_score =
    0.35 × slot_overlap      (required context keys present)
  + 0.25 × tool_family_match (plan needs tools the runtime has)
  + 0.20 × format_audience   (format/audience fields match)
  + 0.20 × side_effect_compat (side-effecting steps allowed?)
```

**Validation — 7 ordered stages:**

| Stage | What it checks |
|---|---|
| 1 | All required slots resolved from context |
| 2 | Slot values match declared type hints |
| 3 | All step inputs satisfied by prior outputs or slots |
| 4 | Required `tool_family` values present in `runtime_caps` |
| 5 | No unsubstituted `{placeholder}` strings remain |
| 6 | Every non-optional output schema field has a producing step |
| 7 | Side-effecting steps permitted by `runtime_caps.allow_side_effects` |

A candidate that fails any stage is discarded silently. The next ranked candidate is tried. If all `top_k` candidates fail, V2 returns a miss.

---

## Safety and invariants

- **V2 never executes plans.** It returns a validated `FilledPlan`. The caller owns execution.
- **Bucket hash is recomputed server-side on store.** Caller-supplied `intent_bucket_hash` is not trusted — a mismatch returns `400 hash_mismatch`.
- **Plans are tenant-isolated.** Every plan is scoped to `api_key`. No cross-tenant reads or writes.
- **Extractor is deterministic.** `extract_plan_template()` calls no LLM and makes no network requests. It generalizes a trace using reverse context mapping. It will refuse extraction (return `ok=False`) if the trace has fewer than 2 steps, all steps are side-effecting with no slots extracted, or extraction confidence is below 0.5.
- **Canonicalization is cached.** Once a task string is canonicalized, the result is stored in Redis for 1 hour. The same task never triggers two OpenAI calls within that window.

---

## Current scope and limitations

- **Text-first.** V2 is designed for text-input agent tasks. Multimodal support (`EvidenceProfile`) is designed but not yet built.
- **Shallow slot extraction.** The extractor handles exact top-level context value → placeholder substitution. Nested placeholder extraction and fuzzy abstraction are not supported in v0.1.
- **No benchmark yet.** V2 hit rate and latency benchmarks across diverse task families are planned for Phase 3.
- **plan_threshold = 0.60.** SBERT cosine similarity between short task strings and plan descriptions typically lands in 0.50–0.70. The threshold may need tuning as your plan bank grows.
- **seed_task vs description split** is future polish. Currently `seed_v2_plans.py` canonicalizes the plan description string, which works but conflates routing vocabulary with reranking text. Not a blocker.

---

## Developer scripts

| Script | What it does |
|---|---|
| `scripts/seed_v2_plans.py --api-key tlm_xxx` | Seeds Supabase with canonical plan templates. Calls `canonicalize()` on each description to get live bucket hashes — no hardcoded intent keys. |
| `scripts/smoke_v2_lookup.py --api-key tlm_xxx --base-url http://localhost:8000 --task "..." --context '{}' --timeout 30` | Fires a single lookup and prints the full JSON response. Use `--timeout 30` on cold starts. |
| `scripts/extract_and_store.py --api-key tlm_xxx --base-url http://localhost:8000 --task "..." --context '{}' --trace trace.json --canon canon.json` | Extracts a `PlanTemplate` from an execution trace and stores it via `/v2/plan/store`. |
| `scripts/debug_v2_lookup.py --api-key tlm_xxx --bucket <hash> --task "..." --context '{}'` | Fetches raw DB rows for a bucket, scores them, and runs `adapt_plan` + `validate_plan` — useful for diagnosing misses. |

---

## V2 API endpoints

| Method + Path | Description |
|---|---|
| `POST /v2/plan/lookup` | Main entry: task + context → `FilledPlan` or miss |
| `POST /v2/plan/store` | Store a template (server recomputes and verifies bucket hash) |
| `GET /v2/plan/bucket/:hash` | List templates for a bucket |
| `DELETE /v2/plan/:id` | Evict a single plan |
| `DELETE /v2/plan/bucket/:hash` | Evict an entire bucket |
| `POST /v2/plan/invalidate-by-version` | Bulk soft-invalidate by version string |
| `GET /v2/metrics` | Server health + version |

---

## V1 metrics dashboard

```bash
thriftlm serve --api-key your-key
# → http://localhost:8000  (opens automatically)
```

Shows hit rate, tokens saved, estimated cost saved, and top cached queries. Reads directly from your Supabase.

---

## V1 benchmark

```
Threshold | Hit Rate | Hits / 200
----------|----------|------------
0.70      |  92.5%   |   185
0.75      |  86.0%   |   172
0.80      |  78.0%   |   156
0.82      |  73.5%   |   147   ← recommended
0.85      |  62.5%   |   125   (default)
0.90      |  40.0%   |    80

Model: all-MiniLM-L6-v2  ·  Dataset: Quora Question Pairs (200 pairs)
```

---

## Project structure

```
ThriftLM/
├── thriftlm/
│   ├── cache.py                 # V1 SemanticCache
│   ├── embedder.py              # SBERT all-MiniLM-L6-v2
│   ├── privacy.py               # Presidio PII scrubbing
│   ├── _server.py               # V1 FastAPI (thriftlm serve)
│   ├── cli.py                   # CLI entry point
│   ├── backends/
│   │   ├── local_index.py       # Numpy cosine index
│   │   ├── redis_backend.py     # Exact hash cache
│   │   └── supabase_backend.py  # pgvector HNSW store
│   └── v2/
│       ├── schemas.py           # TypedDicts: IntentKey, PlanTemplate, FilledPlan, …
│       ├── intent.py            # canonicalize() → IntentKey + bucket hash
│       ├── canonicalization_cache.py  # Redis cache for canonicalization results
│       ├── plan_cache.py        # bucket fetch + composite rerank
│       ├── adapter.py           # slot filling + TransformRegistry
│       ├── validator.py         # 7-stage validation pipeline
│       ├── extractor.py         # trace → PlanTemplate (deterministic)
│       ├── _server.py           # V2 FastAPI endpoints
│       └── adapters/
│           ├── base.py          # BasePlanCache ABC
│           └── generic.py       # ThriftLMPlanCache HTTP client
├── scripts/
│   ├── seed_v2_plans.py
│   ├── smoke_v2_lookup.py
│   ├── extract_and_store.py
│   └── debug_v2_lookup.py
├── tests/                       # 364 passing
├── supabase/setup.sql
├── api/                         # Multi-tenant self-hosted backend
└── pyproject.toml
```

---

## Roadmap

| Item | Status |
|---|---|
| V1 response cache | Shipped (`0.1.x`) |
| V2 plan cache | Shipped (`0.2.0`) |
| V2 benchmark (200 tasks, 5 intent buckets) | Phase 3 |
| Fly.io deploy + hosted endpoint | Phase 3 |
| Claude Code MCP adapter / Codex CLI hook | Roadmap |
| `seed_task` vs `description` split in seed script | Post-0.2.0 polish |
| V2.5 multimodal `EvidenceProfile` | Future |

---

## Development

```bash
git clone https://github.com/samujure/ThriftLM
cd ThriftLM
pip install -e ".[dev]"
cp .env.example .env      # fill in SUPABASE_URL, SUPABASE_KEY, REDIS_URL, OPENAI_API_KEY
docker compose up -d      # local Redis
pytest tests/ -q          # 364 tests
py scripts/seed_v2_plans.py --api-key tlm_test
py scripts/smoke_v2_lookup.py --api-key tlm_test --base-url http://localhost:8000 \
  --task "summarize open PRs for org/myrepo" --context '{"repo":"org/myrepo"}' \
  --runtime-caps '{"tool_families":["github"],"allow_side_effects":false}' --timeout 30
```

---

> This README reflects `thriftlm==0.2.0`.

---

<div align="center">
Built by Srivamsi Amujure & Ivan Thomas Shen
</div>
