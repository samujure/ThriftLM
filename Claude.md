# ThriftLM

## Credits
Ideated by Ivan Thomas Shen and Srivamsi Amujure.


---

## What This Is
A semantic caching layer for LLM applications. Instead of calling the LLM every time, ThriftLM embeds the user query using SBERT, checks a vector store for semantically similar past queries, and returns the cached response if similarity is above threshold. If no match, it calls the LLM, caches the result, and returns it.

**One sentence:** Stop paying for the same LLM call twice just because users phrased it differently.

---

## Target User
Developers building LLM-powered apps — customer support bots, RAG pipelines, LangGraph agents, agentic workflows. Anyone paying per token who has users asking semantically similar questions repeatedly.

---

## Core Flow
```
query comes in
  → embed query (SBERT, all-MiniLM-L6-v2, runs in library)
  → send embedding + api_key to hosted FastAPI backend
  → backend checks Supabase pgvector for similar embeddings (cosine sim > 0.85)
      ├── HIT  → return cached response instantly, no LLM call
      └── MISS → return null to library
  → library gets null → calls dev's LLM function → gets response
  → library sends (embedding + response) to backend to store
  → return response to dev's app
```

---

## How a Dev Uses It (3 lines)
```python
from thriftlm import SemanticCache

cache = SemanticCache(api_key="sc_xxx")
response = cache.get_or_call(query, llm_fn=my_langchain_chain.invoke)
```

That's it. No architecture changes. Just wrap the existing LLM call.

---

## Two Deployment Modes
1. **Hosted** — dev gets an API key, points at our cloud instance. Zero infra to manage.
2. **Self-hosted** — dev runs Docker Compose locally. Same codebase, just `THRIFTLM_URL=localhost`.

One env variable is the entire difference.

---

## V1 Scope (3 weeks, then done)
- [ ] pip-installable Python library (`thriftlm/`)
- [ ] FastAPI backend with API key auth
- [ ] Supabase (pgvector) for vector storage, isolated namespace per API key
- [ ] Upstash Redis for fast cache layer
- [ ] Simple dashboard — hit rate, tokens saved, cost saved per API key
- [ ] LangGraph native integration
- [ ] Docker Compose for self-hosting
- [ ] Published to PyPI
- [ ] MIT licensed, fully open source



---

## Stack
| Layer | Tool |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector storage | Supabase (pgvector) |
| Cache layer | Upstash Redis |
| API | FastAPI |
| Auth | API key (per dev, isolated namespace) |
| Hosting | Railway or Render |
| Packaging | pyproject.toml → PyPI |
| Self-host | Docker Compose |

---

## Repo Structure
```
ThriftLM/
├── thriftlm/                    # pip package
│   ├── __init__.py
│   ├── cache.py                 # SemanticCache class — core logic
│   ├── embedder.py              # SBERT wrapper (all-MiniLM-L6-v2)
│   ├── privacy.py               # PIIScrubber for PII detection/redaction
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── supabase_backend.py  # pgvector via Supabase
│   │   ├── redis_backend.py     # Upstash Redis
│   │   └── local_index.py       # in-process numpy similarity index
│   ├── integrations/
│   │   ├── __init__.py
│   │   └── langgraph.py         # LangGraph native integration
│   └── config.py                # threshold, TTL, env vars
├── api/                         # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── cache.py             # /lookup, /store endpoints
│   │   ├── metrics.py           # /metrics endpoint for dashboard
│   │   └── keys.py              # /keys endpoint
│   ├── auth.py                  # API key validation
│   └── db.py                    # Supabase client
├── dashboard/                   # Simple HTML/JS dashboard
│   └── index.html               # hit rate, tokens saved, cost saved
├── tests/
│   ├── test_cache.py
│   ├── test_embedder.py
│   ├── test_api.py
│   ├── test_privacy.py
│   ├── test_redis_backend.py
│   └── test_supabase_backend.py
├── supabase/
│   └── setup.sql                # pgvector schema + RPC functions
├── docker-compose.yml           # self-host: Redis + API (Supabase is external)
├── pyproject.toml               # pip packaging
├── .env.example                 # SUPABASE_URL, SUPABASE_KEY, REDIS_URL, etc.
├── CLAUDE.md                    # this file
└── README.md
```

---

## Key Classes

### SemanticCache (thriftlm/cache.py)
```python
class SemanticCache:
    def __init__(self, api_key: str, threshold: float = 0.85, ttl: int = 86400)
    def get_or_call(self, query: str, llm_fn: callable) -> str
    def lookup(self, query: str) -> str | None
    def store(self, query: str, response: str) -> None
```

### Embedder (thriftlm/embedder.py)
```python
class Embedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2")
    def embed(self, text: str) -> list[float]
```

---

## API Endpoints
```
POST /lookup        body: {embedding, api_key}      → {response} or null
POST /store         body: {embedding, query, response, api_key} → 200
GET  /metrics       header: api_key                 → {hit_rate, tokens_saved, cost_saved, total_queries}
POST /keys          body: {email}                   → {api_key}  (generate new key)
```

---

## Environment Variables
```
SUPABASE_URL=
SUPABASE_KEY=
REDIS_URL=
SIMILARITY_THRESHOLD=0.85
CACHE_TTL_SECONDS=86400
```

---

## Supabase Schema
```sql
-- One table, namespaced by api_key
CREATE TABLE cache_entries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    api_key TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    embedding VECTOR(384),        -- all-MiniLM-L6-v2 dims
    created_at TIMESTAMP DEFAULT NOW(),
    last_hit_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);

CREATE INDEX ON cache_entries
USING ivfflat (embedding vector_cosine_ops);

-- Metrics table
CREATE TABLE api_keys (
    api_key TEXT PRIMARY KEY,
    email TEXT,
    total_queries INTEGER DEFAULT 0,
    total_hits INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## What NOT To Build
- Custom vector database
- Multiple embedding model options (just all-MiniLM-L6-v2 for now)
- User accounts / billing / paid tiers
- V2 cross-app shared pool
- Anything not in the V1 scope checklist above

---

---

## V2 — architecture (spec v0.4, implementation-ready)

### Project overview

- **V1** (shipped, PyPI `thriftlm==0.1.6`): response-level semantic cache. Same question → same answer.
- **V2** (in progress): plan-level cache. Same job → same reasoning skeleton → fresh inputs → new answer.

Phase 1: COMPLETE — 297/297 tests passing, smoke test green (both tasks hit, from_cache=True on repeat)
Phase 2: ACTIVE — extractor.py is the current build target

### Repository layout (V2 additions)

```
ThriftLM/
├── thriftlm/
│   ├── backends/
│   │   ├── local_index.py       # V1: numpy matmul embedding index (~1ms)
│   │   ├── redis_backend.py     # V1: exact hash cache (~0.5ms)
│   │   └── supabase_backend.py  # V1: pgvector HNSW store
│   ├── static/
│   │   └── dashboard.html       # bundled dashboard for `thriftlm serve`
│   ├── v2/                      # ← BUILD TARGET (may not exist yet)
│   │   ├── schemas.py           # all shared dataclasses — build first
│   │   ├── intent.py            # canonicalize(task) → CanonicalizationResult
│   │   ├── canonicalization_cache.py  # Redis cache of prior canonicalizations
│   │   ├── plan_cache.py        # bucket fetch + composite rerank + top_k
│   │   ├── adapter.py           # fill SlotSpecs via TransformRegistry
│   │   ├── validator.py         # 7-stage ordered validation pipeline
│   │   ├── extractor.py         # Phase 2 active build target — trace → reusable PlanTemplate
│   │   ├── adapters/
│   │   │   ├── base.py          # BasePlanCache ABC
│   │   │   └── generic.py       # ThriftLMPlanCache REST wrapper
│   │   └── _server.py           # FastAPI V2 endpoints
│   ├── cli.py                   # Click CLI (`thriftlm serve`)
│   ├── _server.py               # V1 FastAPI server
│   ├── cache.py                 # V1 SemanticCache — do not touch
│   └── config.py
├── api/                         # multi-tenant FastAPI backend
├── dashboard/index.html         # landing page
├── docs/index.html              # GitHub Pages
├── demo/simulate.py             # demo script
├── tests/                       # 69/69 passing — keep green
├── pyproject.toml               # version 0.1.6
└── .env                         # SUPABASE_URL, SUPABASE_KEY, REDIS_URL, OPENAI_API_KEY
```

### V1 — do not modify

V1 is shipped and working. The pipeline is:

```
query → embed (SBERT all-MiniLM-L6-v2) → Redis exact hash (~0.5ms)
  ├── HIT  → return cached response
  └── MISS → LocalEmbeddingIndex (numpy matmul ~1ms)
        ├── HIT  → Supabase PK fetch → write Redis → return
        └── MISS → LLM call → Presidio PII scrub → store → return
```

V2 wraps V1. V1 sits underneath V2 to cache repeated leaf LLM calls. Do not change `cache.py` or `backends/`.

### The one-sentence description

V2 intercepts a task before planning, finds a reusable plan template, fills it with fresh context, validates it, and returns a `FilledPlan` to the caller. The caller executes. V2 never executes plans.

### Runtime flow

```
task arrives
   ↓
[1] canonicalization_cache   sha256(raw_task_text) lookup (Redis, 1h TTL)
    ├── HIT  → return prior CanonicalizationResult directly
    └── MISS → intent.py canonicalize → store result in cache → continue
   ↓  SLA: p50 <150ms, p95 <300ms, hard timeout 400ms (OpenAI call) → bypass V2 on failure
[2] plan_cache.py            bucket fetch → composite rerank → top_k=5 candidates
   ↓  miss → caller planner runs
   ↓  successful planner execution may be passed into extractor.py (Phase 2)
[3] adapter.py               fill typed SlotSpecs via TransformRegistry
   ↓  SlotFillError / SlotTypeError → try next candidate
[4] validator.py             7-stage ordered pipeline
   ↓  fail → next candidate → miss if all exhausted
return FilledPlan to caller
   ↓
caller executes plan
   ↓
[V1] SemanticCache           leaf LLM/tool call caching underneath
```

### Key invariant

**ThriftLM does not execute plans.** It returns a validated `FilledPlan`. The caller owns execution. This boundary is strict.

---

## Shared types — schemas.py

```python
from typing import TypedDict, Any

class IntentKey(TypedDict, total=False):
    # Core — always present
    action:      str
    target:      str
    goal:        str
    time_scope:  str | None
    # Optional — included when present in task
    domain:      str | None
    format:      str | None
    audience:    str | None
    constraints: list[str] | None   # sorted on normalization
    tool_family: str | None

class CanonicalizationResult(TypedDict):
    intent_key:            IntentKey
    intent_bucket_hash:    str       # deterministic hash — see §intent.py rules
    confidence:            float
    canonicalizer_version: str
    raw_canonical_text:    str
    from_cache:            bool

class SlotSpec(TypedDict):
    name:           str
    type_hint:      str
    required:       bool
    source:         str             # context dict key
    transform:      str | None      # registered transform name
    transform_args: dict | None     # typed args for transform (e.g. {"n": 10})
    default:        Any | None

class PlanStep(TypedDict, total=False):
    step_id:     str
    op:          str
    inputs:      dict[str, str]     # slot_name or prior step_id references
    outputs:     dict[str, str]     # named output keys → type hints
    side_effect: bool
    tool_family: str | None

class PlanTemplate(TypedDict):
    plan_id:               str
    intent_key:            IntentKey
    intent_bucket_hash:    str
    description:           str      # used for semantic reranking
    steps:                 list[PlanStep]
    slots:                 list[SlotSpec]
    output_schema:         dict[str, str]
    optional_outputs:      list[str]
    plan_version:          str
    canonicalizer_version: str
    extractor_version:     str
    validator_version:     str
    created_at:            str

class ScoredPlan(TypedDict):
    plan:                PlanTemplate
    semantic_similarity: float
    structural_score:    float
    final_score:         float

class FilledPlan(TypedDict):
    plan_id:            str
    intent_bucket_hash: str
    filled_slots:       dict[str, Any]
    steps:              list[PlanStep]
    output_schema:      dict[str, str]

class ValidationResult(TypedDict):
    ok:               bool
    failed_stage:     str | None    # '1' through '7', or None on pass
    reason:           str | None
    validator_version: str

class ExtractionResult(TypedDict):
    ok:                    bool
    template:              PlanTemplate | None
    extraction_confidence: float
    generalization_notes:  str | None
    refusal_reason:        str | None
```

---

## Phase 1 implementation decisions (permanent)

These were locked in during Phase 1 smoke testing. Do not reverse without a migration plan.

### 1. Bucket hash uses 4 core fields only — BREAKING

`compute_bucket_hash` hashes only `{action, target, goal, time_scope}`.
`domain`, `format`, `audience`, `constraints`, `tool_family` are **excluded from the hash**.

**Why:** OpenAI at `temperature=0` still varies optional fields across invocations (e.g. `domain="repository"` vs `domain="github"` for the same task). Including them caused the same task to hash to different buckets, making seeded plans unreachable. Optional fields remain on IntentKey for reranking/metadata.

**Consequence:** Plans seeded under old hashes are unreachable. Migration requires re-seeding or a bucket alias table.

**Test coverage:** `test_optional_fields_do_not_affect_hash`, `test_core_fields_all_affect_hash`, `test_two_keys_differing_only_in_optional_fields_share_bucket`.

### 2. `plan_threshold = 0.60`

Lowered from 0.78. SBERT all-MiniLM-L6-v2 cosine similarity between short task strings and plan descriptions typically lands in 0.50–0.70. 0.78 filtered correct plans. Raise back toward 0.70 if wrong plans start adapting successfully as the plan bank grows.

### 3. `_InMemoryCanonCache` is process-local

`_get_or_canonicalize()` checks in-memory → Redis → OpenAI. The in-memory layer guarantees that within one server process the same task resolves to the same bucket hash even when Redis is down or unconfigured. **Resets on server restart. Not shared across workers.** Redis is still required for production (multi-worker, multi-instance).

---

## intent.py — canonicalization rules

**Bucket hash normalization (must be deterministic):**
1. All string values: lowercase, strip whitespace
2. Only 4 core fields serialized: `action`, `target`, `goal`, `time_scope` — all optional fields excluded
3. `time_scope=None` excluded from serialization entirely
4. Field order: fixed alphabetical key sort

```python
_HASH_FIELDS = ("action", "target", "goal", "time_scope")

def compute_bucket_hash(intent_key: IntentKey) -> str:
    normalized = _normalize_intent_key(intent_key)
    serializable = {k: normalized[k] for k in _HASH_FIELDS if normalized.get(k) is not None}
    serialized = json.dumps(serializable, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
```

**Fallback policy:**

| Condition | Action |
|-----------|--------|
| confidence ≥ 0.85 + parse success | Continue to plan_cache |
| confidence < 0.85 | Bypass V2 → full planner |
| JSON parse failure | Bypass V2 → full planner |
| OpenAI call > 400ms | Bypass V2 → full planner |
| Any exception | Bypass V2 → full planner, log error |

Canonicalization cache key: `sha256(raw_task_text)` — raw bytes, no normalization before hashing.

---

## plan_cache.py — composite rerank score

```
final_score = 0.7 * semantic_similarity + 0.3 * structural_score

structural_score =
    0.35 * slot_overlap_score          # |required_context_keys ∩ context.keys()| / |required_context_keys|
  + 0.25 * tool_family_match_score     # 1.0 if plan.tool_family in runtime_caps.tool_families else 0.0
  + 0.20 * format_audience_match       # 1.0 match, 0.5 null, 0.0 mismatch
  + 0.20 * side_effect_compatibility   # 1.0 if no side effects OR allow_side_effects=true, else 0.0
```

**Edge case:** if `required_context_keys == []`, set `slot_overlap_score = 1.0` (avoid division by zero, don't penalize context-light plans).

`semantic_similarity`: cosine similarity between embedding of **raw incoming task text** and embedding of **stored plan description**. Uses V1 `LocalEmbeddingIndex`.

Default: `top_k = 5`, `plan_threshold = 0.60` (see Phase 1 implementation decisions above).

---

## adapter.py — transform registry

Built-in transforms and their typed args:

| Transform | Args | Behavior |
|-----------|------|----------|
| `filter_open` | `{}` | Keep items where status == 'open' |
| `sort_by_date_desc` | `{"field": str}` | Sort by named date field |
| `top_n` | `{"n": int}` | Take first n items |
| `strip_html` | `{}` | Remove HTML tags |
| `group_by_status` | `{"field": str}` | Return dict keyed by named field |
| `truncate` | `{"max_chars": int}` | Truncate string |
| `to_slack_bullets` | `{"prefix": str = "-"}` | Format list as Slack bullets |

Unknown transform name → `TransformNotFoundError` at adapt-time.

---

## validator.py — 7-stage ordered pipeline

Run in fixed order. First failing stage ends validation.

1. All required slots resolved
2. Slot values conform to type_hints
3. All step inputs satisfied by prior step outputs or slot names
4. Referenced tool_families present in runtime_caps
5. No unresolved slot-shaped placeholders remain
6. Every non-optional output_schema field maps to a concrete producing step output key
7. Side-effecting steps permitted by `runtime_caps.allow_side_effects`

On failure: discard candidate, try next. Exhaust top_k → miss path.

---

## Storage schema (Supabase)

```sql
CREATE TABLE plans (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key               TEXT         NOT NULL,
    intent_key_json       JSONB        NOT NULL,
    intent_bucket_hash    TEXT         NOT NULL,
    description           TEXT         NOT NULL,
    embedding             VECTOR(384)  NOT NULL,
    template_json         JSONB        NOT NULL,
    output_schema_json    JSONB        NOT NULL,
    structural_signature  JSONB        NOT NULL,
    plan_version          TEXT         NOT NULL,
    canonicalizer_version TEXT         NOT NULL,
    extractor_version     TEXT         NOT NULL,
    validator_version     TEXT         NOT NULL,
    created_at            TIMESTAMPTZ  DEFAULT NOW(),
    hit_count             INTEGER      DEFAULT 0,
    last_hit_at           TIMESTAMPTZ,
    is_valid              BOOLEAN      DEFAULT TRUE,
    ttl_seconds           INTEGER      DEFAULT 604800
);
-- Indexes:
-- btree on (api_key, intent_bucket_hash)
-- HNSW on embedding
-- btree on (api_key, is_valid)
```

`structural_signature` shape:
```json
{
  "required_context_keys": ["repo", "time_scope"],
  "tool_families": ["github"],
  "has_side_effects": false,
  "format": "slack_message",
  "audience": "engineering",
  "step_count": 4
}
```

---

## API endpoints (V2)

| Method + Path | Description |
|---|---|
| `POST /v2/plan/lookup` | Main entry: task + context → FilledPlan or miss |
| `POST /v2/plan/store` | Store extracted template (server re-verifies bucket hash) |
| `GET /v2/plan/bucket/:intent_bucket_hash` | List templates for a bucket |
| `DELETE /v2/plan/:id` | Evict single plan |
| `DELETE /v2/plan/bucket/:intent_bucket_hash` | Evict entire bucket |
| `POST /v2/plan/invalidate-by-version` | Bulk soft-invalidate by version |
| `GET /v2/metrics` | All observability counters |
| `GET /v1/cache` | V1 — unchanged |

**`/v2/plan/store` trust model:** server recomputes `intent_bucket_hash` from `intent_key` and rejects mismatches with `400 hash_mismatch`. Store requests are scoped to the authenticated `api_key` — no cross-tenant writes.

---

## Public interface

```python
class PlanCacheV2:
    def __init__(self, api_key: str): ...

    def lookup(self, task: str, context: dict, runtime_caps: dict) -> dict:
        # canonicalize → bucket → rerank → adapt → validate → return hit or miss
        ...

    def store(self, task: str, context: dict, execution_trace: dict,
              planner_output: dict | None, canonicalization_result: dict) -> dict:
        # extract template → verify hash → persist plan
        ...
```

Usage pattern:
```python
cache = PlanCacheV2(api_key="tlm_xxx")
hit = cache.lookup(task=task, context=context, runtime_caps={"tool_families": ["github"], "allow_side_effects": False})

if hit["status"] == "hit":
    filled_plan = hit["filled_plan"]
else:
    planner_output = planner_llm(task, context)
    filled_plan = normalize_planner_output(planner_output)
    canon_result = hit.get("canonicalization_result")
    cache.store(task=task, context=context, execution_trace=planner_output["trace"],
                planner_output=planner_output, canonicalization_result=canon_result)

result = executor.run(filled_plan, context)
```

---

## Build order — FOLLOW THIS

### Phase 1 — prove the engine (current phase)

Build in this exact order. Do not skip or reorder.

1. `thriftlm/v2/schemas.py` — all dataclasses, no logic
2. `thriftlm/v2/intent.py` — OpenAI call (gpt-4o-mini via httpx), normalization, deterministic bucket hash, SLA enforcement
3. `thriftlm/v2/canonicalization_cache.py` — Redis, sha256 key, 1h TTL
4. `thriftlm/v2/plan_cache.py` — bucket fetch, composite rerank, top_k
5. `thriftlm/v2/adapter.py` — SlotSpec resolution, TransformRegistry with transform_args
6. `thriftlm/v2/validator.py` — 7 ordered stages, ValidationResult
7. `thriftlm/v2/_server.py` — `/v2/plan/lookup`, `/v2/plan/store` (hash verify), `/v2/metrics`
8. `thriftlm/v2/adapters/base.py` + `generic.py`
9. Hand-seed 3–5 plan templates in Supabase (see seed plans below)
10. Integration tests: hit path, miss path, store round-trip, hash mismatch rejection

**Phase 1 must also include these negative (rejection) test cases:**
- Wrong bucket: task canonicalizes to a different intent bucket — no match returned
- Missing required slot: context does not provide a required slot key — validation fails stage 1
- Wrong slot type after transform: transform produces wrong type — validation fails stage 2
- Tool family not in runtime_caps: plan requires a tool_family the runtime doesn't have — validation fails stage 4
- Unresolved placeholder: a `{slot_name}` placeholder survives into the filled template — validation fails stage 5
- Side-effecting step with `allow_side_effects=false`: validation fails stage 7

### Phase 2 — automate plan creation (current phase)

11. `thriftlm/v2/extractor.py` — trace generalization, ExtractionResult, refusal logic
12. Unit tests against gold-style traces and expected templates
13. Optional store integration path after successful planner runs
14. Keep extractor conservative and compatible with current adapter/validator invariants

Phase 2 begins now. Phase 1 behavior is locked — do not redesign lookup, hashing,
adapter, or validator logic while building extractor.

### Phase 3 — ship

14. Benchmark — 200 tasks, 5 intent buckets, all metrics
15. Deploy (Fly.io + Upstash Redis + Supabase free tier)
16. Claude Code MCP adapter, Codex CLI hook — roadmap

---

## Seed plans for Phase 1 testing

**Slot/output invariant:** A field may be either a required slot OR a produced step output, but never both in the same template. If a step fetches `prs`, then `prs` must not also be a required slot.

**Seed plans should be context-light.** Prefer slots for identifiers and scopes (repo, account_id, time_scope). Let steps fetch heavy objects (prs, emails, issues) unless the caller truly provides pre-fetched data.

Insert these manually into Supabase before running integration tests.

### Seed 1 — PR summary

```json
{
  "description": "fetch open pull requests, group by status, identify blockers, produce summary",
  "intent_key": {
    "action": "summarize",
    "target": "pull_requests",
    "goal": "identify_blockers",
    "tool_family": "github"
  },
  "steps": [
    {"step_id": "1", "op": "fetch_pull_requests", "inputs": {"repo": "{repo}"}, "outputs": {"prs": "list[pr]"}},
    {"step_id": "2", "op": "group_by_status", "inputs": {"prs": "{prs}"}, "outputs": {"grouped": "dict"}},
    {"step_id": "3", "op": "identify_blockers", "inputs": {"grouped": "{grouped}"}, "outputs": {"blockers": "list[str]"}},
    {"step_id": "4", "op": "produce_summary", "inputs": {"prs": "{prs}", "blockers": "{blockers}"}, "outputs": {"summary": "string"}}
  ],
  "slots": [
    {"name": "repo", "source": "repo", "required": true, "type_hint": "str", "transform": null, "transform_args": null, "default": null}
  ],
  "output_schema": {"summary": "string", "blockers": "list[str]"},
  "optional_outputs": []
}
```

### Seed 2 — Inbox urgency triage

```json
{
  "description": "fetch unread emails, rank urgency, summarize action items",
  "intent_key": {
    "action": "summarize",
    "target": "emails",
    "goal": "identify_urgent",
    "tool_family": "gmail"
  },
  "steps": [
    {"step_id": "1", "op": "fetch_emails", "inputs": {"account_id": "{account_id}"}, "outputs": {"emails": "list[email]"}},
    {"step_id": "2", "op": "rank_urgency", "inputs": {"emails": "{emails}"}, "outputs": {"urgent_emails": "list[email]"}},
    {"step_id": "3", "op": "summarize_action_items", "inputs": {"urgent_emails": "{urgent_emails}"}, "outputs": {"summary": "string"}}
  ],
  "slots": [
    {"name": "account_id", "source": "account_id", "required": true, "type_hint": "str", "transform": null, "transform_args": null, "default": null}
  ],
  "output_schema": {"summary": "string", "urgent_emails": "list[email]"},
  "optional_outputs": ["urgent_emails"]
}
```

### Seed 3 — GitHub issue triage

```json
{
  "description": "scan recent github issues, classify by severity, prepare triage summary",
  "intent_key": {
    "action": "triage",
    "target": "issues",
    "goal": "classify_severity",
    "tool_family": "github"
  },
  "steps": [
    {"step_id": "1", "op": "fetch_issues", "inputs": {"repo": "{repo}", "time_scope": "{time_scope}"}, "outputs": {"issues": "list[issue]"}},
    {"step_id": "2", "op": "classify_severity", "inputs": {"issues": "{issues}"}, "outputs": {"classified": "dict"}},
    {"step_id": "3", "op": "produce_triage_summary", "inputs": {"classified": "{classified}"}, "outputs": {"triage": "string"}}
  ],
  "slots": [
    {"name": "repo", "source": "repo", "required": true, "type_hint": "str", "transform": null, "transform_args": null, "default": null},
    {"name": "time_scope", "source": "time_scope", "required": false, "type_hint": "str", "transform": null, "transform_args": null, "default": "last_7d"}
  ],
  "output_schema": {"triage": "string"},
  "optional_outputs": []
}
```

---

## Free hosting stack (zero cost)

| Component | Service | Notes |
|---|---|---|
| Database + pgvector | Supabase free tier | 500MB, keep-alive cron needed (weekly ping via GitHub Actions) |
| Redis (exact hash + canon cache) | Upstash Redis free | 10K req/day, 256MB |
| FastAPI V2 API | Fly.io free | 3 shared VMs, always-on, `fly launch` |

`.env` additions needed for V2:
```
UPSTASH_REDIS_URL=...
UPSTASH_REDIS_TOKEN=...
```

---

## What NOT to do

- Do not modify `cache.py`, `backends/`, or V1 `_server.py`
- Do not redesign Phase 1 lookup, hashing, adapter, or validator logic while building extractor.py
- Do not use intent_key as a REST path segment — use `intent_bucket_hash`
- Do not tie `plan_threshold` to V1's `response_threshold` (0.82) — they are independent
- Do not trust caller-provided `intent_bucket_hash` in `/v2/plan/store` — recompute server-side
- Do not add "Co-authored-by: Claude" to commits
- Do not run `extractor.py` on traces with fewer than 2 steps unless they have a stable typed I/O shape

---

## Architecture decisions (locked — do not change)

- Bucket hash uses only 4 core fields: action, target, goal, time_scope.
  Optional fields (tool_family, domain, format, audience, constraints) do NOT affect bucket routing.
  Plans seeded under old hashes that included optional fields are now unreachable.
- plan_threshold = 0.60 (SBERT short-text). Raise toward 0.70 if wrong plans
  start adapting successfully as plan bank grows.
- _InMemoryCanonCache is process-local. Does not survive across uvicorn workers or gunicorn restarts.
  Redis is the production cache.

---

## extractor.py — Phase 2 rules

Extractor runs only after a successful planner execution on a V2 miss.

Inputs: task, context, execution_trace, planner_output, canonicalization_result
Output: ExtractionResult (defined in schemas.py)

Extractor invariants:
- Does not execute plans
- Does not call an LLM
- Does not call plan_cache, adapter, or validator
- Conservative: refuse extraction rather than store a brittle template
- v0.1 supports shallow top-level placeholder extraction only
- Must emit structural_signature compatible with plan_cache.py scoring
- Extracted templates must be valid under current adapter/validator assumptions

v0.1 scope (supported):
- exact top-level input value → slot placeholder conversion
- deterministic slot inference from context
- structural_signature generation
- refusal when trace is too weak or too specific to generalize safely

v0.1 scope (NOT supported):
- recursive nested placeholder extraction
- fuzzy abstraction over arbitrary literals
- automatic transform synthesis
- multimodal extraction

Public API:
- EXTRACTOR_VERSION = "v0.4"
- extract_plan_template(task, context, execution_trace, planner_output, canonicalization_result) -> ExtractionResult
- is_extractable_trace(trace: dict) -> tuple[bool, str | None]

---

## Future work / not yet started

### V2.5 — Multimodal extension (future, do not build yet)

Architecture decision: keep IntentKey as semantic routing identity.
Add EvidenceProfile as a sidecar object for modality/perception metadata.
Bucket hash stays on core IntentKey fields only.

Objects to add when ready:
- EvidenceProfile (input_modalities, asset_types, evidence_shape, perception_ops, scale)
- Optional ExecutionProfile

Scoring change when ready:
  final_score = 0.55 * semantic + 0.25 * structural + 0.20 * evidence
  Missing evidence_profile = neutral score, not penalty (safe rollout)

Rollout order: schema only → lookup scoring → extractor support → observation cache

Do not start this until Phase 2 (extractor integration path + e2e tests + benchmark) is complete.

### seed_v2_plans.py — seed_task vs description split (minor, post-0.2.0)
Currently canonicalize() is called on the plan description string, which
works but conflates two jobs: routing (what the canonicalizer needs) and
reranking (what the embedder needs). Clean fix: add a "seed_task" field
to each seed definition, canonicalize that for bucket routing, keep
description purely for embedding similarity. Not a blocker for 0.2.0.

---

## Prior work context

- **APC** (arXiv 2506.14852): plan template reuse, 50.3% cost reduction. V2 adapts plan reuse; addresses keyword brittleness with structured intent keying.
- **W5H2** (arXiv 2602.18922): structured intent canonicalization for short personal-agent queries. Directly motivates V2 Layer 1.
- **GenCache** (NeurIPS 2025): executable program caching for structurally similar prompts. V2 simplifies to typed templates + SlotSpec.
- **Novel V2 contribution**: stacked plan-level + response-level caching, shipped as one pip install. Neither paper ships this.
