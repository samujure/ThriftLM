<div align="center">

# ThriftLM

**Semantic cache layer for LLM applications.**
Redis-fast exact hits. Numpy-powered near-miss matching. PII-scrubbed by default.

[![PyPI version](https://badge.fury.io/py/thriftlm.svg)](https://pypi.org/project/thriftlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install thriftlm
```

</div>

---

## Overview

Every repeated or semantically similar LLM query burns tokens and adds latency. ThriftLM intercepts these calls with a three-tier cache — exact hash match in Redis, cosine similarity search in a local numpy index, and HNSW vector search in Supabase — before any request reaches your LLM provider.

**73.5% hit rate at threshold=0.82** on the Quora Question Pairs benchmark. The median semantic cache hit returns in ~1ms vs. 2–12 seconds for a live LLM call.

---

## How It Works

```
query
  │
  ▼
┌─────────────────┐     HIT → return instantly (~0.5ms)
│   Redis         │
│  (exact hash)   │
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐     HIT → Supabase PK fetch → return (~50ms)
│  Local Numpy    │
│  Index (cosine) │
└────────┬────────┘
         │ MISS
         ▼
┌─────────────────┐
│   LLM Call      │     Your llm_fn() called here
│  (your function)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PII Scrubbing  │     Presidio strips names, emails, phone numbers
│  (response only)│
└────────┬────────┘
         │
         ▼
   Store in Supabase + LocalIndex + Redis
```

Cache hit order:
1. **Redis** — exact embedding hash, microseconds, no DB call
2. **Local numpy index** — cosine similarity matmul, ~1ms, Supabase PK fetch for response
3. **LLM** — cache miss only, full latency, stored after Presidio scrub

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Supabase](https://supabase.com) project with pgvector enabled
- Redis (local via Docker or [Upstash](https://upstash.com))

### 1. Install

```bash
pip install thriftlm
```

### 2. Set up Supabase

Run `supabase/setup.sql` in your Supabase SQL editor. It creates:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cache_entries (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key     TEXT NOT NULL,
    query       TEXT NOT NULL,
    response    TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    last_hit_at TIMESTAMPTZ,
    hit_count   INTEGER DEFAULT 0
);

CREATE INDEX cache_entries_embedding_idx
    ON cache_entries
    USING hnsw (embedding vector_cosine_ops);
```

Plus two RPC functions (`match_cache_entries`, `increment_api_key_counters`) — see the full file for those.

### 3. Configure environment

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
REDIS_URL=redis://localhost:6379
```

### 4. Run Redis

```bash
docker compose up -d
```

### 5. Integrate

```python
from thriftlm import SemanticCache
import openai

# Initialize once per process
cache = SemanticCache(threshold=0.85, api_key="your-key")

def call_llm(query: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# Drop-in wrapper
response = cache.get_or_call("Explain semantic caching", call_llm)

# Near-duplicate → instant cache hit, no LLM called
response2 = cache.get_or_call("What is semantic caching?", call_llm)
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.85` | Cosine similarity cutoff. Lower = more aggressive matching. |
| `api_key` | required | Namespaces cache per tenant. Each key has its own LocalIndex. |

**Threshold guide:**

| Threshold | Hit Rate (QQP) | Use case |
|---|---|---|
| 0.70 | 92.5% | Aggressive — high savings, some false positives |
| **0.82** | **73.5%** | **Balanced — recommended for most apps** |
| 0.85 | 62.5% | Default — conservative |
| 0.90 | 40.0% | Near-exact only |

---

## Architecture

**Embedding:** `all-MiniLM-L6-v2` (384-dim). Runs locally, no API cost.

**Local numpy index:** On `SemanticCache()` init, all stored embeddings are bulk-fetched into a `(N, 384)` float32 matrix. Cosine similarity is a single `matrix @ query_vec` matmul — ~1ms regardless of cache size. New entries append via `np.vstack`.

**Supabase HNSW:** pgvector with HNSW index for accurate ANN at scale. Used for cold-start loading and as fallback.

**PII scrubbing:** Presidio + spaCy `en_core_web_lg`. Applied to LLM **responses only** before storage. Queries are not scrubbed — scrubbing before embedding causes embedding drift and kills recall.

---

## Benchmark

200 duplicate question pairs from [Quora Question Pairs](https://huggingface.co/datasets/nyu-mll/glue/viewer/qqp).

```
Threshold | Hit Rate | Hits / 200
----------|----------|------------
0.70      |  92.5%   |   185
0.75      |  86.0%   |   172
0.80      |  78.0%   |   156
0.82 ←    |  73.5%   |   147   (recommended)
0.85      |  62.5%   |   125   (default)
0.90      |  40.0%   |    80

Model: all-MiniLM-L6-v2 · Index: HNSW (Supabase pgvector)
Dataset: mean sim=0.859, min=0.550, max=0.999
```

---

## Project Structure

```
ThriftLM/
├── thriftlm/
│   ├── __init__.py              # Public API: SemanticCache
│   ├── cache.py                 # Core lookup/store logic
│   ├── cli.py                   # thriftlm serve CLI entry point
│   ├── _server.py               # FastAPI app for thriftlm serve
│   ├── config.py                # Env config
│   ├── embedder.py              # SBERT wrapper
│   ├── privacy.py               # Presidio PII scrubbing
│   ├── static/
│   │   └── dashboard.html       # Metrics dashboard (pip-bundled)
│   └── backends/
│       ├── local_index.py       # Numpy cosine index
│       ├── redis_backend.py     # Exact hash cache
│       └── supabase_backend.py  # Vector storage + PK fetch
├── api/
│   ├── main.py                  # FastAPI app
│   ├── auth.py                  # API key auth
│   └── routes/
│       ├── cache.py             # /lookup, /store
│       ├── metrics.py           # /metrics
│       └── keys.py              # /keys
├── docs/
│   └── index.html               # Landing page (GitHub Pages + FastAPI /)
├── tests/                       # 66 passing tests
├── scratch/
│   ├── smoke_test.py
│   ├── openai_test.py
│   ├── populate_test.py
│   └── qqp_benchmark.py
├── supabase/setup.sql
├── docker-compose.yml
└── pyproject.toml
```

---

## REST API

```bash
uvicorn api.main:app --reload
```

```
POST /lookup    { "embedding": [...], "api_key": "..." }           → { "response": "..." | null }
POST /store     { "embedding": [...], "query": "...", "response": "...", "api_key": "..." }  → 200
GET  /metrics   header: X-API-Key                                  → { hit_rate, tokens_saved, cost_saved, total_queries }
POST /keys      { "email": "..." }                                 → { "api_key": "sc_..." }
GET  /health                                                       → { "status": "ok" }
GET  /                                                             → landing page (docs/index.html)
```

---

## Dashboard

`thriftlm serve` starts a local FastAPI server that serves the metrics dashboard and reads your Supabase data directly. Bundled inside the pip package — no separate deploy needed.

```bash
# Requires the api extras for FastAPI + Supabase
pip install thriftlm[api]

# Start the dashboard (auto-opens browser at http://localhost:8000)
thriftlm serve --api-key sc_xxx

# Custom port or host
thriftlm serve --api-key sc_xxx --port 9000 --host 0.0.0.0

# Skip auto-open
thriftlm serve --api-key sc_xxx --no-browser
```

Make sure `SUPABASE_URL` and `SUPABASE_KEY` are set in your `.env` before running.

The dashboard shows hit rate, total queries, tokens saved, cost saved, and top 5 queries by cache hits — updating live every 30 seconds.

---

## Development

```bash
git clone https://github.com/samujure/ThriftLM
cd ThriftLM
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
pytest tests/ -v
python scratch/smoke_test.py
python scratch/qqp_benchmark.py
```

---

## Roadmap

**V1 — Shipped ✓**
- Three-tier cache: Redis → LocalIndex → HNSW
- Presidio PII scrubbing on responses
- Multi-tenant FastAPI + API key auth
- `pip install thriftlm`
- Landing page (`docs/`) + metrics dashboard (`thriftlm/static/`)

**V2 — Agentic Plan Caching (next)**

V1 caches individual responses. V2 caches entire agent **plans** — multi-step action sequences generated for agentic loops. When intent repeats, skip re-planning and replay the cached plan. Built for Claude Code SDK and long-running agent workflows.

Key papers: [APC (arxiv 2506.14852)](https://arxiv.org/abs/2506.14852) · [GenCache](https://openreview.net/pdf?id=MHGViOjZ27)

**Later**
- ClawHub / OpenClaw distribution
- Per-model cost analytics dashboard
- Precision benchmark (false positive rate on non-duplicate pairs)

---

## License

MIT

---

<div align="center">
Built by Srivamsi Amujure & Ivan Thomas Shen
</div>
