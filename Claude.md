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
from semantic_cache import SemanticCache

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

## V1 Scope
- [ ] pip-installable Python library (`semantic_cache/`)
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
semanticCache/
├── semantic_cache/              # pip package
│   ├── __init__.py
│   ├── cache.py                 # SemanticCache class — core logic
│   ├── embedder.py              # SBERT wrapper (all-MiniLM-L6-v2)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── supabase_backend.py  # pgvector via Supabase
│   │   └── redis_backend.py     # Upstash Redis
│   └── config.py                # threshold, TTL, env vars
├── api/                         # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── cache.py             # /lookup, /store endpoints
│   │   └── metrics.py           # /metrics endpoint for dashboard
│   ├── auth.py                  # API key validation
│   └── db.py                    # Supabase client
├── dashboard/                   # Simple HTML/JS dashboard
│   └── index.html               # hit rate, tokens saved, cost saved
├── tests/
│   ├── test_cache.py
│   ├── test_embedder.py
│   └── test_api.py
├── docker-compose.yml           # self-host: Redis + API (Supabase is external)
├── pyproject.toml               # pip packaging
├── .env.example                 # SUPABASE_URL, SUPABASE_KEY, REDIS_URL, etc.
├── CLAUDE.md                    # this file
└── README.md
```

---

## Key Classes

### SemanticCache (semantic_cache/cache.py)
```python
class SemanticCache:
    def __init__(self, api_key: str, threshold: float = 0.85, ttl: int = 86400)
    def get_or_call(self, query: str, llm_fn: callable) -> str
    def lookup(self, query: str) -> str | None
    def store(self, query: str, response: str) -> None
```

### Embedder (semantic_cache/embedder.py)
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

## Current Status
Scaffolding phase. Start with:
1. `semantic_cache/embedder.py` — get SBERT working and returning embeddings
2. `semantic_cache/backends/supabase_backend.py` — lookup and store against pgvector
3. `semantic_cache/cache.py` — wire embedder + backend into SemanticCache class
4. `api/` — FastAPI wrapper around the same backend
5. `dashboard/index.html` — metrics UI
6. `pyproject.toml` — pip packaging
7. Publish to PyPI

---

## What NOT To Build
- Custom vector database
- Multiple embedding model options (just all-MiniLM-L6-v2 for now)
- User accounts / billing / paid tiers
- V2 cross-app shared pool
- Anything not in the V1 scope checklist above

