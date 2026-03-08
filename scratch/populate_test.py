"""
populate_test.py — cache population and paraphrase validation for ThriftLM.

Stores 15 hardcoded ThriftLM Q&A pairs, then tests 10 paraphrased queries
against the cache to validate semantic retrieval. No LLM calls made.

Run from repo root:
    py scratch/populate_test.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Presidio patch — use en_core_web_lg (no C++ build tools needed)
# ---------------------------------------------------------------------------

def _make_analyzer_lg(self=None):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": "en_core_web_lg"}]
    )
    return AnalyzerEngine(nlp_engine=nlp_engine)

print("[Scrubber] Patching _get_analyzer to use en_core_web_lg.")

p_scrubber = patch(
    "thriftlm.privacy.PIIScrubber._get_analyzer",
    side_effect=_make_analyzer_lg,
)
p_scrubber.start()

# ---------------------------------------------------------------------------
# SemanticCache setup
# ---------------------------------------------------------------------------

from thriftlm.cache import SemanticCache

cache = SemanticCache(api_key="sc_demo", threshold=0.75)
print(f"[OK] SemanticCache instantiated (api_key=sc_demo, threshold={cache.config.threshold})\n")

# ---------------------------------------------------------------------------
# 15 Q&A pairs about ThriftLM
# ---------------------------------------------------------------------------

QA_PAIRS = [
    (
        "What is ThriftLM?",
        "ThriftLM is a semantic caching layer for LLM applications. It embeds user queries "
        "with SBERT, checks a vector store for semantically similar past queries, and returns "
        "the cached response when similarity exceeds the threshold — avoiding redundant LLM calls.",
    ),
    (
        "How do I install ThriftLM?",
        "Install ThriftLM with: pip install thriftlm. It is published on PyPI and works with "
        "Python 3.9+.",
    ),
    (
        "How does get_or_call work?",
        "get_or_call(query, llm_fn) embeds the query, checks Redis then Supabase for a cache "
        "hit, and returns the cached response if found. On a miss it calls llm_fn(query), "
        "stores the result in both backends, and returns the fresh response.",
    ),
    (
        "What is the cache lookup order in ThriftLM?",
        "ThriftLM checks three layers in order: (1) Redis — exact embedding hash, microsecond "
        "latency; (2) Supabase pgvector — cosine similarity search, millisecond latency; "
        "(3) LLM function — called only on a full cache miss.",
    ),
    (
        "What does PII scrubbing do in ThriftLM?",
        "Before storing any query or response, ThriftLM runs Presidio to detect and redact "
        "personally identifiable information (names, emails, phone numbers, etc.). PII is "
        "never written to Redis or Supabase.",
    ),
    (
        "How do I self-host ThriftLM with Docker Compose?",
        "Clone the repo and run: docker compose up. This starts the FastAPI backend and Redis "
        "locally. Set THRIFTLM_URL=http://localhost:8000 in your environment. Supabase remains "
        "an external dependency — provide your own SUPABASE_URL and SUPABASE_KEY.",
    ),
    (
        "What does the threshold parameter control?",
        "The threshold (default 0.85) sets the minimum cosine similarity required for a "
        "Supabase vector search to count as a cache hit. Lower values match more liberally; "
        "higher values require near-identical phrasing.",
    ),
    (
        "What does the ttl parameter control?",
        "The ttl (time-to-live, default 86400 seconds = 24 hours) controls how long cached "
        "entries persist in Redis before expiring. Supabase entries do not expire automatically.",
    ),
    (
        "What are the two deployment modes for ThriftLM?",
        "ThriftLM has two modes: (1) Hosted — get an API key and point at the cloud instance, "
        "zero infrastructure to manage; (2) Self-hosted — run docker compose up locally and "
        "set THRIFTLM_URL=http://localhost:8000. One environment variable is the entire difference.",
    ),
    (
        "How do I get a ThriftLM API key?",
        "POST your email to /keys: curl -X POST https://api.thriftlm.dev/keys "
        "-d '{\"email\": \"you@example.com\"}'. The response contains your sc_xxx API key.",
    ),
    (
        "What embedding model does ThriftLM use?",
        "ThriftLM uses all-MiniLM-L6-v2 from sentence-transformers. It runs locally inside "
        "the library — no external embedding API calls are made.",
    ),
    (
        "What vector database does ThriftLM use?",
        "ThriftLM uses Supabase with the pgvector extension for vector storage. Each cache "
        "entry is namespaced by API key so tenants are fully isolated.",
    ),
    (
        "How do I use ThriftLM in 3 lines of code?",
        "from thriftlm import SemanticCache\n"
        "cache = SemanticCache(api_key='sc_xxx')\n"
        "response = cache.get_or_call(query, llm_fn=my_chain.invoke)",
    ),
    (
        "What is the Redis layer used for in ThriftLM?",
        "Redis is the fast-path cache layer (Upstash Redis in hosted mode). It stores an "
        "exact hash of the embedding and returns cached responses in microseconds, "
        "bypassing Supabase entirely on repeated identical queries.",
    ),
    (
        "Does ThriftLM support LangGraph?",
        "Yes. ThriftLM includes a native LangGraph integration. Wrap any LangGraph node's "
        "invoke call with cache.get_or_call and the cache layer is transparent to the rest "
        "of the graph.",
    ),
]

# ---------------------------------------------------------------------------
# Store all 15 pairs
# ---------------------------------------------------------------------------

print(f"{'-' * 60}")
print(f"  STORING {len(QA_PAIRS)} Q&A PAIRS")
print(f"{'-' * 60}")

for q, a in QA_PAIRS:
    cache.store(q, a)
    print(f"[STORED] {q}")

# ---------------------------------------------------------------------------
# 10 paraphrased queries
# ---------------------------------------------------------------------------

PARAPHRASES = [
    "Can you explain what ThriftLM is?",
    "What pip package do I install to use ThriftLM?",
    "Walk me through what get_or_call does step by step.",
    "In what order does ThriftLM check the cache before calling the LLM?",
    "Why does ThriftLM scrub personal information before caching?",
    "What is the similarity threshold for?",
    "How long do cached entries live in ThriftLM?",
    "What's the difference between hosted and self-hosted ThriftLM?",
    "How can I generate an API key for ThriftLM?",
    "Which sentence-transformers model does ThriftLM embed queries with?",
]

# ---------------------------------------------------------------------------
# Lookup paraphrases
# ---------------------------------------------------------------------------

print(f"\n{'-' * 60}")
print(f"  TESTING {len(PARAPHRASES)} PARAPHRASED QUERIES")
print(f"{'-' * 60}")

hits = 0
for paraphrase in PARAPHRASES:
    result = cache.lookup(paraphrase)
    if result is not None:
        hits += 1
        preview = result[:80].replace("\n", " ") + ("…" if len(result) > 80 else "")
        print(f"[HIT]  {paraphrase}")
        print(f"       > {preview}")
    else:
        print(f"[MISS] {paraphrase}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = len(PARAPHRASES)
pct = hits / total * 100
print(f"\n{'-' * 60}")
print(f"  Results: {hits}/{total} hits ({pct:.0f}%)")
print(f"{'-' * 60}")

p_scrubber.stop()
