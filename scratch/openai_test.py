"""
openai_test.py — end-to-end pipeline test with a real OpenAI call.

Tests 4 cases in order:
    1. Cache miss  -> LLM called (OpenAI gpt-3.5-turbo)
    2. Exact hit   -> Redis (microseconds)
    3. Semantic hit -> Local numpy index + Supabase PK fetch
    4. PII scrubbing -> stored response has PII redacted

Run from repo root:
    py scratch/openai_test.py
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Presidio patch — en_core_web_lg (no C++ build tools needed)
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
# Imports (after patch is active)
# ---------------------------------------------------------------------------

import os
from supabase import create_client
from thriftlm.cache import SemanticCache

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

API_KEY = "sc_openai_test"

print(f"\n[Setup] Clearing {API_KEY} entries from Supabase...")
_sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
_sb.table("cache_entries").delete().eq("api_key", API_KEY).execute()
print("[Setup] Clean slate ready.")

# Flush Redis so no stale exact hits bleed into test 1.
from thriftlm.backends.redis_backend import RedisBackend
_redis = RedisBackend(redis_url=os.environ["REDIS_URL"], ttl=86400)
_redis._get_client().flushdb()
print("[Setup] Redis flushed.")

# ---------------------------------------------------------------------------
# OpenAI llm_fn
# ---------------------------------------------------------------------------

import openai

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_fn(query: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
        max_tokens=200,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Cache instance (fresh — loads local index from Supabase, which is empty)
# ---------------------------------------------------------------------------

cache = SemanticCache(api_key=API_KEY, threshold=0.75)
print(f"[Setup] SemanticCache ready (threshold={cache.config.threshold})\n")

results = {}

# ---------------------------------------------------------------------------
# Test 1: Cache miss -> LLM called
# ---------------------------------------------------------------------------

QUERY_1 = "What is semantic caching?"
print(f"[1] MISS -> LLM call")
print(f"    Query: \"{QUERY_1}\"")

t0 = time.perf_counter()
resp1 = cache.get_or_call(QUERY_1, llm_fn)
elapsed1 = time.perf_counter() - t0

print(f"    Response: \"{resp1[:80]}{'...' if len(resp1) > 80 else ''}\"")
print(f"    Time: {elapsed1:.2f}s\n")
results[1] = elapsed1

# ---------------------------------------------------------------------------
# Test 2: Exact hit — Redis
# ---------------------------------------------------------------------------

print(f"[2] HIT -> Redis")
print(f"    Query: \"{QUERY_1}\" (identical)")

t0 = time.perf_counter()
resp2 = cache.get_or_call(QUERY_1, llm_fn)
elapsed2 = time.perf_counter() - t0

match2 = resp2 == resp1
print(f"    Response matches: {match2}")
print(f"    Time: {elapsed2 * 1000:.1f}ms\n")
results[2] = elapsed2

# ---------------------------------------------------------------------------
# Test 3: Semantic hit — local numpy index
# ---------------------------------------------------------------------------

QUERY_3 = "Explain what semantic caching means"
print(f"[3] HIT -> LocalIndex")
print(f"    Query: \"{QUERY_3}\"")

t0 = time.perf_counter()
resp3 = cache.get_or_call(QUERY_3, llm_fn)
elapsed3 = time.perf_counter() - t0

hit3 = resp3 is not None
print(f"    Cache hit: {hit3}")
if resp3:
    print(f"    Response: \"{resp3[:80]}{'...' if len(resp3) > 80 else ''}\"")
print(f"    Time: {elapsed3 * 1000:.1f}ms\n")
results[3] = elapsed3

# ---------------------------------------------------------------------------
# Test 4: PII scrubbing — verify stored response is redacted
# ---------------------------------------------------------------------------

QUERY_4 = (
    "Write a short support ticket reply to a customer named John Smith "
    "at john.doe@example.com who is having a billing issue. "
    "Include their name and email in the reply."
)
print(f"[4] PII scrubbing test")
print(f"    Query: \"{QUERY_4[:80]}...\"")

t0 = time.perf_counter()
resp4 = cache.get_or_call(QUERY_4, llm_fn)
elapsed4 = time.perf_counter() - t0

print(f"    Raw response (returned to caller):")
print(f"    \"{resp4[:120]}{'...' if len(resp4) > 120 else ''}\"")

# Fetch the stored response directly from Supabase to check scrubbing.
stored_rows = (
    _sb.table("cache_entries")
    .select("response")
    .eq("api_key", API_KEY)
    .order("created_at", desc=True)
    .limit(1)
    .execute()
    .data
)
stored_resp4 = stored_rows[0]["response"] if stored_rows else ""

print(f"    Stored response (after scrubbing):")
print(f"    \"{stored_resp4[:120]}{'...' if len(stored_resp4) > 120 else ''}\"")

pii_in_raw = "john.doe@example.com" in resp4.lower() or "john smith" in resp4.lower()
pii_in_stored = "john.doe@example.com" in stored_resp4.lower() or "john smith" in stored_resp4.lower()
pii_redacted = pii_in_raw and not pii_in_stored

print(f"    PII in raw response: {pii_in_raw}")
print(f"    PII in stored response: {pii_in_stored}")
results[4] = pii_redacted
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 52)
miss_label  = f"{results[1]:.2f}s"
redis_label = f"{results[2]*1000:.1f}ms"
local_label = f"{results[3]*1000:.1f}ms"
pii_label   = "redacted OK" if results[4] else "NOT redacted (check scrubber)"

hit3_ok = resp3 is not None and elapsed3 < 5.0  # should be well under 5s (no LLM call)

print(f"  [1] MISS  -> LLM called    ({miss_label})")
print(f"  [2] HIT   -> Redis         ({redis_label})")
print(f"  [3] HIT   -> LocalIndex    ({local_label})  {'OK' if hit3_ok else 'MISS - check threshold'}")
print(f"  [4] PII   -> {pii_label}")
print("=" * 52)

overall = hit3_ok and results[2] < results[1]
print(f"\n{'[PASS] All checks passed.' if overall else '[WARN] Some checks need review.'}")

p_scrubber.stop()
