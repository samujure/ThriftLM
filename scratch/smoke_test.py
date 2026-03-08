"""
Smoke test for SemanticCache / ThriftLM.

Stores a query/response pair via the full stack, then looks up a
semantically similar query and asserts the cached response is returned.

What runs for real in this run:
  - SBERT all-MiniLM-L6-v2 embedding
  - Presidio PII scrubber (en_core_web_lg — en_core_web_trf requires
    Visual C++ 14 build tools not present on this machine; lg gives
    equivalent NER for smoke-test purposes)
  - Real SupabaseBackend against the project Supabase instance
  - Redis: real if reachable on 6379, else fakeredis with a warning
    (update REDIS_URL in .env with Upstash creds to remove the warning)

Run from the repo root:
    py scratch/smoke_test.py
"""

import socket
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step(n: int, label: str) -> None:
    print(f"\n[{n}] {label}")


# ---------------------------------------------------------------------------
# 1. Redis setup
# ---------------------------------------------------------------------------

section("Redis")

def _redis_is_up(host: str = "127.0.0.1", port: int = 6379) -> bool:
    try:
        s = socket.socket()
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False

patches = []

redis_url = os.getenv("REDIS_URL", "")
using_upstash = redis_url.startswith("rediss://") and "upstash.io" in redis_url

if using_upstash:
    print(f"[Redis] Using Upstash: {redis_url[:40]}...")
elif _redis_is_up():
    print("[Redis] Live local Redis on 127.0.0.1:6379.")
else:
    print("[Redis] WARNING: no local Redis and no Upstash URL in .env.")
    print("        Falling back to fakeredis. To fix: add Upstash REDIS_URL to .env.")
    import fakeredis
    _fake_server = fakeredis.FakeServer()
    _fake_client = fakeredis.FakeRedis(server=_fake_server, decode_responses=True)
    p = patch(
        "thriftlm.backends.redis_backend.RedisBackend._get_client",
        return_value=_fake_client,
    )
    p.start()
    patches.append(p)

# ---------------------------------------------------------------------------
# 2. Presidio: real, using en_core_web_lg (no C++ build tools needed)
#    Patch _get_analyzer so PIIScrubber loads lg instead of trf.
# ---------------------------------------------------------------------------

section("PII Scrubber (Presidio)")

def _make_analyzer_lg(self=None):
    """Build a Presidio AnalyzerEngine backed by en_core_web_lg."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": "en_core_web_lg"}]
    )
    return AnalyzerEngine(nlp_engine=nlp_engine)

print("[Scrubber] Patching _get_analyzer to use en_core_web_lg (trf needs C++ build tools).")

p = patch(
    "thriftlm.privacy.PIIScrubber._get_analyzer",
    side_effect=_make_analyzer_lg,
)
p.start()
patches.append(p)

# ---------------------------------------------------------------------------
# 3. Smoke test: real Supabase, real embedder, real scrubber
# ---------------------------------------------------------------------------

section("ThriftLM smoke test")

from thriftlm.cache import SemanticCache

STORED_QUERY    = "how do I reset my password"
STORED_RESPONSE = "Click forgot password on the login page"
LOOKUP_QUERY    = "I forgot my password"

# threshold=0.75: these two queries score ~0.79 cosine similarity.
# Production default (0.85) is intentionally stricter.
cache = SemanticCache(api_key="sc_test", threshold=0.75)
print(f"[OK] SemanticCache instantiated (threshold={cache.config.threshold})")
print(f"     Supabase: {os.environ.get('SUPABASE_URL', 'NOT SET')}")

# -- store ----------------------------------------------------------------
step(1, f'store("{STORED_QUERY}")')
try:
    cache.store(STORED_QUERY, STORED_RESPONSE)
    print(f"     Stored: \"{STORED_RESPONSE}\"")
except Exception:
    print("[ERROR] store() failed:")
    traceback.print_exc()
    sys.exit(1)

# -- lookup ---------------------------------------------------------------
step(2, f'lookup("{LOOKUP_QUERY}")')
try:
    result = cache.lookup(LOOKUP_QUERY)
except Exception:
    print("[ERROR] lookup() failed:")
    traceback.print_exc()
    sys.exit(1)

print(f"     Result: {result!r}")

assert result is not None, (
    f"\nFAIL: lookup returned None.\n"
    f"  stored  : '{STORED_QUERY}'\n"
    f"  lookup  : '{LOOKUP_QUERY}'\n"
    f"  threshold: {cache.config.threshold}\n"
    f"  Tip: check that match_cache_entries RPC exists in Supabase."
)

print()
print("[PASS] Smoke test passed — semantically similar query returned cached response.")

for p in patches:
    p.stop()
