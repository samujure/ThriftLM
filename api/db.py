"""
Infrastructure singletons for the API layer.

Provides lazy-initialised, module-level singletons for:
  - The raw Supabase client  (used by /metrics and /keys for direct table queries)
  - SupabaseBackend          (used by /lookup and /store for vector operations)
  - RedisBackend             (used by /lookup and /store for fast exact-match cache)

All factories are safe to use as FastAPI Depends() callables.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from thriftlm.backends.redis_backend import RedisBackend
from thriftlm.backends.supabase_backend import SupabaseBackend

# ---------------------------------------------------------------------------
# Module-level singletons (None until first request)
# ---------------------------------------------------------------------------

_supabase_client: Optional[Any] = None
_supabase_backend: Optional[SupabaseBackend] = None
_redis_backend: Optional[RedisBackend] = None


# ---------------------------------------------------------------------------
# Raw Supabase client
# ---------------------------------------------------------------------------

def get_supabase_client() -> Any:
    """Return the shared Supabase client, creating it on first call.

    Reads SUPABASE_URL and SUPABASE_KEY from the environment.

    Returns:
        A ``supabase.Client`` instance.

    Raises:
        KeyError: If SUPABASE_URL or SUPABASE_KEY are not set.
    """
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        _supabase_client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _supabase_client


# ---------------------------------------------------------------------------
# SupabaseBackend — pgvector similarity search
# ---------------------------------------------------------------------------

def get_supabase_backend() -> SupabaseBackend:
    """Return the shared SupabaseBackend, creating it on first call.

    Reads SUPABASE_URL, SUPABASE_KEY, and SIMILARITY_THRESHOLD from env.

    Returns:
        A ready-to-use :class:`SupabaseBackend` instance.
    """
    global _supabase_backend
    if _supabase_backend is None:
        _supabase_backend = SupabaseBackend(
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=os.environ["SUPABASE_KEY"],
            threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        )
    return _supabase_backend


# ---------------------------------------------------------------------------
# RedisBackend — fast exact-match cache
# ---------------------------------------------------------------------------

def get_redis_backend() -> RedisBackend:
    """Return the shared RedisBackend, creating it on first call.

    Reads REDIS_URL and CACHE_TTL_SECONDS from the environment.

    Returns:
        A ready-to-use :class:`RedisBackend` instance.
    """
    global _redis_backend
    if _redis_backend is None:
        _redis_backend = RedisBackend(
            redis_url=os.environ["REDIS_URL"],
            ttl=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
        )
    return _redis_backend
