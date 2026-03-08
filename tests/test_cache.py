"""
Tests for thriftlm.cache.SemanticCache.

All three backends (Embedder, RedisBackend, SupabaseBackend) are replaced
with MagicMock instances — no real network, DB, or model calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FAKE_API_KEY = "sc_test123"
FAKE_EMBEDDING = [0.1] * 384
FAKE_RESPONSE = "Paris is the capital of France."
FAKE_QUERY = "What is the capital of France?"

ENV_VARS = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "test-key",
    "REDIS_URL": "redis://localhost:6379",
}

# ---------------------------------------------------------------------------
# Fixture: a fully-mocked SemanticCache instance
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(monkeypatch):
    """Return a SemanticCache with all backends replaced by MagicMocks."""
    for k, v in ENV_VARS.items():
        monkeypatch.setenv(k, v)

    # Patch constructors so __init__ never touches real services
    with patch("thriftlm.cache.Embedder") as MockEmbedder, \
         patch("thriftlm.cache.RedisBackend") as MockRedis, \
         patch("thriftlm.cache.SupabaseBackend") as MockSupabase, \
         patch("thriftlm.cache.PIIScrubber") as MockScrubber:

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = FAKE_EMBEDDING
        MockEmbedder.return_value = mock_embedder

        mock_redis = MagicMock()
        MockRedis.return_value = mock_redis

        mock_supabase = MagicMock()
        MockSupabase.return_value = mock_supabase

        # Passthrough scrubber — existing tests are not testing PII logic.
        mock_scrubber = MagicMock()
        mock_scrubber.scrub.side_effect = lambda text: text
        MockScrubber.return_value = mock_scrubber

        from thriftlm.cache import SemanticCache
        sc = SemanticCache(api_key=FAKE_API_KEY)

        # Expose mocks on the instance for assertions
        sc._embedder = mock_embedder
        sc._redis = mock_redis
        sc._supabase = mock_supabase
        sc._scrubber = mock_scrubber

        yield sc


# ---------------------------------------------------------------------------
# get_or_call — Redis HIT
# ---------------------------------------------------------------------------

def test_cache_hit_redis(cache):
    """Redis hit: llm_fn never called, Supabase never queried."""
    cache._redis.get.return_value = FAKE_RESPONSE
    llm_fn = MagicMock()

    result = cache.get_or_call(FAKE_QUERY, llm_fn)

    assert result == FAKE_RESPONSE
    llm_fn.assert_not_called()
    cache._supabase.lookup.assert_not_called()
    cache._redis.set.assert_not_called()


# ---------------------------------------------------------------------------
# get_or_call — Redis MISS, Supabase HIT
# ---------------------------------------------------------------------------

def test_cache_hit_supabase(cache):
    """Supabase hit: llm_fn never called, response written back to Redis."""
    cache._redis.get.return_value = None
    cache._supabase.lookup.return_value = FAKE_RESPONSE
    llm_fn = MagicMock()

    result = cache.get_or_call(FAKE_QUERY, llm_fn)

    assert result == FAKE_RESPONSE
    llm_fn.assert_not_called()
    cache._redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# get_or_call — both MISS
# ---------------------------------------------------------------------------

def test_cache_miss(cache):
    """Both miss: llm_fn called once, response stored in both backends."""
    cache._redis.get.return_value = None
    cache._supabase.lookup.return_value = None
    llm_fn = MagicMock(return_value=FAKE_RESPONSE)

    result = cache.get_or_call(FAKE_QUERY, llm_fn)

    assert result == FAKE_RESPONSE
    llm_fn.assert_called_once_with(FAKE_QUERY)
    cache._supabase.store.assert_called_once_with(
        FAKE_QUERY, FAKE_RESPONSE, FAKE_EMBEDDING, FAKE_API_KEY
    )
    cache._redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_store_writes_both_backends(cache):
    """store() embeds the query and writes to both Supabase and Redis."""
    cache.store(FAKE_QUERY, FAKE_RESPONSE)

    cache._embedder.embed.assert_called_once_with(FAKE_QUERY)
    cache._supabase.store.assert_called_once_with(
        FAKE_QUERY, FAKE_RESPONSE, FAKE_EMBEDDING, FAKE_API_KEY
    )
    cache._redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

def test_lookup_returns_none_on_miss(cache):
    """lookup() returns None when both Redis and Supabase miss."""
    cache._redis.get.return_value = None
    cache._supabase.lookup.return_value = None

    result = cache.lookup(FAKE_QUERY)

    assert result is None


def test_lookup_returns_redis_hit(cache):
    """lookup() returns Redis cached value without touching Supabase."""
    cache._redis.get.return_value = FAKE_RESPONSE

    result = cache.lookup(FAKE_QUERY)

    assert result == FAKE_RESPONSE
    cache._supabase.lookup.assert_not_called()


def test_lookup_supabase_hit_promotes_to_redis(cache):
    """lookup() promotes a Supabase hit into Redis for the next call."""
    cache._redis.get.return_value = None
    cache._supabase.lookup.return_value = FAKE_RESPONSE

    result = cache.lookup(FAKE_QUERY)

    assert result == FAKE_RESPONSE
    cache._redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_threshold(cache):
    assert cache.config.threshold == 0.85


def test_default_ttl(cache):
    assert cache.config.ttl == 86400


def test_api_key_stored(cache):
    assert cache.config.api_key == FAKE_API_KEY
