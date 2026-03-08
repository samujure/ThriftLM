"""
Tests for thriftlm.backends.redis_backend.RedisBackend.

The Redis client is fully mocked — no real Redis connection is made.
"""

from unittest.mock import MagicMock, patch

import pytest

from thriftlm.backends.redis_backend import RedisBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_URL = "redis://localhost:6379"
FAKE_EMBEDDING = [round(i * 0.001, 4) for i in range(384)]
FAKE_RESPONSE = "The boiling point of water is 100°C."


def _make_backend(mock_redis: MagicMock) -> RedisBackend:
    """Inject a pre-built mock Redis client into a RedisBackend instance."""
    backend = RedisBackend(FAKE_URL, ttl=3600)
    backend._client = mock_redis
    return backend


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_hit():
    """get() returns the cached response string when Redis has the key."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = FAKE_RESPONSE

    backend = _make_backend(mock_redis)
    result = backend.get(FAKE_EMBEDDING)

    assert result == FAKE_RESPONSE
    mock_redis.get.assert_called_once_with(backend._make_key(FAKE_EMBEDDING))


def test_get_miss():
    """get() returns None when Redis does not have the key."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    backend = _make_backend(mock_redis)
    result = backend.get(FAKE_EMBEDDING)

    assert result is None


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

def test_set_stores_with_default_ttl():
    """set() calls setex with the instance TTL when no override is given."""
    mock_redis = MagicMock()

    backend = _make_backend(mock_redis)
    backend.set(FAKE_EMBEDDING, FAKE_RESPONSE)

    expected_key = backend._make_key(FAKE_EMBEDDING)
    mock_redis.setex.assert_called_once_with(expected_key, 3600, FAKE_RESPONSE)


def test_set_stores_with_ttl_override():
    """set() uses the provided ttl argument instead of the instance default."""
    mock_redis = MagicMock()

    backend = _make_backend(mock_redis)
    backend.set(FAKE_EMBEDDING, FAKE_RESPONSE, ttl=600)

    expected_key = backend._make_key(FAKE_EMBEDDING)
    mock_redis.setex.assert_called_once_with(expected_key, 600, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_removes_key():
    """delete() calls Redis DELETE with the correct hashed key."""
    mock_redis = MagicMock()

    backend = _make_backend(mock_redis)
    backend.delete(FAKE_EMBEDDING)

    expected_key = backend._make_key(FAKE_EMBEDDING)
    mock_redis.delete.assert_called_once_with(expected_key)


# ---------------------------------------------------------------------------
# _make_key
# ---------------------------------------------------------------------------

def test_make_key_is_deterministic():
    """Same embedding always produces the same cache key."""
    backend = RedisBackend(FAKE_URL)
    key_a = backend._make_key(FAKE_EMBEDDING)
    key_b = backend._make_key(FAKE_EMBEDDING)
    assert key_a == key_b


def test_make_key_different_embeddings_differ():
    """Different embeddings produce different cache keys."""
    backend = RedisBackend(FAKE_URL)
    other_embedding = [round(i * 0.002, 4) for i in range(384)]
    assert backend._make_key(FAKE_EMBEDDING) != backend._make_key(other_embedding)


def test_make_key_is_short():
    """Cache key must be under 80 characters (sc: prefix + 64-char hex digest = 67)."""
    backend = RedisBackend(FAKE_URL)
    key = backend._make_key(FAKE_EMBEDDING)
    assert len(key) < 80


def test_make_key_prefix():
    """Cache key must start with 'sc:'."""
    backend = RedisBackend(FAKE_URL)
    assert backend._make_key(FAKE_EMBEDDING).startswith("sc:")


# ---------------------------------------------------------------------------
# Client lazy-loading
# ---------------------------------------------------------------------------

def test_client_lazy_loaded_on_first_use():
    """_get_client() instantiates the Redis client only on first call."""
    with patch("redis.Redis") as MockRedis:
        mock_instance = MagicMock()
        mock_instance.get.return_value = None
        MockRedis.from_url.return_value = mock_instance

        backend = RedisBackend(FAKE_URL)
        assert backend._client is None  # not yet loaded

        backend.get(FAKE_EMBEDDING)
        MockRedis.from_url.assert_called_once_with(FAKE_URL, decode_responses=True)
        assert backend._client is mock_instance

        backend.get(FAKE_EMBEDDING)
        MockRedis.from_url.assert_called_once()  # still only once
