"""
Tests for thriftlm.backends.supabase_backend.SupabaseBackend.

The Supabase client is fully mocked — no real network or DB calls are made.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from thriftlm.backends.supabase_backend import SupabaseBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_URL = "https://fake.supabase.co"
FAKE_KEY = "fake-service-role-key"
FAKE_API_KEY = "sc_test123"
FAKE_EMBEDDING = [0.1] * 384
FAKE_RESPONSE = "Paris is the capital of France."
FAKE_ROW_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_backend(mock_client: MagicMock) -> SupabaseBackend:
    """Create a SupabaseBackend with its client pre-injected."""
    backend = SupabaseBackend(FAKE_URL, FAKE_KEY, threshold=0.85)
    backend._client = mock_client
    return backend


def _rpc_result(data: list) -> MagicMock:
    """Return a mock that simulates client.rpc(...).execute() -> data."""
    result = MagicMock()
    result.data = data
    execute = MagicMock(return_value=result)
    rpc = MagicMock()
    rpc.execute = execute
    return rpc


# ---------------------------------------------------------------------------
# lookup — HIT
# ---------------------------------------------------------------------------

def test_lookup_hit_returns_response():
    """lookup() returns the cached response string on a similarity hit."""
    mock_client = MagicMock()

    hit_row = {"id": FAKE_ROW_ID, "response": FAKE_RESPONSE, "similarity": 0.93, "hit_count": 2}

    # First rpc call (match_cache_entries) returns a hit row.
    # Second rpc call (increment_api_key_counters) is a fire-and-forget.
    mock_client.rpc.side_effect = [
        _rpc_result([hit_row]),   # match_cache_entries
        _rpc_result([]),           # increment_api_key_counters
    ]

    # table("cache_entries").update(...).eq(...).execute() — chain mock
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "fake-uuid"}])

    backend = _make_backend(mock_client)
    result = backend.lookup(FAKE_EMBEDDING, FAKE_API_KEY)

    assert result == FAKE_RESPONSE


def test_lookup_hit_updates_hit_count():
    """On a hit, cache_entries row is updated with incremented hit_count and last_hit_at."""
    mock_client = MagicMock()

    hit_row = {"id": FAKE_ROW_ID, "response": FAKE_RESPONSE, "similarity": 0.91, "hit_count": 5}

    mock_client.rpc.side_effect = [
        _rpc_result([hit_row]),
        _rpc_result([]),
    ]

    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "fake-uuid"}])

    backend = _make_backend(mock_client)
    backend.lookup(FAKE_EMBEDDING, FAKE_API_KEY)

    # update() should have been called with hit_count = old + 1 = 6
    update_call_args = mock_table.update.call_args[0][0]
    assert update_call_args["hit_count"] == 6
    assert "last_hit_at" in update_call_args


def test_lookup_hit_increments_total_hits_and_queries():
    """On a hit, increment_api_key_counters RPC is called with hits_delta=1, queries_delta=1."""
    mock_client = MagicMock()

    hit_row = {"id": FAKE_ROW_ID, "response": FAKE_RESPONSE, "similarity": 0.88, "hit_count": 0}

    mock_client.rpc.side_effect = [
        _rpc_result([hit_row]),
        _rpc_result([]),
    ]

    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "fake-uuid"}])

    backend = _make_backend(mock_client)
    backend.lookup(FAKE_EMBEDDING, FAKE_API_KEY)

    counter_call = mock_client.rpc.call_args_list[1]
    expected_tokens = len(FAKE_RESPONSE) // 4
    assert counter_call == call(
        "increment_api_key_counters",
        {"target_api_key": FAKE_API_KEY, "hits_delta": 1, "queries_delta": 1, "tokens_delta": expected_tokens},
    )


# ---------------------------------------------------------------------------
# lookup — MISS
# ---------------------------------------------------------------------------

def test_lookup_miss_returns_none():
    """lookup() returns None when no result meets the similarity threshold."""
    mock_client = MagicMock()

    mock_client.rpc.side_effect = [
        _rpc_result([]),   # match_cache_entries — empty
        _rpc_result([]),   # increment_api_key_counters
    ]

    backend = _make_backend(mock_client)
    result = backend.lookup(FAKE_EMBEDDING, FAKE_API_KEY)

    assert result is None


def test_lookup_miss_increments_total_queries_only():
    """On a miss, increment_api_key_counters is called with hits_delta=0, queries_delta=1."""
    mock_client = MagicMock()

    mock_client.rpc.side_effect = [
        _rpc_result([]),
        _rpc_result([]),
    ]

    backend = _make_backend(mock_client)
    backend.lookup(FAKE_EMBEDDING, FAKE_API_KEY)

    counter_call = mock_client.rpc.call_args_list[1]
    assert counter_call == call(
        "increment_api_key_counters",
        {"target_api_key": FAKE_API_KEY, "hits_delta": 0, "queries_delta": 1, "tokens_delta": 0},
    )


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_store_inserts_correct_data():
    """store() calls table('cache_entries').insert() with all required fields."""
    mock_client = MagicMock()

    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "fake-uuid"}])

    backend = _make_backend(mock_client)
    backend.store(
        query="What is the capital of France?",
        response=FAKE_RESPONSE,
        embedding=FAKE_EMBEDDING,
        api_key=FAKE_API_KEY,
    )

    mock_client.table.assert_called_once_with("cache_entries")
    insert_payload = mock_table.insert.call_args[0][0]
    assert insert_payload["api_key"] == FAKE_API_KEY
    assert insert_payload["query"] == "What is the capital of France?"
    assert insert_payload["response"] == FAKE_RESPONSE
    assert insert_payload["embedding"] == FAKE_EMBEDDING


# ---------------------------------------------------------------------------
# Client lazy-loading
# ---------------------------------------------------------------------------

def test_client_lazy_loaded_on_first_use():
    """_get_client() creates the Supabase client only once."""
    with patch("thriftlm.backends.supabase_backend.SupabaseBackend._get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[{"id": "fake-uuid"}])

        backend = SupabaseBackend(FAKE_URL, FAKE_KEY)
        backend.store("q", "r", FAKE_EMBEDDING, FAKE_API_KEY)

        mock_get.assert_called_once()
