"""
Integration tests for the FastAPI API layer.

Uses httpx AsyncClient with ASGI transport — no real server, DB, or Redis needed.
All infrastructure singletons (SupabaseBackend, RedisBackend, Supabase client)
are replaced via FastAPI dependency overrides so tests are fully isolated.
"""

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app
from api.db import get_redis_backend, get_supabase_backend, get_supabase_client

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_API_KEY = "sc_testapikey123"
FAKE_EMBEDDING = [0.1] * 384
FAKE_RESPONSE = "Paris is the capital of France."
FAKE_QUERY = "What is the capital of France?"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> MagicMock:
    m = MagicMock()
    m.get.return_value = None
    return m


@pytest.fixture
def mock_supabase_backend() -> MagicMock:
    m = MagicMock()
    m.lookup.return_value = None
    return m


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
async def client(mock_redis, mock_supabase_backend, mock_supabase_client):
    """AsyncClient with all infrastructure dependencies overridden."""
    app.dependency_overrides[get_redis_backend] = lambda: mock_redis
    app.dependency_overrides[get_supabase_backend] = lambda: mock_supabase_backend
    app.dependency_overrides[get_supabase_client] = lambda: mock_supabase_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health(client):
    """GET /health returns 200 and status ok."""
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /lookup — schema validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_requires_api_key(client):
    """POST /lookup without api_key returns 422."""
    r = await client.post("/lookup", json={"embedding": FAKE_EMBEDDING})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_lookup_requires_embedding(client):
    """POST /lookup without embedding returns 422."""
    r = await client.post("/lookup", json={"api_key": FAKE_API_KEY})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /lookup — behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_redis_hit(client, mock_redis):
    """Redis hit: returns the cached response without touching Supabase."""
    mock_redis.get.return_value = FAKE_RESPONSE

    r = await client.post(
        "/lookup", json={"embedding": FAKE_EMBEDDING, "api_key": FAKE_API_KEY}
    )

    assert r.status_code == 200
    assert r.json() == {"response": FAKE_RESPONSE}


@pytest.mark.asyncio
async def test_lookup_supabase_hit(client, mock_redis, mock_supabase_backend):
    """Supabase hit: returns cached response and writes it back to Redis."""
    mock_redis.get.return_value = None
    mock_supabase_backend.lookup.return_value = FAKE_RESPONSE

    r = await client.post(
        "/lookup", json={"embedding": FAKE_EMBEDDING, "api_key": FAKE_API_KEY}
    )

    assert r.status_code == 200
    assert r.json() == {"response": FAKE_RESPONSE}
    mock_redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


@pytest.mark.asyncio
async def test_lookup_miss(client, mock_redis, mock_supabase_backend):
    """Both miss: returns {response: null}."""
    mock_redis.get.return_value = None
    mock_supabase_backend.lookup.return_value = None

    r = await client.post(
        "/lookup", json={"embedding": FAKE_EMBEDDING, "api_key": FAKE_API_KEY}
    )

    assert r.status_code == 200
    assert r.json() == {"response": None}


# ---------------------------------------------------------------------------
# POST /store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_requires_fields(client):
    """POST /store without required fields returns 422."""
    r = await client.post("/store", json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_store_writes_both_backends(client, mock_redis, mock_supabase_backend):
    """POST /store calls supabase.store() and redis.set() with correct args."""
    r = await client.post(
        "/store",
        json={
            "embedding": FAKE_EMBEDDING,
            "query": FAKE_QUERY,
            "response": FAKE_RESPONSE,
            "api_key": FAKE_API_KEY,
        },
    )

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    mock_supabase_backend.store.assert_called_once_with(
        FAKE_QUERY, FAKE_RESPONSE, FAKE_EMBEDDING, FAKE_API_KEY
    )
    mock_redis.set.assert_called_once_with(FAKE_EMBEDDING, FAKE_RESPONSE)


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_requires_api_key_header(client):
    """GET /metrics without X-API-Key header returns 422."""
    r = await client.get("/metrics")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_metrics_returns_computed_values(client, mock_supabase_client):
    """GET /metrics returns correct hit_rate and cost_saved calculations."""
    mock_row = MagicMock()
    mock_row.data = {"total_hits": 80, "total_queries": 100, "tokens_saved": 5000}
    mock_supabase_client.table.return_value.select.return_value \
        .eq.return_value.maybe_single.return_value.execute.return_value = mock_row

    r = await client.get("/metrics", headers={"X-API-Key": FAKE_API_KEY})

    assert r.status_code == 200
    body = r.json()
    assert body["hit_rate"] == pytest.approx(0.80)
    assert body["total_queries"] == 100
    assert body["tokens_saved"] == 5000
    assert body["cost_saved"] == pytest.approx(5000 * 0.000002)


@pytest.mark.asyncio
async def test_metrics_zero_queries_hit_rate(client, mock_supabase_client):
    """hit_rate is 0.0 when total_queries is 0 (no division by zero)."""
    mock_row = MagicMock()
    mock_row.data = {"total_hits": 0, "total_queries": 0, "tokens_saved": 0}
    mock_supabase_client.table.return_value.select.return_value \
        .eq.return_value.maybe_single.return_value.execute.return_value = mock_row

    r = await client.get("/metrics", headers={"X-API-Key": FAKE_API_KEY})
    assert r.status_code == 200
    assert r.json()["hit_rate"] == 0.0


@pytest.mark.asyncio
async def test_metrics_unknown_key_returns_404(client, mock_supabase_client):
    """GET /metrics for an unregistered key returns 404."""
    mock_row = MagicMock()
    mock_row.data = None
    mock_supabase_client.table.return_value.select.return_value \
        .eq.return_value.maybe_single.return_value.execute.return_value = mock_row

    r = await client.get("/metrics", headers={"X-API-Key": "sc_unknown"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_key_requires_email(client):
    """POST /keys without email returns 422."""
    r = await client.post("/keys", json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_key_invalid_email(client):
    """POST /keys with a non-email string returns 422."""
    r = await client.post("/keys", json={"email": "notanemail"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_key_returns_sc_prefixed_key(client, mock_supabase_client):
    """POST /keys returns a new sc_-prefixed API key and inserts it into DB."""
    # No existing key for this email.
    no_existing = MagicMock()
    no_existing.data = None

    insert_result = MagicMock()
    insert_result.data = [{}]

    (
        mock_supabase_client.table.return_value
        .select.return_value
        .eq.return_value
        .maybe_single.return_value
        .execute.return_value
    ) = no_existing
    (
        mock_supabase_client.table.return_value
        .insert.return_value
        .execute.return_value
    ) = insert_result

    r = await client.post("/keys", json={"email": "dev@example.com"})

    assert r.status_code == 201
    body = r.json()
    assert "api_key" in body
    assert body["api_key"].startswith("sc_")
    assert len(body["api_key"]) == 35  # "sc_" + 32 hex chars


@pytest.mark.asyncio
async def test_create_key_duplicate_email_returns_409(client, mock_supabase_client):
    """POST /keys for an already-registered email returns 409."""
    existing = MagicMock()
    existing.data = {"api_key": "sc_existingkey"}
    (
        mock_supabase_client.table.return_value
        .select.return_value
        .eq.return_value
        .maybe_single.return_value
        .execute.return_value
    ) = existing

    r = await client.post("/keys", json={"email": "existing@example.com"})
    assert r.status_code == 409
