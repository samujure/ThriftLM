"""
Upstash Redis backend for the fast cache layer.

Sits in front of Supabase to serve hot entries without a DB round-trip.
The cache key is a sha256 hash of the JSON-serialized embedding vector,
keeping keys short (~75 chars) regardless of embedding dimensionality.

Flow:
    query → RedisBackend.get(embedding)
        HIT  → return response immediately (skip Supabase)
        MISS → SupabaseBackend.lookup(embedding, api_key)
             → if result: RedisBackend.set(embedding, response)
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional


class RedisBackend:
    """Thin wrapper around redis-py for fast embedding-keyed cache operations.

    Args:
        redis_url: Connection URL — redis:// for plain TCP,
                   rediss:// for TLS (Upstash requires TLS in production).
        ttl: Default TTL in seconds for stored entries.
    """

    def __init__(self, redis_url: str, ttl: int = 86400) -> None:
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None  # lazy-loaded on first use

    def _get_client(self):
        """Return (or create) the Redis client (idempotent)."""
        if self._client is None:
            from redis import Redis
            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _make_key(self, embedding: list[float]) -> str:
        """Derive a short, stable cache key from an embedding vector.

        Serializes the embedding with compact JSON (no spaces), hashes it
        with sha256, and prefixes with 'sc:' for easy namespace inspection.

        Args:
            embedding: 384-dim float list.

        Returns:
            A string of the form 'sc:<64-char hex digest>' (67 chars total).
        """
        serialized = json.dumps(embedding, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"sc:{digest}"

    def get(self, embedding: list[float]) -> Optional[str]:
        """Retrieve a cached response by embedding vector.

        Args:
            embedding: 384-dim query embedding.

        Returns:
            The cached response string, or None if not found or expired.
        """
        key = self._make_key(embedding)
        return self._get_client().get(key)

    def set(self, embedding: list[float], response: str, ttl: Optional[int] = None) -> None:
        """Store a response string keyed by embedding vector.

        Args:
            embedding: 384-dim query embedding.
            response: LLM response string to cache.
            ttl: TTL in seconds. Falls back to instance default if not provided.
        """
        key = self._make_key(embedding)
        self._get_client().setex(key, ttl if ttl is not None else self.ttl, response)

    def delete(self, embedding: list[float]) -> None:
        """Remove a cached entry by embedding vector.

        Args:
            embedding: 384-dim query embedding whose cache entry should be removed.
        """
        key = self._make_key(embedding)
        self._get_client().delete(key)
