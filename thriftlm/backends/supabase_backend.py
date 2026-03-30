"""
Supabase (pgvector) backend for semantic similarity lookups and storage.

Uses cosine similarity on 384-dim SBERT embeddings stored in the
cache_entries table via a PostgreSQL RPC function.

Required SQL RPC function (see supabase/setup.sql for full schema):

    CREATE OR REPLACE FUNCTION match_cache_entries(
        query_embedding vector(384),
        match_api_key   text,
        match_threshold float
    )
    RETURNS TABLE (id uuid, response text, similarity float)
    LANGUAGE sql STABLE
    AS $$
        SELECT
            id,
            response,
            1 - (embedding <=> query_embedding) AS similarity
        FROM cache_entries
        WHERE
            api_key = match_api_key
            AND 1 - (embedding <=> query_embedding) >= match_threshold
        ORDER BY embedding <=> query_embedding
        LIMIT 1;
    $$;
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


class SupabaseBackend:
    """Manages vector similarity lookups and cache entry storage via Supabase pgvector.

    Args:
        supabase_url: Supabase project URL.
        supabase_key: Supabase service role key.
        threshold: Cosine similarity threshold for a cache hit (0–1).
    """

    def __init__(self, supabase_url: str, supabase_key: str, threshold: float = 0.85) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.threshold = threshold
        self._client = None  # lazy-loaded on first use

    def _get_client(self):
        """Return (or create) the Supabase client (idempotent)."""
        if self._client is None:
            from supabase import create_client
            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client

    def lookup(self, embedding: list[float], api_key: str) -> Optional[str]:
        """Find the most similar cached response for the given embedding.

        Calls the match_cache_entries RPC function which performs a cosine
        similarity search (via pgvector's <=> operator) within the api_key
        namespace and returns at most one result above threshold.

        On a HIT the matched row's hit_count and last_hit_at are updated,
        and total_hits + total_queries are incremented in api_keys.
        On a MISS only total_queries is incremented in api_keys.

        Args:
            embedding: 384-dim unit-normalized query embedding.
            api_key: Developer API key used to namespace the search.

        Returns:
            The cached response string on a hit, None on a miss.
        """
        client = self._get_client()

        result = (
            client.rpc(
                "match_cache_entries",
                {
                    "query_embedding": embedding,
                    "match_api_key": api_key,
                    "match_threshold": self.threshold,
                },
            )
            .execute()
        )

        rows = result.data

        if rows:
            hit = rows[0]

            # Update the matched row: bump hit_count and last_hit_at.
            now = datetime.now(timezone.utc).isoformat()
            (
                client.table("cache_entries")
                .update({"hit_count": hit.get("hit_count", 0) + 1, "last_hit_at": now})
                .eq("id", hit["id"])
                .execute()
            )

            # Increment total_hits, total_queries, and tokens_saved for this API key.
            response_text = hit["response"]
            tokens_delta = len(response_text) // 4
            (
                client.rpc(
                    "increment_api_key_counters",
                    {
                        "target_api_key": api_key,
                        "hits_delta": 1,
                        "queries_delta": 1,
                        "tokens_delta": tokens_delta,
                    },
                )
                .execute()
            )

            return response_text

        # MISS — increment total_queries only.
        (
            client.rpc(
                "increment_api_key_counters",
                {"target_api_key": api_key, "hits_delta": 0, "queries_delta": 1, "tokens_delta": 0},
            )
            .execute()
        )

        return None

    def fetch_response_by_id(self, row_id: str, api_key: str) -> Optional[str]:
        """Fetch a cached response by primary key and update hit metrics.

        Called after LocalEmbeddingIndex.search() returns a row id. Performs
        a fast PK lookup (no vector math) and bumps hit counters.

        Args:
            row_id: UUID of the cache_entries row to fetch.
            api_key: Developer API key (used for metrics update).

        Returns:
            The cached response string, or None if the row is not found.
        """
        client = self._get_client()

        result = (
            client.table("cache_entries")
            .select("response,hit_count")
            .eq("id", row_id)
            .single()
            .execute()
        )

        row = result.data
        if not row:
            return None

        # Update hit metrics on the matched row.
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        (
            client.table("cache_entries")
            .update({"last_hit_at": now, "hit_count": (row.get("hit_count") or 0) + 1})
            .eq("id", row_id)
            .execute()
        )
        response_text = row["response"]
        tokens_delta = len(response_text) // 4
        (
            client.rpc(
                "increment_api_key_counters",
                {
                    "target_api_key": api_key,
                    "hits_delta": 1,
                    "queries_delta": 1,
                    "tokens_delta": tokens_delta,
                },
            )
            .execute()
        )

        return response_text

    def record_hit(self, api_key: str, response_text: str) -> None:
        """Increment hit + query counters for a Redis cache hit."""
        tokens_delta = len(response_text) // 4
        self._get_client().rpc(
            "increment_api_key_counters",
            {
                "target_api_key": api_key,
                "hits_delta": 1,
                "queries_delta": 1,
                "tokens_delta": tokens_delta,
            },
        ).execute()

    def record_miss(self, api_key: str) -> None:
        """Increment query counter for a cache miss (before LLM call)."""
        self._get_client().rpc(
            "increment_api_key_counters",
            {"target_api_key": api_key, "hits_delta": 0, "queries_delta": 1, "tokens_delta": 0},
        ).execute()

    def store(self, query: str, response: str, embedding: list[float], api_key: str) -> str:
        """Insert a new cache entry and return its generated UUID.

        Args:
            query: Original raw query string.
            response: LLM response to cache.
            embedding: 384-dim unit-normalized query embedding.
            api_key: Developer API key used to namespace the entry.

        Returns:
            The UUID string of the newly inserted row.
        """
        client = self._get_client()

        result = (
            client.table("cache_entries")
            .insert(
                {
                    "api_key": api_key,
                    "query": query,
                    "response": response,
                    "embedding": embedding,
                }
            )
            .execute()
        )
        return result.data[0]["id"]
