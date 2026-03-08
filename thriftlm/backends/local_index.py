"""
LocalEmbeddingIndex — in-process numpy similarity search.

Loads all embeddings for a given api_key from Supabase on init, stores
them as a (N, 384) float32 matrix, and performs similarity search via a
single matrix-vector multiply. Supabase is used only for durable storage
and response retrieval by primary key — no vector search RPC needed.

Lookup latency: ~1ms (numpy matmul) vs ~500ms (Supabase pgvector RPC).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


class LocalEmbeddingIndex:
    """In-memory cosine similarity index backed by a numpy float32 matrix.

    Args:
        supabase_client: Live Supabase client used to fetch existing entries.
        api_key: Developer API key — scopes the index to one tenant.
    """

    _DIMS = 384

    def __init__(self, supabase_client, api_key: str) -> None:
        self._client = supabase_client
        self._api_key = api_key
        self._ids: list[str] = []
        self._matrix = np.empty((0, self._DIMS), dtype=np.float32)
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Bulk-fetch all embeddings for this api_key from Supabase."""
        rows = (
            self._client.table("cache_entries")
            .select("id, embedding")
            .eq("api_key", self._api_key)
            .execute()
            .data
        )
        if rows:
            self._ids = [r["id"] for r in rows]
            # pgvector returns the embedding column as a JSON string; parse it.
            embs = [
                json.loads(r["embedding"]) if isinstance(r["embedding"], str)
                else r["embedding"]
                for r in rows
            ]
            self._matrix = np.array(embs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query_vec: list[float], threshold: float) -> Optional[str]:
        """Return the Supabase row id of the best match, or None.

        Computes cosine similarity as a dot product (valid because both
        stored and query embeddings are L2-normalised unit vectors).

        Args:
            query_vec: 384-dim unit-normalised query embedding.
            threshold: Minimum cosine similarity to count as a hit.

        Returns:
            The UUID string of the best matching cache_entries row,
            or None if no entry meets the threshold.
        """
        if len(self._ids) == 0:
            return None

        q = np.array(query_vec, dtype=np.float32)
        scores = self._matrix @ q  # (N,)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) >= threshold:
            return self._ids[best_idx]
        return None

    def add(self, row_id: str, embedding: list[float]) -> None:
        """Append a new entry to the in-memory index.

        Args:
            row_id: UUID of the newly stored cache_entries row.
            embedding: 384-dim unit-normalised embedding for that row.
        """
        vec = np.array(embedding, dtype=np.float32).reshape(1, self._DIMS)
        self._matrix = np.vstack([self._matrix, vec])
        self._ids.append(row_id)
