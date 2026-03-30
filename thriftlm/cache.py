"""
Core SemanticCache class.

Orchestrates embedding, Redis fast-path, Supabase vector search,
LLM fallback, and write-back to both cache layers.

Cache lookup order:
    1. Redis (exact embedding hash, microseconds)
    2. Local numpy index (cosine similarity matmul, ~1ms) → Supabase PK fetch
    3. LLM function (cache miss, full latency)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from thriftlm.backends.local_index import LocalEmbeddingIndex
from thriftlm.backends.redis_backend import RedisBackend
from thriftlm.backends.supabase_backend import SupabaseBackend
from thriftlm.config import Config
from thriftlm.embedder import Embedder
from thriftlm.privacy import PIIScrubber


class SemanticCache:
    """Semantic caching layer for LLM calls.

    Reads SUPABASE_URL, SUPABASE_KEY, and REDIS_URL from environment
    variables at instantiation time and wires up the full cache stack.

    Args:
        api_key: Developer API key (sc_xxx) that namespaces all cache entries.
        threshold: Cosine similarity threshold for a Supabase hit (0–1).
        ttl: TTL for stored cache entries in seconds.
        base_url: Base URL of the SemanticCache hosted API (stored for future use).
    """

    def __init__(
        self,
        api_key: str,
        threshold: float = 0.85,
        ttl: int = 86400,
        base_url: Optional[str] = None,
    ) -> None:
        self.config = Config(
            api_key=api_key,
            threshold=threshold,
            ttl=ttl,
            base_url=base_url or os.getenv("THRIFTLM_URL", "https://api.thriftlm.dev"),
        )

        self._embedder = Embedder()
        self._scrubber = PIIScrubber(self._embedder)

        self._redis = RedisBackend(
            redis_url=os.environ["REDIS_URL"],
            ttl=ttl,
        )

        self._supabase = SupabaseBackend(
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=os.environ["SUPABASE_KEY"],
            threshold=threshold,
        )

        self._local_index = LocalEmbeddingIndex(
            supabase_client=self._supabase._get_client(),
            api_key=api_key,
        )

    def get_or_call(self, query: str, llm_fn: Callable[[str], str]) -> str:
        """Return a cached response or call the LLM and cache the result.

        Full flow:
            1. Embed the raw query (no scrubbing — preserves semantic fidelity).
            2. Check Redis — return cached response immediately on hit.
            3. Check Supabase — write to Redis and return cached response on hit.
            4. On miss: call llm_fn with the original query.
            5. Scrub PII from the response only; store the clean response.
            6. Return the original (unscrubbed) response to the caller.

        Args:
            query: The raw user query string.
            llm_fn: Callable that accepts the query and returns an LLM response string.

        Returns:
            The LLM response, from cache or freshly generated. Only response
            PII is scrubbed before storage; query embeddings use raw text for
            accurate semantic matching.
        """
        embedding = self._embedder.embed(query)

        # 1. Redis fast-path (exact embedding match — sub-millisecond)
        cached = self._redis.get(embedding)
        if cached is not None:
            self._supabase.record_hit(self.config.api_key, cached)
            return cached

        # 2. Local numpy index (microseconds) → Supabase PK fetch on hit
        row_id = self._local_index.search(embedding, self.config.threshold)
        if row_id is not None:
            cached = self._supabase.fetch_response_by_id(row_id, self.config.api_key)
            if cached is not None:
                self._redis.set(embedding, cached)
                return cached

        # 3. LLM fallback
        self._supabase.record_miss(self.config.api_key)
        response = llm_fn(query)
        clean_response = self._scrubber.scrub(response)
        row_id = self._supabase.store(query, clean_response, embedding, self.config.api_key)
        self._local_index.add(row_id, embedding)
        self._redis.set(embedding, clean_response)
        return response  # original, not clean_response

    def lookup(self, query: str) -> Optional[str]:
        """Check both cache layers for a semantically similar response.

        Does not call any LLM function. Promotes a Supabase hit into Redis.
        The raw query is embedded directly — no scrubbing — to preserve
        semantic fidelity for vector comparison.

        Args:
            query: The raw user query string.

        Returns:
            The cached response string on a hit, None on a miss.
        """
        embedding = self._embedder.embed(query)

        cached = self._redis.get(embedding)
        if cached is not None:
            return cached

        row_id = self._local_index.search(embedding, self.config.threshold)
        if row_id is not None:
            cached = self._supabase.fetch_response_by_id(row_id, self.config.api_key)
            if cached is not None:
                self._redis.set(embedding, cached)
                return cached

        return None

    def store(self, query: str, response: str) -> None:
        """Embed the raw query and store the scrubbed response in both backends.

        The query is embedded without scrubbing to preserve semantic fidelity.
        Only the response is scrubbed for PII before storage.

        Args:
            query: The raw user query string.
            response: The LLM response to cache.
        """
        embedding = self._embedder.embed(query)
        clean_response = self._scrubber.scrub(response)
        row_id = self._supabase.store(query, clean_response, embedding, self.config.api_key)
        self._local_index.add(row_id, embedding)
        self._redis.set(embedding, clean_response)
