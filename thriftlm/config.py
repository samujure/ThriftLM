"""
Configuration dataclass for SemanticCache.

Values can be set explicitly or read from environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Runtime configuration for the SemanticCache client.

    Attributes:
        api_key: Developer API key (sc_xxx) issued by the hosted service.
        threshold: Cosine similarity threshold for a cache hit (0–1).
        ttl: Time-to-live for cache entries in seconds.
        base_url: Base URL of the SemanticCache API backend.
    """

    api_key: str
    threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.85")))
    ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "86400")))
    base_url: str = field(default_factory=lambda: os.getenv("SEMANTIC_CACHE_URL", "https://api.semanticcache.dev"))
