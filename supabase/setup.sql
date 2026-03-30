-- =============================================================================
-- ThriftLM — Supabase schema setup
-- Run this once against your Supabase project (SQL editor or psql).
-- =============================================================================

-- Enable pgvector extension (requires Supabase project with pgvector support).
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- cache_entries
-- One row per stored (query, response) pair, namespaced by api_key.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cache_entries (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    api_key     TEXT        NOT NULL,
    query       TEXT        NOT NULL,
    response    TEXT        NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_hit_at TIMESTAMP WITH TIME ZONE,
    hit_count   INTEGER     DEFAULT 0
);

-- HNSW index for cosine similarity search.
-- Better recall than IVFFlat at any dataset size — no probes tuning needed.
CREATE INDEX IF NOT EXISTS cache_entries_embedding_idx
    ON cache_entries
    USING hnsw (embedding vector_cosine_ops);

-- Index to speed up per-api_key filtering.
CREATE INDEX IF NOT EXISTS cache_entries_api_key_idx
    ON cache_entries (api_key);


-- ---------------------------------------------------------------------------
-- api_keys
-- One row per developer. Tracks aggregate metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_keys (
    api_key       TEXT PRIMARY KEY,
    email         TEXT,
    total_queries INTEGER DEFAULT 0,
    total_hits    INTEGER DEFAULT 0,
    tokens_saved  INTEGER DEFAULT 0,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- ---------------------------------------------------------------------------
-- RPC: match_cache_entries
-- Called by SupabaseBackend.lookup() to find the nearest cached embedding.
-- Returns at most 1 row whose cosine similarity >= match_threshold.
-- ---------------------------------------------------------------------------
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
        (1 - (embedding <=> query_embedding))::float AS similarity
    FROM cache_entries
    WHERE
        api_key = match_api_key
        AND (1 - (embedding <=> query_embedding)) >= match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT 1;
$$;


-- ---------------------------------------------------------------------------
-- RPC: increment_api_key_counters
-- Called by SupabaseBackend.lookup() to update hit/query counters atomically.
-- hits_delta is 1 on a cache hit, 0 on a miss.
-- tokens_delta is estimated tokens saved (len(response) // 4) on a hit, 0 on a miss.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION increment_api_key_counters(
    target_api_key text,
    hits_delta     integer,
    queries_delta  integer,
    tokens_delta   integer DEFAULT 0
)
RETURNS void
LANGUAGE sql
AS $$
    INSERT INTO api_keys (api_key, total_hits, total_queries, tokens_saved)
    VALUES (target_api_key, hits_delta, queries_delta, tokens_delta)
    ON CONFLICT (api_key) DO UPDATE
        SET total_hits    = api_keys.total_hits    + EXCLUDED.total_hits,
            total_queries = api_keys.total_queries + EXCLUDED.total_queries,
            tokens_saved  = api_keys.tokens_saved  + EXCLUDED.tokens_saved;
$$;
