"""
Tests for thriftlm.embedder.Embedder.
"""

import math

import pytest

from thriftlm.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# Shared instance — model loads once across all tests in this module.
@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

def test_embed_returns_list(embedder):
    """embed() should return a plain Python list."""
    assert isinstance(embedder.embed("What is RAG?"), list)


def test_embed_dimension(embedder):
    """all-MiniLM-L6-v2 produces 384-dimensional embeddings."""
    assert len(embedder.embed("Hello world")) == 384


def test_embed_values_are_floats(embedder):
    """Every element in the embedding vector should be a float."""
    assert all(isinstance(v, float) for v in embedder.embed("Test query"))


def test_same_query_same_embedding(embedder):
    """Identical queries must produce identical embeddings (deterministic)."""
    a = embedder.embed("What is semantic caching?")
    b = embedder.embed("What is semantic caching?")
    assert a == b


def test_different_queries_different_embeddings(embedder):
    """Semantically different queries should produce different embeddings."""
    a = embedder.embed("What is the capital of France?")
    b = embedder.embed("How do I sort a list in Python?")
    assert a != b


def test_embedding_is_unit_vector(embedder):
    """Embeddings should be L2-normalized (magnitude ≈ 1.0)."""
    vec = embedder.embed("Normalization check")
    magnitude = math.sqrt(sum(x * x for x in vec))
    assert abs(magnitude - 1.0) < 1e-5


def test_model_loaded_only_once(embedder):
    """Calling embed() multiple times should not reload the model."""
    model_before = embedder._model
    embedder.embed("first call")
    embedder.embed("second call")
    assert embedder._model is model_before


# ---------------------------------------------------------------------------
# Semantic similarity tests
# ---------------------------------------------------------------------------

def test_similar_sentences_high_cosine_similarity(embedder):
    """Two semantically similar sentences should have cosine similarity > 0.8."""
    a = embedder.embed("How do I reset my password?")
    b = embedder.embed("What are the steps to change my password?")
    sim = cosine_similarity(a, b)
    assert sim > 0.8, f"Expected > 0.8, got {sim:.4f}"


def test_unrelated_sentences_low_cosine_similarity(embedder):
    """Two semantically unrelated sentences should have cosine similarity < 0.5."""
    a = embedder.embed("What is the boiling point of water?")
    b = embedder.embed("How do I reverse a linked list in Java?")
    sim = cosine_similarity(a, b)
    assert sim < 0.5, f"Expected < 0.5, got {sim:.4f}"
