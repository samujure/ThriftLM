"""
Tests for thriftlm.privacy.PIIScrubber.

Both the Presidio AnalyzerEngine and the SBERT model are fully mocked —
no model downloads or real network calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from thriftlm.privacy import PIIScrubber

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THRESHOLD = 0.95


def _make_scrubber(mock_embedder=None) -> PIIScrubber:
    """Return a PIIScrubber with a pre-configured mock Embedder."""
    if mock_embedder is None:
        mock_embedder = MagicMock()
    return PIIScrubber(embedder=mock_embedder, threshold=THRESHOLD)


def _presidio_result(entity_type: str, start: int, end: int) -> MagicMock:
    """Build a fake Presidio RecognizerResult."""
    r = MagicMock()
    r.entity_type = entity_type
    r.start = start
    r.end = end
    return r


def _inject_analyzer(scrubber: PIIScrubber, results: list) -> MagicMock:
    """Inject a mock AnalyzerEngine whose analyze() returns `results`."""
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = results
    scrubber._analyzer = mock_analyzer
    return mock_analyzer


def _make_batch_encode(full_emb: np.ndarray, candidate_sims: list[float]):
    """
    Return a mock encode() function that produces controlled embeddings.

    index 0  → full_emb (unit vector)
    index 1+ → unit vectors whose dot product with full_emb equals candidate_sims[i]
    """
    dim = len(full_emb)
    rows = [full_emb]
    for sim in candidate_sims:
        # Build a vector v such that v · full_emb = sim and |v| = 1.
        # Decompose: v = sim * full_emb + sqrt(1 - sim²) * perp
        perp = np.zeros(dim)
        perp[1] = 1.0  # perpendicular to full_emb (assuming full_emb[1] ≈ 0)
        perp = perp - np.dot(perp, full_emb) * full_emb
        perp /= np.linalg.norm(perp)
        v = sim * full_emb + np.sqrt(max(1 - sim**2, 0.0)) * perp
        v /= np.linalg.norm(v)
        rows.append(v)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pass 1 — Presidio: explicit PII
# ---------------------------------------------------------------------------

class TestPresidioPass:
    def test_scrubs_person_name(self):
        """A PERSON entity is replaced with [PERSON]."""
        scrubber = _make_scrubber()
        text = "Alice booked a flight."
        #       01234
        _inject_analyzer(scrubber, [_presidio_result("PERSON", 0, 5)])

        result = scrubber._presidio_pass(text)

        assert "[PERSON]" in result
        assert "Alice" not in result

    def test_scrubs_email(self):
        """An EMAIL_ADDRESS entity is replaced with [EMAIL_ADDRESS]."""
        scrubber = _make_scrubber()
        text = "Contact support@example.com for help."
        #                8          27
        _inject_analyzer(scrubber, [_presidio_result("EMAIL_ADDRESS", 8, 27)])

        result = scrubber._presidio_pass(text)

        assert "[EMAIL_ADDRESS]" in result
        assert "support@example.com" not in result

    def test_scrubs_phone_number(self):
        """A PHONE_NUMBER entity is replaced with [PHONE_NUMBER]."""
        scrubber = _make_scrubber()
        text = "Call me at 415-555-0198 anytime."
        _inject_analyzer(scrubber, [_presidio_result("PHONE_NUMBER", 11, 23)])

        result = scrubber._presidio_pass(text)

        assert "[PHONE_NUMBER]" in result
        assert "415-555-0198" not in result

    def test_scrubs_multiple_entities_in_one_pass(self):
        """Multiple entities in the same text are all replaced correctly."""
        scrubber = _make_scrubber()
        text = "Bob called 555-1234 from bob@corp.com."
        results = [
            _presidio_result("PERSON", 0, 3),
            _presidio_result("PHONE_NUMBER", 11, 19),
            _presidio_result("EMAIL_ADDRESS", 25, 37),
        ]
        _inject_analyzer(scrubber, results)

        result = scrubber._presidio_pass(text)

        assert "Bob" not in result
        assert "555-1234" not in result
        assert "bob@corp.com" not in result
        assert "[PERSON]" in result
        assert "[PHONE_NUMBER]" in result
        assert "[EMAIL_ADDRESS]" in result

    def test_no_pii_returns_text_unchanged(self):
        """Text with no detected entities passes through unmodified."""
        scrubber = _make_scrubber()
        text = "The sky is blue."
        _inject_analyzer(scrubber, [])

        assert scrubber._presidio_pass(text) == text


# ---------------------------------------------------------------------------
# Pass 2 — SBERT: implicit identifiers
# ---------------------------------------------------------------------------

class TestSbertPass:
    def _setup(self, tokens: list[str], candidate_sims: list[float]) -> PIIScrubber:
        """Wire up a PIIScrubber whose mock model.encode returns controlled embeddings."""
        full_emb = np.zeros(384, dtype=np.float32)
        full_emb[0] = 1.0  # unit vector along axis 0

        batch_result = _make_batch_encode(full_emb, candidate_sims)

        mock_model = MagicMock()
        mock_model.encode.return_value = batch_result

        mock_embedder = MagicMock()
        mock_embedder._model = mock_model

        scrubber = _make_scrubber(mock_embedder)
        return scrubber

    def test_masks_implicit_identifier_room_number(self):
        """An implicit room/account number (high leave-one-out similarity) is masked."""
        # "Patient in room 314 needs medication"
        # 6 tokens → 6 candidates. Token "314" is at index 3.
        tokens = ["Patient", "in", "room", "314", "needs", "medication"]
        # Similarity when each token is removed:
        # Only token 3 ("314") removal keeps sim >= 0.95.
        sims = [0.60, 0.70, 0.65, 0.97, 0.55, 0.50]

        scrubber = self._setup(tokens, sims)
        text = " ".join(tokens)

        result = scrubber._sbert_pass(text)

        assert "314" not in result
        assert "[REDACTED]" in result
        # Content words should be preserved
        assert "Patient" in result
        assert "medication" in result

    def test_masks_account_reference(self):
        """An account reference code with high leave-one-out sim is masked."""
        tokens = ["Your", "account", "ACC-7891", "is", "active"]
        sims = [0.80, 0.72, 0.96, 0.85, 0.78]

        scrubber = self._setup(tokens, sims)
        result = scrubber._sbert_pass(" ".join(tokens))

        assert "ACC-7891" not in result
        assert "[REDACTED]" in result

    def test_preserves_content_tokens(self):
        """Tokens whose removal significantly changes meaning are kept."""
        tokens = ["The", "diagnosis", "is", "critical"]
        # "critical" is semantically heavy — removing it drops similarity.
        sims = [0.92, 0.45, 0.90, 0.30]

        scrubber = self._setup(tokens, sims)
        result = scrubber._sbert_pass(" ".join(tokens))

        assert "diagnosis" in result
        assert "critical" in result

    def test_single_token_returned_unchanged(self):
        """A single-token string bypasses the SBERT pass entirely."""
        mock_embedder = MagicMock()
        scrubber = _make_scrubber(mock_embedder)
        assert scrubber._sbert_pass("Hello") == "Hello"
        mock_embedder._model.encode.assert_not_called()

    def test_batch_encode_called_once(self):
        """All candidates are encoded in exactly one model.encode() call."""
        tokens = ["foo", "bar", "baz"]
        sims = [0.50, 0.50, 0.50]
        scrubber = self._setup(tokens, sims)

        scrubber._sbert_pass(" ".join(tokens))

        # One encode call, with len(tokens)+1 texts.
        mock_encode = scrubber._embedder._model.encode
        mock_encode.assert_called_once()
        texts_arg = mock_encode.call_args[0][0]
        assert len(texts_arg) == len(tokens) + 1  # full text + N candidates


# ---------------------------------------------------------------------------
# scrub() — both passes in order
# ---------------------------------------------------------------------------

class TestScrub:
    def test_presidio_then_sbert(self):
        """scrub() calls _presidio_pass then _sbert_pass in that order."""
        scrubber = _make_scrubber()
        call_order = []

        scrubber._presidio_pass = MagicMock(
            side_effect=lambda t: (call_order.append("presidio"), t)[1]
        )
        scrubber._sbert_pass = MagicMock(
            side_effect=lambda t: (call_order.append("sbert"), t)[1]
        )

        scrubber.scrub("some text")

        assert call_order == ["presidio", "sbert"]

    def test_sbert_receives_presidio_output(self):
        """The output of _presidio_pass is fed directly into _sbert_pass."""
        scrubber = _make_scrubber()
        scrubber._presidio_pass = MagicMock(return_value="[PERSON] asked a question")
        scrubber._sbert_pass = MagicMock(return_value="[PERSON] asked a question")

        scrubber.scrub("Alice asked a question")

        scrubber._sbert_pass.assert_called_once_with("[PERSON] asked a question")


# ---------------------------------------------------------------------------
# cache.py integration — original response returned to caller
# ---------------------------------------------------------------------------

class TestOriginalResponseReturned:
    """Verify that SemanticCache.get_or_call returns the original response
    even though the scrubbed version is what gets stored in both backends."""

    def _make_cache(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        with patch("thriftlm.cache.Embedder") as MockEmb, \
             patch("thriftlm.cache.RedisBackend") as MockRedis, \
             patch("thriftlm.cache.SupabaseBackend") as MockSupa, \
             patch("thriftlm.cache.PIIScrubber") as MockScrubber:

            mock_emb = MagicMock()
            mock_emb.embed.return_value = [0.1] * 384
            MockEmb.return_value = mock_emb

            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            MockRedis.return_value = mock_redis

            mock_supa = MagicMock()
            mock_supa.lookup.return_value = None
            MockSupa.return_value = mock_supa

            # Scrubber transforms responses — strips names from output.
            mock_scrubber = MagicMock()
            mock_scrubber.scrub.side_effect = lambda t: t.replace("Alice", "[PERSON]")
            MockScrubber.return_value = mock_scrubber

            from importlib import reload
            import thriftlm.cache as cache_module
            reload(cache_module)

            sc = cache_module.SemanticCache(api_key="sc_test")
            sc._embedder = mock_emb
            sc._redis = mock_redis
            sc._supabase = mock_supa
            sc._scrubber = mock_scrubber
            yield sc

    def test_original_response_returned_to_caller(self, monkeypatch):
        """Caller receives the original LLM response even if scrub() modifies it."""
        original_response = "Alice's account balance is $500."
        scrubbed_response = "[PERSON]'s account balance is $500."

        for sc in self._make_cache(monkeypatch):
            llm_fn = MagicMock(return_value=original_response)

            result = sc.get_or_call("What is Alice's balance?", llm_fn)

            # Caller gets the original.
            assert result == original_response
            assert result != scrubbed_response

            # But the scrubbed version is what was stored.
            stored_response = sc._supabase.store.call_args[0][1]
            assert stored_response == scrubbed_response

    def test_scrubbed_response_stored_in_redis(self, monkeypatch):
        """Redis receives the scrubbed response, not the original."""
        original_response = "Contact Bob at bob@corp.com."
        scrubbed_response = "Contact [PERSON] at [PERSON]@corp.com."

        for sc in self._make_cache(monkeypatch):
            sc._scrubber.scrub.side_effect = lambda t: t.replace(
                "Bob", "[PERSON]"
            ).replace("bob", "[PERSON]")
            llm_fn = MagicMock(return_value=original_response)

            result = sc.get_or_call("How do I reach Bob?", llm_fn)

            assert result == original_response
            redis_value = sc._redis.set.call_args[0][1]
            assert "Bob" not in redis_value
            assert "bob" not in redis_value
