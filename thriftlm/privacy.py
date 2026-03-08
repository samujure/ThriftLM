"""
PIIScrubber — two-pass PII sanitization for queries and responses.

Pass 1 (Presidio): Detects explicit PII (names, emails, phones, IDs, …)
    using presidio-analyzer backed by spacy en_core_web_trf. Entities are
    replaced in-place with [ENTITY_TYPE] placeholders.

Pass 2 (SBERT leave-one-out): For each whitespace token, computes the
    cosine similarity between the full-text embedding and the embedding of
    the text with that token removed — all in a single batch encode call.
    Tokens whose removal still yields similarity >= threshold are flagged as
    low-information identifiers (room numbers, account codes, numeric refs)
    and replaced with [REDACTED].

Design notes:
    - Both the Presidio AnalyzerEngine and the SBERT model are lazy-loaded;
      importing this module is free.
    - The shared Embedder instance is passed in from SemanticCache so the
      model is loaded at most once across the whole application.
    - Pass 2 uses simple whitespace tokenization. Common stop-words may also
      score above threshold (their removal doesn't change semantics much).
      In practice Pass 1 handles high-value explicit PII; Pass 2 is a safety
      net for opaque identifiers Presidio cannot name-match.

Setup requirements:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_trf
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import re

from thriftlm.embedder import Embedder


def _looks_like_identifier(token: str) -> bool:
    """Return True if the token structurally resembles an opaque identifier."""
    # Skip tokens that are already Presidio placeholders
    if token.startswith("[") and token.endswith("]"):
        return False
    # Pure digits (e.g. account numbers, zip codes, room numbers)
    if re.fullmatch(r"\d{4,}", token):
        return True
    # Alphanumeric codes: mix of letters and digits (e.g. X7291, abc123, A1B2C3)
    if re.fullmatch(r"[A-Za-z0-9]{4,}", token) and re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
        return True
    # Hyphenated codes (e.g. room-4B, ref-9921, ACC-00123)
    if re.fullmatch(r"[A-Za-z0-9]+-[A-Za-z0-9]+", token):
        return True
    # UUID-like segments
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", token):
        return True
    return False

# Explicit PII entity list — excludes DATE_TIME, URL, LOCATION, NRP which
# false-positive on technical content (model names, API endpoints, log timestamps).
PII_ENTITIES = [
    # Global
    "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE",
    "IP_ADDRESS", "MAC_ADDRESS", "PERSON",
    "PHONE_NUMBER", "MEDICAL_LICENSE",
    # USA
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_MBI",
    "US_PASSPORT", "US_SSN",
    # UK
    "UK_NHS", "UK_NINO",
    # Spain
    "ES_NIF", "ES_NIE",
    # Italy
    "IT_FISCAL_CODE", "IT_DRIVER_LICENSE", "IT_VAT_CODE",
    "IT_PASSPORT", "IT_IDENTITY_CARD",
    # Poland
    "PL_PESEL",
    # Singapore
    "SG_NRIC_FIN", "SG_UEN",
    # Australia
    "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
    # India
    "IN_PAN", "IN_AADHAAR", "IN_VEHICLE_REGISTRATION",
    "IN_VOTER", "IN_PASSPORT", "IN_GSTIN",
    # Finland
    "FI_PERSONAL_IDENTITY_CODE",
    # Korea
    "KR_DRIVER_LICENSE", "KR_FRN", "KR_PASSPORT", "KR_BRN", "KR_RRN",
    # Thailand
    "TH_TNIN",
]


class PIIScrubber:
    """Two-pass PII scrubber: Presidio NER + SBERT leave-one-out masking.

    Args:
        embedder: Shared :class:`Embedder` instance. Its underlying
            sentence-transformers model is used directly for batch encoding
            in Pass 2 to avoid N separate forward passes.
        threshold: Cosine similarity threshold for the SBERT pass.
            Tokens whose leave-one-out similarity to the full text is
            >= this value are replaced with ``[REDACTED]``.
            Default 0.95 is intentionally tight to minimise false positives.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.95) -> None:
        self._embedder = embedder
        self.threshold = threshold
        self._analyzer: Optional[Any] = None  # lazy-loaded on first scrub()

    # ------------------------------------------------------------------
    # Internal: Presidio pass
    # ------------------------------------------------------------------

    def _get_analyzer(self) -> Any:
        """Lazy-load the Presidio AnalyzerEngine with the en_core_web_trf backend.

        Returns:
            A ready-to-use ``AnalyzerEngine`` instance.
        """
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import TransformersNlpEngine

            model_config = [
                {"lang_code": "en", "model_name": {"spacy": "en_core_web_trf"}}
            ]
            nlp_engine = TransformersNlpEngine(models=model_config)
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        return self._analyzer

    def _presidio_pass(self, text: str) -> str:
        """Replace explicit PII detected by Presidio with ``[ENTITY_TYPE]`` tags.

        Results are processed in descending start-index order so that earlier
        replacements do not shift the character offsets of later ones.

        Args:
            text: Raw input text.

        Returns:
            Text with PII spans replaced, e.g. ``"Hi [PERSON], your [EMAIL_ADDRESS]…"``.
        """
        analyzer = self._get_analyzer()
        results = analyzer.analyze(text=text, language="en", entities=PII_ENTITIES)
        # Reverse order preserves character offsets across replacements.
        for result in sorted(results, key=lambda r: r.start, reverse=True):
            text = text[: result.start] + f"[{result.entity_type}]" + text[result.end :]
        return text

    # ------------------------------------------------------------------
    # Internal: SBERT leave-one-out pass
    # ------------------------------------------------------------------

    def _sbert_pass(self, text: str) -> str:
        """Mask opaque identifier tokens via batched SBERT leave-one-out scoring.

        Only tokens that pass the ``_looks_like_identifier`` structural pre-filter
        (pure digit strings, alphanumeric codes, hyphenated codes, UUIDs) are
        tested. All other tokens pass through unchanged without any SBERT check.

        For candidate tokens, their leave-one-out embeddings are computed in a
        single batch encode call. A candidate is masked when its removal leaves
        cosine similarity to the full text >= ``self.threshold``, indicating the
        token carries no semantic content.

        Args:
            text: Input text (after Presidio pass).

        Returns:
            Text with opaque identifiers replaced by ``[REDACTED]``.
        """
        tokens = text.split()
        if len(tokens) <= 1:
            return text

        # Indices of tokens that pass the structural pre-filter.
        candidate_indices = [i for i, t in enumerate(tokens) if _looks_like_identifier(t)]
        if not candidate_indices:
            return text

        # Build leave-one-out strings only for candidate tokens.
        loo_texts = [
            " ".join(tokens[:i] + tokens[i + 1:]) for i in candidate_indices
        ]

        # Batch encode: index 0 = full text, then one per candidate.
        self._embedder._load()
        embeddings = self._embedder._model.encode([text] + loo_texts, normalize_embeddings=True)

        full_emb: np.ndarray = embeddings[0]
        loo_embs: np.ndarray = embeddings[1:]

        # Cosine similarity = dot product for L2-normalised unit vectors.
        similarities: np.ndarray = loo_embs @ full_emb

        redact = {
            idx
            for idx, sim in zip(candidate_indices, similarities)
            if float(sim) >= self.threshold
        }

        return " ".join(
            "[REDACTED]" if i in redact else token
            for i, token in enumerate(tokens)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub(self, text: str) -> str:
        """Sanitize text through both PII scrubbing passes.

        Pass 1 (Presidio) runs first to handle named entities, then
        Pass 2 (SBERT) catches any remaining opaque identifiers.

        Args:
            text: Raw query or response text that may contain PII.

        Returns:
            Sanitized text safe to embed and store.
        """
        text = self._presidio_pass(text)
        text = self._sbert_pass(text)
        return text
