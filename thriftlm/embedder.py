"""
SBERT embedder wrapper.

Uses the all-MiniLM-L6-v2 model to produce 384-dimensional embeddings
that are sent to the backend for cosine similarity lookups.
"""

from typing import List


class Embedder:
    """Wraps sentence-transformers to produce fixed-size, normalized query embeddings.

    The model is lazy-loaded on the first call to embed() and cached for the
    lifetime of the instance, so repeated calls pay no additional load cost.

    Args:
        model: HuggingFace model name. Defaults to all-MiniLM-L6-v2.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model = None  # lazy-loaded on first call to embed()

    def _load(self) -> None:
        """Load the sentence-transformers model into self._model (idempotent)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> List[float]:
        """Embed a string into a normalized 384-dimensional vector.

        The returned vector is L2-normalized (unit length), making cosine
        similarity equivalent to a dot product — which is what pgvector uses
        under the hood.

        Args:
            text: The raw query string to embed.

        Returns:
            A list of 384 floats representing the unit-normalized embedding.
        """
        self._load()
        # normalize_embeddings=True produces a unit vector in one shot
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
