"""
thriftlm — A semantic caching layer for LLM applications.

Stop paying for the same LLM call twice just because users phrased it differently.
"""

from thriftlm.cache import SemanticCache
from thriftlm.config import Config

__all__ = ["SemanticCache", "Config"]
__version__ = "0.1.0"
