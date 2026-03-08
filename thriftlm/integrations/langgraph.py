"""
LangGraph native integration for SemanticCache.

Provides a wrap() helper that intercepts graph invocations and applies
semantic caching transparently, without requiring changes to graph internals.
"""

from typing import Any, Callable

from thriftlm.cache import SemanticCache


def wrap(graph: Any, cache: SemanticCache, input_key: str = "input") -> Callable[[Any], Any]:
    """Wrap a LangGraph graph with semantic caching.

    The returned callable has the same signature as graph.invoke().
    On each call it checks the cache; on a miss it runs the graph
    and stores the result before returning.

    Args:
        graph: A compiled LangGraph graph (must expose .invoke()).
        cache: An initialised SemanticCache instance.
        input_key: Key in the input dict that holds the user query string.

    Returns:
        A callable with the same interface as graph.invoke() but cache-aware.

    Example::

        from thriftlm import SemanticCache
        from thriftlm.integrations import wrap

        cache = SemanticCache(api_key="sc_xxx")
        cached_graph = wrap(graph, cache, input_key="question")
        result = cached_graph({"question": "What is RAG?"})
    """
    ...
