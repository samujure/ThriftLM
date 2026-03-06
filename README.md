# SemanticCache
A semantic caching layer for LLM applications. Instead of calling the LLM every time, semanticCache embeds the user query using SBERT, checks a vector store for semantically similar past queries, and returns the cached response if similarity is above threshold. If no match, it calls the LLM, caches the result, and returns it.
