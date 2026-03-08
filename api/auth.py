"""
API key authentication dependency for FastAPI routes.

Auth is intentionally minimal in V1 — the dependency reads the X-API-Key
header and returns it as-is. Full DB validation (checking the api_keys
table and rejecting unknown keys) will be added before public launch.
"""

from fastapi import Header


async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """FastAPI dependency that extracts the X-API-Key request header.

    The key is returned directly to the route handler. No database lookup
    is performed yet; any non-empty string is accepted.

    Args:
        x_api_key: Value of the ``X-API-Key`` header (injected by FastAPI).

    Returns:
        The raw API key string from the request header.

    Raises:
        HTTPException 422: If the header is missing entirely (enforced
            automatically by FastAPI because the parameter is required).
    """
    return x_api_key
