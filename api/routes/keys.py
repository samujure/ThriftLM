"""
Keys route.

POST /keys  {email}  →  {api_key}

Generates a new developer API key prefixed with 'sc_' and backed by
128 bits of cryptographic entropy. Inserts a zeroed metrics row into
the api_keys table so /metrics works immediately after key creation.
"""

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from api.db import get_supabase_client

router = APIRouter()


class KeyRequest(BaseModel):
    email: EmailStr


class KeyResponse(BaseModel):
    api_key: str


def _generate_api_key() -> str:
    """Return a new unique API key of the form ``sc_<32 hex chars>``.

    Uses :func:`secrets.token_hex` (128-bit entropy) so keys are
    unguessable and safe for use as bearer tokens.
    """
    return f"sc_{secrets.token_hex(16)}"


@router.post("/keys", response_model=KeyResponse, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: KeyRequest,
    client: Any = Depends(get_supabase_client),
) -> KeyResponse:
    """Generate and return a new API key for the given email address.

    Inserts a row into ``api_keys`` with all metric counters zeroed.
    The key is returned once and never stored in plaintext again —
    the caller must save it immediately.

    Raises:
        HTTPException 409: If a key already exists for this email address.
    """
    # Guard against duplicate registrations for the same email.
    existing = (
        client.table("api_keys")
        .select("api_key")
        .eq("email", body.email)
        .maybe_single()
        .execute()
    )
    if existing.data is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An API key for this email already exists.",
        )

    api_key = _generate_api_key()

    client.table("api_keys").insert(
        {
            "api_key": api_key,
            "email": body.email,
            "total_queries": 0,
            "total_hits": 0,
            "tokens_saved": 0,
        }
    ).execute()

    return KeyResponse(api_key=api_key)
