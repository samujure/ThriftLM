"""HTTP client wrapper for the ThriftLM V2 plan cache server."""
from __future__ import annotations

import httpx

from thriftlm.v2.adapters.base import BasePlanCache


class ThriftLMPlanCache(BasePlanCache):
    """
    REST client for the ThriftLM V2 server.

    Sends requests to /v2/plan/lookup and /v2/plan/store.
    Raises RuntimeError on non-200 responses and on timeouts.
    No retries, no caching, no canonicalization logic.
    """

    def __init__(self, api_key: str, base_url: str, timeout: float = 5.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def lookup(self, task: str, context: dict, runtime_caps: dict) -> dict:
        """POST /v2/plan/lookup — returns parsed JSON dict."""
        try:
            resp = httpx.post(
                f"{self._base_url}/v2/plan/lookup",
                json={
                    "api_key": self._api_key,
                    "task": task,
                    "context": context,
                    "runtime_caps": runtime_caps,
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"lookup timed out: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"lookup request failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"lookup failed with status {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def store(self, plan: dict) -> dict:
        """POST /v2/plan/store — returns parsed JSON dict."""
        try:
            resp = httpx.post(
                f"{self._base_url}/v2/plan/store",
                json={"api_key": self._api_key, "plan": plan},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"store timed out: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"store request failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"store failed with status {resp.status_code}: {resp.text}"
            )
        return resp.json()
