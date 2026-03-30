from __future__ import annotations

import json
from typing import Any

from thriftlm.v2.schemas import PlanTemplate, ScoredPlan


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(x * x for x in v2) ** 0.5
    return dot / (norm1 * norm2 + 1e-9)


def _parse_embedding(raw: Any) -> list[float] | None:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        raw = parsed
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def _row_to_plan(row: dict[str, Any]) -> PlanTemplate | None:
    try:
        tmpl = row["template_json"]
        if isinstance(tmpl, str):
            tmpl = json.loads(tmpl)
        if not isinstance(tmpl, dict):
            return None
        return tmpl  # type: ignore[return-value]
    except (KeyError, json.JSONDecodeError, ValueError):
        return None


def _is_valid_structural_signature(value: object) -> bool:
    """Reject wrong types on known fields; absent optional fields pass."""
    if not isinstance(value, dict):
        return False
    if "required_context_keys" in value:
        rck = value["required_context_keys"]
        if not isinstance(rck, list) or not all(isinstance(k, str) for k in rck):
            return False
    if "tool_families" in value:
        tf = value["tool_families"]
        if not isinstance(tf, list) or not all(isinstance(f, str) for f in tf):
            return False
    if "has_side_effects" in value and not isinstance(value["has_side_effects"], bool):
        return False
    if "format" in value and value["format"] is not None and not isinstance(value["format"], str):
        return False
    if "audience" in value and value["audience"] is not None and not isinstance(value["audience"], str):
        return False
    if "step_count" in value and not isinstance(value["step_count"], int):
        return False
    return True


def _parse_sig(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        sig = row["structural_signature"]
        if isinstance(sig, str):
            sig = json.loads(sig)
        if not _is_valid_structural_signature(sig):
            return None
        return sig
    except (KeyError, json.JSONDecodeError, ValueError):
        return None


def _slot_overlap_score(sig: dict[str, Any], context: dict) -> float:
    required_keys: list = sig.get("required_context_keys") or []
    if not required_keys:
        return 1.0
    overlap = len(set(required_keys) & set(context.keys()))
    return overlap / len(required_keys)


def _tool_family_match_score(sig: dict[str, Any], runtime_caps: dict) -> float:
    plan_families: list = sig.get("tool_families") or []
    if not plan_families:
        return 1.0
    runtime_families: list = runtime_caps.get("tool_families") or []
    return 1.0 if set(plan_families) & set(runtime_families) else 0.0


def _format_audience_score(sig: dict[str, Any], runtime_caps: dict) -> float:
    def _field_score(plan_val, runtime_val) -> float:
        if plan_val is None:
            return 0.5
        if runtime_val is None:
            return 0.5
        return 1.0 if plan_val == runtime_val else 0.0

    fmt = _field_score(sig.get("format"), runtime_caps.get("format"))
    aud = _field_score(sig.get("audience"), runtime_caps.get("audience"))
    return (fmt + aud) / 2.0


def _side_effect_compat(sig: dict[str, Any], runtime_caps: dict) -> float:
    has_side_effects: bool = sig.get("has_side_effects", False)
    if not has_side_effects:
        return 1.0
    return 1.0 if runtime_caps.get("allow_side_effects", False) else 0.0


def _structural_score(sig: dict[str, Any], context: dict, runtime_caps: dict) -> float:
    return (
        0.35 * _slot_overlap_score(sig, context)
        + 0.25 * _tool_family_match_score(sig, runtime_caps)
        + 0.20 * _format_audience_score(sig, runtime_caps)
        + 0.20 * _side_effect_compat(sig, runtime_caps)
    )


class PlanCache:
    """
    Fetch, score, and rerank plan templates from the `plans` table for one
    intent bucket.

    Does NOT adapt slots, validate plans, execute plans, call OpenAI, or
    invoke the extractor.

    Args:
        supabase_client: Live Supabase client (passed in; no env reads here).
        api_key:         Developer API key — scopes queries to one tenant.
        plan_threshold:  Minimum final_score to include a candidate.
        top_k:           Maximum number of candidates to return.
        embedder:        Optional Embedder instance; instantiated lazily if None.
    """

    def __init__(
        self,
        supabase_client,
        api_key: str,
        plan_threshold: float = 0.60,
        top_k: int = 5,
        embedder=None,
    ) -> None:
        self._client = supabase_client
        self.api_key = api_key
        self.plan_threshold = plan_threshold
        self.top_k = top_k
        self._embedder = embedder

    def _get_embedder(self):
        if self._embedder is None:
            from thriftlm.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def get(
        self,
        intent_bucket_hash: str,
        task: str,
        context: dict,
        runtime_caps: dict,
    ) -> list[ScoredPlan]:
        """
        Return scored, filtered, and ranked plan candidates for *intent_bucket_hash*.

        Steps:
        1. Fetch rows from `plans` filtered by api_key, intent_bucket_hash, is_valid.
        2. Embed the raw task text.
        3. Score each row (semantic + structural).
        4. Filter by plan_threshold, sort descending, return top_k.
        """
        try:
            rows: list[dict[str, Any]] = (
                self._client.table("plans")
                .select(
                    "id, description, embedding, template_json, structural_signature, "
                    "intent_bucket_hash, plan_version, canonicalizer_version, "
                    "extractor_version, validator_version, created_at"
                )
                .eq("api_key", self.api_key)
                .eq("intent_bucket_hash", intent_bucket_hash)
                .eq("is_valid", True)
                .execute()
                .data
            )
        except Exception:
            return []

        if not rows:
            return []

        task_vec = self._get_embedder().embed(task)

        scored: list[ScoredPlan] = []
        for row in rows:
            try:
                stored_vec = _parse_embedding(row.get("embedding"))
                if stored_vec is None or len(stored_vec) != len(task_vec):
                    continue

                plan = _row_to_plan(row)
                if plan is None:
                    continue

                sig = _parse_sig(row)
                if sig is None:
                    continue

                sem = _cosine_similarity(task_vec, stored_vec)
                struct = _structural_score(sig, context, runtime_caps)
                final = 0.7 * sem + 0.3 * struct

                if final < self.plan_threshold:
                    continue

                scored.append(
                    ScoredPlan(
                        plan=plan,
                        semantic_similarity=sem,
                        structural_score=struct,
                        final_score=final,
                    )
                )
            except Exception:
                continue  # malformed row — skip, do not crash

        scored.sort(key=lambda s: s["final_score"], reverse=True)
        return scored[: self.top_k]
