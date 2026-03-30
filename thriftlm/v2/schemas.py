from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired, Required


class IntentKey(TypedDict):
    # Core — always present
    action:      Required[str]
    target:      Required[str]
    goal:        Required[str]
    time_scope:  Required[str | None]
    # Optional — included when present in task
    domain:      NotRequired[str | None]
    format:      NotRequired[str | None]
    audience:    NotRequired[str | None]
    constraints: NotRequired[list[str] | None]   # sorted on normalization
    tool_family: NotRequired[str | None]


class CanonicalizationResult(TypedDict):
    intent_key:            IntentKey
    intent_bucket_hash:    str       # deterministic hash — see §intent.py rules
    confidence:            float
    canonicalizer_version: str
    raw_canonical_text:    str
    from_cache:            bool


class SlotSpec(TypedDict):
    name:           str
    type_hint:      str
    required:       bool
    source:         str             # context dict key
    transform:      str | None      # registered transform name
    transform_args: dict[str, Any] | None  # typed args for transform (e.g. {"n": 10})
    default:        Any | None


class PlanStep(TypedDict, total=False):
    step_id:     str
    op:          str
    inputs:      dict[str, str]     # slot_name or prior step_id references
    outputs:     dict[str, str]     # named output keys → type hints
    side_effect: bool
    tool_family: str | None


class PlanTemplate(TypedDict):
    plan_id:               str
    intent_key:            IntentKey
    intent_bucket_hash:    str
    description:           str      # used for semantic reranking
    steps:                 list[PlanStep]
    slots:                 list[SlotSpec]
    output_schema:         dict[str, str]
    optional_outputs:      list[str]
    plan_version:          str
    canonicalizer_version: str
    extractor_version:     str
    validator_version:     str
    created_at:            str


class ScoredPlan(TypedDict):
    plan:                PlanTemplate
    semantic_similarity: float
    structural_score:    float
    final_score:         float


class FilledPlan(TypedDict):
    plan_id:            str
    intent_bucket_hash: str
    filled_slots:       dict[str, Any]
    steps:              list[PlanStep]
    output_schema:      dict[str, str]


class ValidationResult(TypedDict):
    ok:                bool
    failed_stage:      str | None    # '1' through '7', or None on pass
    reason:            str | None
    validator_version: str


class ExtractionResult(TypedDict):
    ok:                    bool
    template:              PlanTemplate | None
    extraction_confidence: float
    generalization_notes:  str | None
    refusal_reason:        str | None
