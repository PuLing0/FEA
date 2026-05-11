"""Minimal tool schemas and a shared invocation log type."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .base import StrictModel, ToolName


class UnderstandArgs(StrictModel):
    image_ref: str
    question: str | None = None


class GroundingArgs(StrictModel):
    image_ref: str
    grounding_query: str
    top_k: int | None = Field(default=1, ge=1)


class GroundingPoint(StrictModel):
    x: int
    y: int


BBox = Annotated[list[int], Field(min_length=4, max_length=4)]


class GroundingCandidate(StrictModel):
    label: str
    bbox: BBox
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_points: list[GroundingPoint] = Field(default_factory=list, max_length=3)
    negative_points: list[GroundingPoint] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_bbox_semantics(self) -> "GroundingCandidate":
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2:
            raise ValueError("bbox x1 must be less than x2")
        if y1 >= y2:
            raise ValueError("bbox y1 must be less than y2")
        return self


class GroundingLLMOutput(StrictModel):
    candidates: list[GroundingCandidate] = Field(min_length=1)


class SegmentArgs(StrictModel):
    image_ref: str
    prompt: str
    backend_name: str | None = None


class CropArgs(StrictModel):
    image_ref: str
    mask_ref: str | None = None
    grounding_ref: str | None = None
    padding: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_sources(self) -> "CropArgs":
        if self.mask_ref is None and self.grounding_ref is None:
            raise ValueError("crop requires mask_ref or grounding_ref")
        return self


class CollageArgs(StrictModel):
    block_artifact_ids: list[str] = Field(min_length=2)
    layout_goal: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_collage_args(self) -> "CollageArgs":
        normalized_goal = self.layout_goal.strip()
        if not normalized_goal:
            raise ValueError("layout_goal must be non-empty")
        self.layout_goal = normalized_goal
        if len(set(self.block_artifact_ids)) != len(self.block_artifact_ids):
            raise ValueError("block_artifact_ids must not contain duplicates")
        return self


class PromptReconstructArgs(StrictModel):
    input_artifact_ids: list[str] = Field(default_factory=list)


class EditArgs(StrictModel):
    instruction: str = Field(min_length=1)
    image_refs: list[str] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_image_refs(self) -> "EditArgs":
        normalized_instruction = self.instruction.strip()
        if not normalized_instruction:
            raise ValueError("instruction must be non-empty")
        self.instruction = normalized_instruction
        if len(set(self.image_refs)) != len(self.image_refs):
            raise ValueError("image_refs must not contain duplicates")
        return self


class EvaluateArgs(StrictModel):
    input_refs: list[str] = Field(default_factory=list)
    candidate_ref: str | None = None
    instruction: str | None = None
    candidate_refs: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evaluate_args(self) -> "EvaluateArgs":
        if self.candidate_ref is None:
            if not self.candidate_refs:
                raise ValueError("evaluate requires candidate_ref or candidate_refs")
            self.candidate_ref = self.candidate_refs[-1]
        if not self.candidate_refs:
            self.candidate_refs = [self.candidate_ref]
        if self.candidate_ref not in self.candidate_refs:
            self.candidate_refs.append(self.candidate_ref)
        if self.instruction is not None:
            normalized_instruction = self.instruction.strip()
            if not normalized_instruction:
                raise ValueError("instruction must be non-empty when provided")
            self.instruction = normalized_instruction
        return self


EvaluationVerdict = Literal["pass", "pass_with_issues", "needs_revision", "replan"]


class EvaluationScores(StrictModel):
    instruction_success: int = Field(ge=0, le=5)
    reference_consistency: int = Field(ge=0, le=5)
    overediting: int = Field(ge=0, le=5)
    naturalness: int = Field(ge=0, le=5)
    artifacts: int = Field(ge=0, le=5)


class EvaluateLLMOutput(StrictModel):
    is_satisfied: bool
    verdict: EvaluationVerdict
    scores: EvaluationScores
    reason: str
    issues: list[str] = Field(default_factory=list)
    new_rewritten_prompt: str | None = None


class ToolInvocationRecord(StrictModel):
    """Generic execution log for a single tool invocation."""

    id: str
    task_id: str
    loop_index: int
    tool_name: ToolName
    args: dict[str, Any]
    status: Literal["running", "succeeded", "failed"]
    output_refs: list[str] = Field(default_factory=list)
    result_payload: dict[str, Any] | None = None
    raw_output_uri: str | None = None
    error: dict[str, Any] | None = None
