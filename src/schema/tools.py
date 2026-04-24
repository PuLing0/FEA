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
    block_artifact_ids: list[str] = Field(min_length=1)
    layout_goal: str
    previous_collage_ref: str | None = None


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
    candidate_refs: list[str] = Field(min_length=1)
    checks: list[str] = Field(default_factory=list)


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
