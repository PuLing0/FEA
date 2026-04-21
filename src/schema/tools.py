"""Minimal tool schemas and a shared invocation log type."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .base import EditMode, StrictModel, ToolName


class UnderstandArgs(StrictModel):
    image_ref: str
    question: str | None = None


class GroundingArgs(StrictModel):
    image_ref: str
    target_description: str
    top_k: int | None = Field(default=1, ge=1)


class SegmentArgs(StrictModel):
    image_ref: str
    target: str
    region_hint: str | None = None


class CropArgs(StrictModel):
    image_ref: str
    mask_ref: str


class CollageArgs(StrictModel):
    block_artifact_ids: list[str] = Field(min_length=1)
    layout_goal: str
    previous_collage_ref: str | None = None


class PromptReconstructArgs(StrictModel):
    input_artifact_ids: list[str] = Field(default_factory=list)


class LocalEditArgs(StrictModel):
    mode: Literal[EditMode.LOCAL_EDIT] = EditMode.LOCAL_EDIT
    image_ref: str
    mask_ref: str
    preserve: list[str] = Field(default_factory=list)


class GlobalEditArgs(StrictModel):
    mode: Literal[EditMode.GLOBAL_EDIT] = EditMode.GLOBAL_EDIT
    image_ref: str
    preserve: list[str] = Field(default_factory=list)


class ReferenceEditArgs(StrictModel):
    mode: Literal[EditMode.REFERENCE_EDIT] = EditMode.REFERENCE_EDIT
    image_ref: str
    reference_refs: list[str] = Field(min_length=1)
    preserve: list[str] = Field(default_factory=list)


EditArgs = Annotated[
    LocalEditArgs | GlobalEditArgs | ReferenceEditArgs,
    Field(discriminator="mode"),
]


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
