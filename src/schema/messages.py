"""Message-first runtime schemas for agent orchestration."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field

from .base import DecisionRoute, ExecutionOutcome, ReplanMode, StrictModel, ToolName


MessageRole = Literal["system", "user", "assistant", "tool"]
MessageVisibility = Literal["model", "log_only", "ui_only"]
ToolResultStatus = Literal["succeeded", "failed"]


class TextBlock(StrictModel):
    type: Literal["text"] = "text"
    text: str


class ImageRefBlock(StrictModel):
    type: Literal["image_ref"] = "image_ref"
    artifact_id: str
    uri: str | None = None
    summary: str | None = None
    role: str | None = None


class ArtifactRefBlock(StrictModel):
    type: Literal["artifact_ref"] = "artifact_ref"
    artifact_id: str
    kind: str
    uri: str | None = None
    summary: str | None = None
    role: str | None = None
    created_by: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class PlanBlock(StrictModel):
    type: Literal["plan"] = "plan"
    plan_id: str
    instruction: str
    task_ids: list[str] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    understanding_artifact_ids: list[str] = Field(default_factory=list)


class TaskBlock(StrictModel):
    type: Literal["task"] = "task"
    task_id: str
    plan_id: str
    task_type: str
    instruction: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ThinkingSummaryBlock(StrictModel):
    type: Literal["thinking_summary"] = "thinking_summary"
    text: str


class ToolCallBlock(StrictModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: ToolName
    task_id: str | None = None
    loop_index: int | None = None
    act_index: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(StrictModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: ToolName
    task_id: str | None = None
    loop_index: int | None = None
    act_index: int | None = None
    status: ToolResultStatus
    args: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    result_payload: dict[str, Any] | None = None
    raw_output_uri: str | None = None
    summary: str | None = None
    error: dict[str, Any] | None = None


class ObservationBlock(StrictModel):
    type: Literal["observation"] = "observation"
    task_id: str
    loop_index: int
    act_index: int | None = None
    outcome: str | None = None
    text: str
    artifact_ids: list[str] = Field(default_factory=list)


class EvaluationBlock(StrictModel):
    type: Literal["evaluation"] = "evaluation"
    evaluation_ref: str | None = None
    task_id: str
    verdict: str | None = None
    reason: str | None = None
    candidate_ref: str | None = None
    candidate_refs: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class DecisionBlock(StrictModel):
    type: Literal["decision"] = "decision"
    decision_id: str
    route: DecisionRoute
    task_id: str | None = None
    plan_id: str | None = None
    source_execution_outcome: ExecutionOutcome | None = None
    candidate_artifact_ids: list[str] = Field(default_factory=list)
    summary: str
    issues: list[str] = Field(default_factory=list)
    task_retry: dict[str, Any] | None = None
    replan: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


class ContextSummaryBlock(StrictModel):
    type: Literal["context_summary"] = "context_summary"
    text: str
    covered_message_ids: list[str] = Field(default_factory=list)


class ErrorBlock(StrictModel):
    type: Literal["error"] = "error"
    error_type: str
    message: str
    recoverable: bool = False
    detail: dict[str, Any] | None = None


ContentBlock: TypeAlias = Annotated[
    TextBlock
    | ImageRefBlock
    | ArtifactRefBlock
    | PlanBlock
    | TaskBlock
    | ThinkingSummaryBlock
    | ToolCallBlock
    | ToolResultBlock
    | ObservationBlock
    | EvaluationBlock
    | DecisionBlock
    | ContextSummaryBlock
    | ErrorBlock,
    Field(discriminator="type"),
]


class RuntimeMessage(StrictModel):
    role: MessageRole
    content: list[ContentBlock] = Field(default_factory=list)


class MessageEnvelope(StrictModel):
    message_id: str
    parent_message_id: str | None = None
    session_id: str
    run_id: str
    agent_id: str = "main"
    parent_agent_id: str | None = None
    task_id: str | None = None
    loop_index: int | None = None
    act_index: int | None = None
    timestamp: str
    visibility: MessageVisibility = "model"
    message: RuntimeMessage
    usage: dict[str, Any] | None = None
