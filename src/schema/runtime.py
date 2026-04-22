"""Minimal runtime schemas for orchestration and decisions."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .artifacts import ArtifactIndex
from .base import (
    DecisionRoute,
    ExecutionOutcome,
    ReplanMode,
    SessionPhase,
    StrictModel,
    TaskStatus,
)


class Task(StrictModel):
    id: str
    plan_id: str
    type: str
    instruction: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    meta: dict[str, Any] | None = None


class Plan(StrictModel):
    id: str
    instruction: str
    task_ids: list[str] = Field(min_length=1)
    input_artifact_ids: list[str] = Field(default_factory=list)
    understanding_artifact_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] | None = None


class PlanLLMOutput(StrictModel):
    """Minimal structured output for the planning step."""

    plan_instruction: str
    tasks: list["PlanTaskSpec"] = Field(default_factory=list, min_length=1)


class PlanTaskSpec(StrictModel):
    """Structured planner output for a single task spec."""

    id: str
    type: str
    instruction: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list, min_length=1)


class ExecuteLLMOutput(StrictModel):
    """Minimal structured output for selecting execution strategy."""

    reasoning: str
    selected_tools: list[str] = Field(default_factory=list)
    edit_mode: str | None = None
    base_image_artifact_id: str | None = None


class ObserveArtifactSummary(StrictModel):
    artifact_id: str
    summary: str


class ObserveLLMOutput(StrictModel):
    outcome: str
    observation: str
    artifact_summaries: list[ObserveArtifactSummary] = Field(default_factory=list)


class TaskActRecord(StrictModel):
    """Single thinking-act-observe record inside an execute loop."""

    task_id: str
    loop_index: int
    act_index: int
    thinking_text: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    output_artifact_ids: list[str] = Field(default_factory=list)
    observation_text: str = ""


class TaskLoop(StrictModel):
    """Minimal single-loop execution record."""

    id: str
    task_id: str
    loop_index: int
    thinking: str
    selected_tools: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    observation: str = ""


class DecisionLLMOutput(StrictModel):
    """Minimal structured output for evaluator routing."""

    route: DecisionRoute
    summary: str
    issues: list[str] = Field(default_factory=list)


class TaskState(StrictModel):
    task_id: str
    status: TaskStatus
    task_artifact_ids: list[str] = Field(default_factory=list)
    final_artifact_id: str | None = None
    resolved_input_artifact_ids: list[str] = Field(default_factory=list)
    retry_input_artifact_ids: list[str] = Field(default_factory=list)
    retry_context_text: str | None = None
    input_selection_reasoning: str | None = None
    input_validation_summary: str | None = None
    latest_artifact_ids: list[str] = Field(default_factory=list)
    latest_execution_outcome: ExecutionOutcome | None = None
    latest_execute_checkpoint: str | None = None
    latest_evaluate_checkpoint: str | None = None
    loop_count: int = 0
    evaluator_checkpoint_count: int = 0


class TaskRetryAdvice(StrictModel):
    reason: str
    base_candidate_artifact_id: str | None = None
    reuse_artifact_ids: list[str] = Field(default_factory=list)
    fix_focuses: list[str] = Field(min_length=1)
    avoid_changes: list[str] = Field(default_factory=list)


class ReplanRequest(StrictModel):
    mode: ReplanMode
    reason: str
    preserve_task_ids: list[str] = Field(default_factory=list)
    preserve_artifact_ids: list[str] = Field(default_factory=list)


class Decision(StrictModel):
    id: str
    route: DecisionRoute
    task_id: str | None = None
    plan_id: str | None = None
    source_execution_outcome: ExecutionOutcome | None = None
    candidate_artifact_ids: list[str] = Field(default_factory=list)
    summary: str
    issues: list[str] = Field(default_factory=list)
    task_retry: TaskRetryAdvice | None = None
    replan: ReplanRequest | None = None
    meta: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_route_payloads(self) -> "Decision":
        if self.route == DecisionRoute.CONTINUE_EXECUTE:
            if self.task_retry is None or self.replan is not None:
                raise ValueError(
                    "continue_execute requires task_retry and forbids replan"
                )
        elif self.route == DecisionRoute.REPLAN:
            if self.replan is None or self.task_retry is not None:
                raise ValueError("replan requires replan and forbids task_retry")
        else:
            if self.task_retry is not None or self.replan is not None:
                raise ValueError(
                    "pass/fail decisions must not include task_retry or replan"
                )
        return self


class SessionState(StrictModel):
    session_id: str
    phase: SessionPhase
    current_plan_id: str | None = None
    current_task_id: str | None = None
    task_states: dict[str, TaskState] = Field(default_factory=dict)
    artifact_index: ArtifactIndex | None = None
    latest_decision_id: str | None = None
    final_result_id: str | None = None
