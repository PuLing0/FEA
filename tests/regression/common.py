from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from agent import agent, create_agent, main as agent_main
from agents import EvaluatorAgent, ExecuteAgent, PlanAgent
from llm import (
    encode_image_path_to_data_url,
    invoke_llm,
    invoke_multimodal_llm,
    invoke_structured_llm,
    invoke_structured_multimodal_llm,
    load_llm_config,
)
from runtime.graph import build_runtime_graph
from runtime.instruction_resolver import (
    resolve_active_instruction_artifact,
    resolve_active_instruction_text,
)
from runtime.input_selector import (
    TaskInputSelectionOutput,
    build_candidate_image_pool,
    build_candidate_images_text,
)
from runtime.run_logger import summarize_task_state
from runtime.scheduler import select_next_runnable_task
from runtime.tool_runner import ToolRunner
from schema import (
    ArtifactIndex,
    ArtifactKind,
    ArtifactStage,
    CollageArgs,
    CropArgs,
    Decision,
    DecisionLLMOutput,
    DecisionRoute,
    EditArgs,
    EvaluateArgs,
    EvaluateLLMOutput,
    EvaluationScores,
    EvaluationArtifact,
    ExecuteLLMOutput,
    ExecutionOutcome,
    GeometryArtifact,
    GroundingArgs,
    GroundingCandidate,
    GroundingLLMOutput,
    GroundingPoint,
    ImageArtifact,
    InstructionArtifact,
    MaskArtifact,
    ObserveArtifactSummary,
    ObserveLLMOutput,
    Plan,
    PlanLLMOutput,
    PlanTaskSpec,
    ReplanMode,
    ReplanRequest,
    SegmentArgs,
    SessionPhase,
    SessionState,
    Task,
    TaskActRecord,
    TaskLoop,
    TaskRetryAdvice,
    TaskState,
    TaskStatus,
    ToolInvocationRecord,
    ToolName,
    UnderstandArgs,
    UnderstandingArtifact,
    WorkingSetEntry,
)
from tools import BaseTool, ToolExecutionResult, build_default_tool_registry
from tools.evaluate_tool import EvaluateTool
from tools.registry import ToolRegistry


def _evaluation_scores(score: int = 4, **overrides: int) -> EvaluationScores:
    values = {
        "instruction_success": score,
        "reference_consistency": score,
        "overediting": score,
        "naturalness": score,
        "artifacts": score,
    }
    values.update(overrides)
    return EvaluationScores(**values)


def _run_tool(tool: BaseTool, state: dict, *, task_id: str, loop_index: int, args):
    return ToolRunner(ToolRegistry({tool.name: tool})).run(
        state,
        tool.name,
        task_id=task_id,
        loop_index=loop_index,
        args=args,
    )


def _assert_tool_failed(result, *, error_type: str, message: str) -> None:
    assert result.invocation.status == "failed"
    assert result.invocation.error is not None
    assert result.invocation.error["type"] == error_type
    assert message in result.invocation.error["message"]


def _make_instruction_resolution_state(
    *,
    artifacts: dict[str, object],
    task_artifact_ids: list[str],
    task_instruction: str = "任务默认指令",
) -> dict:
    return {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction=task_instruction,
                acceptance_criteria=["满足编辑要求"],
            )
        },
        "session": SessionState(
            session_id="sess_instruction_resolution",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=list(task_artifact_ids),
                )
            },
        ),
        "artifacts": artifacts,
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }


class EditArgsHolder(BaseModel):
    value: EditArgs
