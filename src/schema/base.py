"""Base schema primitives for the fig edit agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model with strict validation defaults."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        use_enum_values=True,
    )


class SessionPhase(str, Enum):
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    DONE = "done"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_EVALUATION = "waiting_evaluation"
    PASSED = "passed"
    REPLANNED = "replanned"
    ABANDONED = "abandoned"
    FAILED = "failed"


class ExecutionOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class DecisionRoute(str, Enum):
    CONTINUE_EXECUTE = "continue_execute"
    PASS = "pass"
    REPLAN = "replan"
    FAIL = "fail"


class ArtifactKind(str, Enum):
    IMAGE = "image"
    MASK = "mask"
    GEOMETRY = "geometry"
    INSTRUCTION = "instruction"
    UNDERSTANDING = "understanding"
    EVALUATION = "evaluation"


class ArtifactStage(str, Enum):
    TEMP = "temp"
    WORKING = "working"
    COMMITTED = "committed"
    ARCHIVED = "archived"


class ToolName(str, Enum):
    UNDERSTAND = "understand"
    GROUNDING = "grounding"
    SEGMENT = "segment"
    CROP = "crop"
    COLLAGE = "collage"
    PROMPT_RECONSTRUCT = "prompt_reconstruct"
    EDIT = "edit"
    EVALUATE = "evaluate"


class ReplanMode(str, Enum):
    SPLIT_TASK = "split_task"
    REROUTE_PLAN = "reroute_plan"
    RESET_AND_REPLAN = "reset_and_replan"
