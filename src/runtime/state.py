"""Runtime state models for the minimal LangGraph skeleton."""

from __future__ import annotations

from typing import Any, TypedDict

from schema import (
    Artifact,
    Decision,
    Plan,
    SessionState,
    Task,
    TaskActRecord,
    TaskLoop,
    ToolInvocationRecord,
)


class RuntimeInput(TypedDict):
    """Input payload for the minimal runtime graph."""

    session_id: str
    image_uri: str
    image_uris: list[str]
    instruction_text: str
    desired_decision_route: str
    use_llm: bool


class RuntimeState(TypedDict, total=False):
    """Graph state used by the runtime skeleton."""

    input: RuntimeInput
    session: SessionState
    plans: dict[str, Plan]
    tasks: dict[str, Task]
    task_act_records: list[TaskActRecord]
    task_loops: list[TaskLoop]
    artifacts: dict[str, Artifact]
    operations: list[ToolInvocationRecord]
    decision: Decision
    max_task_loops: int
    max_execute_acts: int
    max_evaluator_checkpoints: int
    max_tool_failures: int
    run_id: str
    run_log_uri: str | None
    run_logger: Any
