"""Runtime state models for the minimal LangGraph skeleton."""

from __future__ import annotations

from typing import Any, TypedDict

from schema import (
    Artifact,
    Decision,
    MessageEnvelope,
    Plan,
    SessionState,
    Task,
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
    artifacts: dict[str, Artifact]
    messages: list[MessageEnvelope]
    decision: Decision
    max_task_loops: int
    max_execute_acts: int
    max_evaluator_checkpoints: int
    max_tool_failures: int
    run_id: str
    output_dir: str | None
    run_log_uri: str | None
    message_log_uri: str | None
    run_logger: Any
