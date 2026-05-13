"""Runtime message append and JSONL transcript helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.output_paths import run_logs_dir
from schema import (
    ArtifactRefBlock,
    ContentBlock,
    Decision,
    DecisionBlock,
    ErrorBlock,
    EvaluationBlock,
    MessageEnvelope,
    MessageVisibility,
    ObservationBlock,
    RuntimeMessage,
    TextBlock,
    ThinkingSummaryBlock,
    ToolCallBlock,
    ToolName,
    ToolResultBlock,
)

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return str(value)

def init_message_store(state: dict[str, Any]) -> None:
    """Ensure message memory and transcript path are initialized."""

    state.setdefault("messages", [])
    run_id = state.get("run_id")
    if not run_id:
        run_id = uuid4().hex[:12]
        state["run_id"] = run_id
    if state.get("message_log_uri"):
        return
    run_log_dir = run_logs_dir(state)
    assert run_log_dir is not None
    state["message_log_uri"] = str(run_log_dir / "messages.jsonl")


def append_message(
    state: dict[str, Any],
    *,
    role: str,
    content: list[ContentBlock],
    task_id: str | None = None,
    loop_index: int | None = None,
    act_index: int | None = None,
    visibility: MessageVisibility = "model",
    agent_id: str = "main",
    parent_agent_id: str | None = None,
    usage: dict[str, Any] | None = None,
) -> MessageEnvelope:
    """Append one runtime message to state and the JSONL transcript."""

    init_message_store(state)
    messages: list[MessageEnvelope] = state["messages"]
    session = state.get("session")
    session_id = getattr(session, "session_id", None) or state.get("input", {}).get("session_id", "session")
    parent_message_id = messages[-1].message_id if messages else None
    envelope = MessageEnvelope(
        message_id=uuid4().hex,
        parent_message_id=parent_message_id,
        session_id=str(session_id),
        run_id=str(state["run_id"]),
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        task_id=task_id,
        loop_index=loop_index,
        act_index=act_index,
        timestamp=_timestamp(),
        visibility=visibility,
        message=RuntimeMessage(role=role, content=content),
        usage=usage,
    )
    messages.append(envelope)
    log_uri = state.get("message_log_uri")
    if log_uri:
        with Path(log_uri).open("a", encoding="utf-8") as file:
            file.write(json.dumps(envelope, ensure_ascii=False, default=_json_default) + "\n")
    return envelope


def load_messages(log_uri: str | Path) -> list[MessageEnvelope]:
    """Load a JSONL transcript into envelopes."""

    path = Path(log_uri)
    if not path.is_file():
        return []
    envelopes: list[MessageEnvelope] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelopes.append(MessageEnvelope.model_validate_json(line))
    return envelopes


def next_tool_call_id(state: dict[str, Any], tool_name: ToolName) -> str:
    """Allocate a deterministic-looking tool call id from existing messages."""

    init_message_store(state)
    count = 0
    for envelope in state.get("messages", []):
        for block in envelope.message.content:
            if isinstance(block, ToolCallBlock) and block.tool_name == tool_name:
                count += 1
    return f"call_{tool_name.value}_{count + 1:03d}"


def append_text_message(
    state: dict[str, Any],
    *,
    role: str,
    text: str,
    task_id: str | None = None,
    visibility: MessageVisibility = "model",
) -> MessageEnvelope:
    return append_message(
        state,
        role=role,
        content=[TextBlock(text=text)],
        task_id=task_id,
        visibility=visibility,
    )


def append_artifact_ref_message(
    state: dict[str, Any],
    artifact: Any,
    *,
    role: str = "assistant",
    task_id: str | None = None,
    visibility: MessageVisibility = "model",
) -> MessageEnvelope:
    return append_message(
        state,
        role=role,
        task_id=task_id,
        visibility=visibility,
        content=[
            ArtifactRefBlock(
                artifact_id=artifact.id,
                kind=str(getattr(artifact.kind, "value", artifact.kind)),
                uri=getattr(artifact, "uri", None),
                summary=getattr(artifact, "summary", None),
                role=getattr(artifact, "role", None),
                created_by=getattr(artifact, "created_by", None),
                source_ids=list(getattr(artifact, "source_ids", [])),
            )
        ],
    )


def append_tool_call_message(
    state: dict[str, Any],
    *,
    tool_call_id: str,
    tool_name: ToolName,
    args: dict[str, Any],
    task_id: str,
    loop_index: int,
    act_index: int | None = None,
) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=task_id,
        loop_index=loop_index,
        act_index=act_index,
        content=[
            ToolCallBlock(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                task_id=task_id,
                loop_index=loop_index,
                act_index=act_index,
                args=args,
            )
        ],
    )


def append_tool_result_message(
    state: dict[str, Any],
    *,
    tool_call_id: str,
    tool_name: ToolName,
    status: str,
    args: dict[str, Any],
    artifact_ids: list[str],
    task_id: str,
    loop_index: int,
    result_payload: dict[str, Any] | None = None,
    raw_output_uri: str | None = None,
    error: dict[str, Any] | None = None,
    summary: str | None = None,
    act_index: int | None = None,
) -> MessageEnvelope:
    return append_message(
        state,
        role="tool",
        task_id=task_id,
        loop_index=loop_index,
        act_index=act_index,
        content=[
            ToolResultBlock(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                task_id=task_id,
                loop_index=loop_index,
                act_index=act_index,
                status=status,  # type: ignore[arg-type]
                args=args,
                artifact_ids=artifact_ids,
                result_payload=result_payload,
                raw_output_uri=raw_output_uri,
                summary=summary,
                error=error,
            )
        ],
    )


def append_thinking_message(
    state: dict[str, Any],
    *,
    text: str,
    task_id: str,
    loop_index: int,
    act_index: int,
) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=task_id,
        loop_index=loop_index,
        act_index=act_index,
        content=[ThinkingSummaryBlock(text=text)],
    )


def append_observation_message(
    state: dict[str, Any],
    *,
    task_id: str,
    loop_index: int,
    text: str,
    outcome: str | None = None,
    artifact_ids: list[str] | None = None,
    act_index: int | None = None,
) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=task_id,
        loop_index=loop_index,
        act_index=act_index,
        content=[
            ObservationBlock(
                task_id=task_id,
                loop_index=loop_index,
                act_index=act_index,
                outcome=outcome,
                text=text,
                artifact_ids=list(artifact_ids or []),
            )
        ],
    )


def append_evaluation_message(
    state: dict[str, Any],
    *,
    task_id: str,
    evaluation_ref: str | None,
    verdict: str | None,
    reason: str | None,
    candidate_ref: str | None,
    candidate_refs: list[str],
    checks: list[str],
) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=task_id,
        content=[
            EvaluationBlock(
                task_id=task_id,
                evaluation_ref=evaluation_ref,
                verdict=verdict,
                reason=reason,
                candidate_ref=candidate_ref,
                candidate_refs=list(candidate_refs),
                checks=list(checks),
            )
        ],
    )


def append_decision_message(state: dict[str, Any], decision: Decision) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=decision.task_id,
        content=[
            DecisionBlock(
                decision_id=decision.id,
                route=decision.route,
                task_id=decision.task_id,
                plan_id=decision.plan_id,
                source_execution_outcome=decision.source_execution_outcome,
                candidate_artifact_ids=list(decision.candidate_artifact_ids),
                summary=decision.summary,
                issues=list(decision.issues),
                task_retry=decision.task_retry.model_dump(mode="json") if decision.task_retry else None,
                replan=decision.replan.model_dump(mode="json") if decision.replan else None,
                meta=decision.meta,
            )
        ],
    )


def append_error_message(
    state: dict[str, Any],
    *,
    error_type: str,
    message: str,
    recoverable: bool = False,
    detail: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> MessageEnvelope:
    return append_message(
        state,
        role="assistant",
        task_id=task_id,
        visibility="log_only",
        content=[
            ErrorBlock(
                error_type=error_type,
                message=message,
                recoverable=recoverable,
                detail=detail,
            )
        ],
    )
