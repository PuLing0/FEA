"""Runtime JSONL logging for agent graph runs."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.instruction_resolver import (
    resolve_active_instruction_artifact,
    resolve_active_instruction_text,
)
from schema import ArtifactKind


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_LOG_DIR = "generated/agent_logs"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass(frozen=True)
class RunLoggerConfig:
    enabled: bool
    console: bool
    log_dir: Path
    level: str

    @classmethod
    def from_env(cls) -> "RunLoggerConfig":
        return cls(
            enabled=_env_bool("AGENT_LOG_ENABLED", True),
            console=_env_bool("AGENT_LOG_CONSOLE", True),
            log_dir=Path(os.getenv("AGENT_LOG_DIR", DEFAULT_LOG_DIR)),
            level=os.getenv("AGENT_LOG_LEVEL", "debug").strip().lower() or "debug",
        )


class RunLogger:
    """Small JSONL logger that also mirrors concise events to stdout."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        log_path: Path | None,
        enabled: bool,
        console: bool,
        level: str,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.log_path = log_path
        self.enabled = enabled
        self.console = console
        self.level = level
        if self.enabled and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def uri(self) -> str | None:
        return str(self.log_path) if self.log_path is not None else None

    def event(self, name: str, state: dict[str, Any] | None = None, **payload: Any) -> None:
        if not self.enabled and not self.console:
            return
        session = state.get("session") if state else None
        record = {
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event": name,
            "phase": getattr(getattr(session, "phase", None), "value", getattr(session, "phase", None)),
            "current_plan_id": getattr(session, "current_plan_id", None),
            "current_task_id": getattr(session, "current_task_id", None),
            "payload": payload,
        }
        if self.enabled and self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        if self.console:
            print(_format_console_line(record), file=sys.stdout, flush=True)


def create_run_logger(session_id: str) -> RunLogger:
    config = RunLoggerConfig.from_env()
    run_id = uuid4().hex[:12]
    log_path = None
    if config.enabled:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.log_dir / f"{_safe_slug(session_id)}_{stamp}_{run_id}.jsonl"
    return RunLogger(
        run_id=run_id,
        session_id=session_id,
        log_path=log_path,
        enabled=config.enabled,
        console=config.console,
        level=config.level,
    )


def get_run_logger(state: dict[str, Any]) -> RunLogger | None:
    logger = state.get("run_logger")
    return logger if isinstance(logger, RunLogger) else None


def log_event(state: dict[str, Any], name: str, **payload: Any) -> None:
    logger = get_run_logger(state)
    if logger is not None:
        logger.event(name, state, **payload)


def summarize_operation(operation: Any) -> dict[str, Any]:
    return {
        "id": operation.id,
        "task_id": operation.task_id,
        "loop_index": operation.loop_index,
        "tool_name": getattr(operation.tool_name, "value", operation.tool_name),
        "status": operation.status,
        "args": operation.args,
        "output_refs": list(operation.output_refs),
        "error": operation.error,
        "raw_output_uri": operation.raw_output_uri,
    }


def summarize_artifact(artifact: Any) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "kind": getattr(artifact.kind, "value", artifact.kind),
        "uri": artifact.uri,
        "created_by": artifact.created_by,
        "source_ids": list(artifact.source_ids),
        "role": getattr(artifact, "role", None),
    }


def summarize_decision(decision: Any | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "id": decision.id,
        "route": getattr(decision.route, "value", decision.route),
        "task_id": decision.task_id,
        "plan_id": decision.plan_id,
        "summary": decision.summary,
        "issues": list(decision.issues),
        "candidate_artifact_ids": list(decision.candidate_artifact_ids),
    }


def summarize_task_state(
    task_state: Any | None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if task_state is None:
        return None
    summary = {
        "task_id": task_state.task_id,
        "status": getattr(task_state.status, "value", task_state.status),
        "latest_artifact_ids": list(task_state.latest_artifact_ids),
        "task_working_set_ids": [entry.artifact_id for entry in getattr(task_state, "task_working_set", [])],
        "latest_execute_checkpoint": task_state.latest_execute_checkpoint,
        "latest_evaluate_checkpoint": task_state.latest_evaluate_checkpoint,
        "loop_count": task_state.loop_count,
        "evaluator_checkpoint_count": task_state.evaluator_checkpoint_count,
        "edit_input_budget_overflow_count": getattr(task_state, "edit_input_budget_overflow_count", 0),
    }
    if state is not None:
        artifact = resolve_active_instruction_artifact(state, task_state.task_id)
        active_instruction = (
            artifact.get_instruction_text()
            if artifact is not None
            else resolve_active_instruction_text(state, task_state.task_id)
        )
        summary["active_instruction_artifact_id"] = artifact.id if artifact is not None else None
        summary["active_instruction_role"] = artifact.role if artifact is not None else None
        summary["active_instruction_preview"] = active_instruction[:120]
    return summary


def summarize_image_index(state: dict[str, Any]) -> list[str]:
    session = state.get("session")
    if session is None or session.artifact_index is None:
        return []
    return list(session.artifact_index.by_type.get(ArtifactKind.IMAGE, []))


def _format_console_line(record: dict[str, Any]) -> str:
    phase = record.get("phase") or "-"
    task_id = record.get("current_task_id") or "-"
    payload = record.get("payload") or {}
    details = []
    for key in ("node", "route", "tool_name", "status", "artifact_id", "decision_route", "log_uri"):
        if key in payload and payload[key] is not None:
            details.append(f"{key}={payload[key]}")
    detail_text = " ".join(details)
    if detail_text:
        detail_text = " " + detail_text
    return f"[agent:{record['run_id']}] {record['event']} phase={phase} task={task_id}{detail_text}"
