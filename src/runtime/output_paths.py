"""Run-scoped filesystem paths for runtime outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RUN_OUTPUT_ROOT = "generated/agent_runs"


def safe_slug(value: Any, *, fallback: str = "session", limit: int = 80) -> str:
    """Return a filesystem-safe slug while keeping ids recognizable."""

    text = str(value or "").strip()
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    slug = slug.strip("_")[:limit]
    return slug or fallback


def default_run_output_root() -> Path:
    """Return the root directory for run-scoped outputs.

    AGENT_LOG_DIR is kept as a compatibility fallback, but AGENT_OUTPUT_DIR is
    the preferred setting because the directory now contains more than logs.
    """

    return Path(os.getenv("AGENT_OUTPUT_DIR") or os.getenv("AGENT_LOG_DIR") or DEFAULT_RUN_OUTPUT_ROOT)


def build_run_output_dir(session_id: str, run_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return default_run_output_root() / f"{safe_slug(session_id)}_{stamp}_{safe_slug(run_id, fallback='run')}"


def get_run_output_dir(state: dict[str, Any]) -> Path | None:
    raw_output_dir = state.get("output_dir")
    if raw_output_dir is None:
        return None
    return Path(raw_output_dir)


def run_logs_dir(state: dict[str, Any]) -> Path | None:
    output_dir = get_run_output_dir(state)
    if output_dir is None:
        return None
    path = output_dir / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tool_artifact_dir(state: dict[str, Any], tool_name: Any) -> Path:
    tool_slug = safe_slug(getattr(tool_name, "value", tool_name), fallback="tool")
    output_dir = get_run_output_dir(state)
    if output_dir is None:
        path = Path("generated") / tool_slug
    else:
        path = output_dir / "artifacts" / tool_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_tool_result_path(
    state: dict[str, Any],
    *,
    tool_name: Any,
    task_id: str,
    loop_index: int,
    tool_call_id: str | None = None,
) -> Path | None:
    output_dir = get_run_output_dir(state)
    if output_dir is None:
        return None
    tool_slug = safe_slug(getattr(tool_name, "value", tool_name), fallback="tool")
    task_slug = safe_slug(task_id, fallback="task")
    name = f"{tool_slug}_{loop_index:03d}"
    if tool_call_id:
        name = f"{name}_{safe_slug(tool_call_id, fallback='call')}"
    path = output_dir / "tool_results" / task_slug / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=json_default, indent=2) + "\n",
        encoding="utf-8",
    )
