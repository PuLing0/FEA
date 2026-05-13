"""Run-scoped filesystem paths for runtime outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_RUN_OUTPUT_ROOT = "generated/runs"
DEFAULT_TEST_OUTPUT_ROOT = "generated/tests"
DEFAULT_SERVICE_OUTPUT_ROOT = "generated/services"


def safe_slug(value: Any, *, fallback: str = "session", limit: int = 80) -> str:
    """Return a filesystem-safe slug while keeping ids recognizable."""

    text = str(value or "").strip()
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    slug = slug.strip("_")[:limit]
    return slug or fallback


def default_run_output_root() -> Path:
    """Return the root directory for user-facing runtime runs."""

    return Path(os.getenv("AGENT_OUTPUT_DIR") or DEFAULT_RUN_OUTPUT_ROOT)


def default_test_output_root() -> Path:
    """Return the root directory for pytest-generated runtime outputs."""

    return Path(os.getenv("AGENT_TEST_OUTPUT_DIR") or DEFAULT_TEST_OUTPUT_ROOT)


def default_service_output_root() -> Path:
    """Return the root directory for long-lived backend service outputs."""

    return Path(os.getenv("AGENT_SERVICE_OUTPUT_DIR") or DEFAULT_SERVICE_OUTPUT_ROOT)


def _pytest_output_prefix() -> Path | None:
    raw_current_test = os.getenv("PYTEST_CURRENT_TEST", "").strip()
    if not raw_current_test:
        return None
    normalized = raw_current_test
    if normalized.endswith(")"):
        normalized = normalized.rsplit(" (", 1)[0]
    parts = [part for part in normalized.split("::") if part]
    file_label = safe_slug(Path(parts[0]).stem if parts else "pytest", fallback="pytest")
    if len(parts) > 1:
        case_label = safe_slug("__".join(parts[1:]), fallback="case", limit=120)
    else:
        case_label = "case"
    return default_test_output_root() / "pytest" / file_label / case_label


def output_root_prefix() -> Path:
    explicit_root = os.getenv("AGENT_OUTPUT_DIR", "").strip()
    if explicit_root:
        return Path(explicit_root)
    pytest_prefix = _pytest_output_prefix()
    if pytest_prefix is not None:
        return pytest_prefix
    return default_run_output_root()


def build_run_output_dir(session_id: str, run_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root_prefix() / f"{safe_slug(session_id)}_{stamp}_{safe_slug(run_id, fallback='run')}"


def _resolve_session_id(state: dict[str, Any]) -> str:
    session = state.get("session")
    session_id = getattr(session, "session_id", None)
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    runtime_input = state.get("input", {})
    input_session_id = runtime_input.get("session_id") if isinstance(runtime_input, dict) else None
    if isinstance(input_session_id, str) and input_session_id.strip():
        return input_session_id.strip()
    return "session"


def ensure_run_output_dir(state: dict[str, Any]) -> Path:
    raw_output_dir = state.get("output_dir")
    if isinstance(raw_output_dir, str) and raw_output_dir.strip():
        path = Path(raw_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = uuid4().hex[:12]
        state["run_id"] = run_id
    output_dir = build_run_output_dir(_resolve_session_id(state), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    state["output_dir"] = str(output_dir)
    return output_dir


def get_run_output_dir(state: dict[str, Any]) -> Path | None:
    raw_output_dir = state.get("output_dir")
    if raw_output_dir is None:
        return None
    return Path(raw_output_dir)


def run_logs_dir(state: dict[str, Any]) -> Path | None:
    output_dir = ensure_run_output_dir(state)
    path = output_dir / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tool_artifact_dir(state: dict[str, Any], tool_name: Any) -> Path:
    tool_slug = safe_slug(getattr(tool_name, "value", tool_name), fallback="tool")
    output_dir = ensure_run_output_dir(state)
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
    output_dir = ensure_run_output_dir(state)
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
