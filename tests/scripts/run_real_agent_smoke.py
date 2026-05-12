"""Run a real end-to-end fig edit agent smoke test.

This script intentionally requires explicit opt-in because it may call a real
LLM endpoint and load the real FireRed CUDA backend.

Example:

    FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
    RUN_REAL_AGENT_SMOKE=1 \
    ./.venv/bin/python tests/scripts/run_real_agent_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(REPO_ROOT / ".env")

from agent import create_agent, format_final_summary  # noqa: E402
from llm import load_llm_config  # noqa: E402
from runtime.config import default_max_execute_acts  # noqa: E402
from runtime.message_query import find_tool_results  # noqa: E402
from runtime.prompts import REAL_AGENT_SMOKE_DEFAULT_INSTRUCTION  # noqa: E402
from schema import ArtifactKind, SessionPhase, ToolName  # noqa: E402
from vision_backends.firered_edit_backend import unload_pipeline  # noqa: E402

DEFAULT_IMAGES = [
    REPO_ROOT / "examples" / "fig1.jpg",
    REPO_ROOT / "examples" / "fig2.jpg",
    REPO_ROOT / "examples" / "fig3.jpg",
    REPO_ROOT / "examples" / "fig4.jpg",
]
DEFAULT_INSTRUCTION = REAL_AGENT_SMOKE_DEFAULT_INSTRUCTION


def _require_opt_in() -> None:
    if os.getenv("RUN_REAL_AGENT_SMOKE") != "1":
        raise SystemExit("set RUN_REAL_AGENT_SMOKE=1 to run the real agent smoke test")


def _require_llm_config() -> None:
    config = load_llm_config()
    missing = []
    if not config.api_key:
        missing.append("LLM_API_KEY")
    if not config.base_url:
        missing.append("LLM_BASE_URL")
    if not config.model_name:
        missing.append("LLM_MODEL_NAME")
    if missing:
        raise SystemExit(f"missing LLM configuration: {', '.join(missing)}")


def _resolve_image_paths() -> list[Path]:
    configured = os.getenv("REAL_AGENT_SMOKE_IMAGES")
    if configured:
        image_paths = [Path(item.strip()) for item in configured.split(",") if item.strip()]
    else:
        single_image = os.getenv("REAL_AGENT_SMOKE_IMAGE")
        image_paths = [Path(single_image)] if single_image else list(DEFAULT_IMAGES)
    resolved = []
    for image_path in image_paths:
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        if not image_path.is_file():
            raise SystemExit(f"real agent smoke image not found: {image_path}")
        resolved.append(image_path)
    if not resolved:
        raise SystemExit("real agent smoke requires at least one image")
    return resolved


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)



def _is_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _find_latest_edit_output_id(state: dict[str, Any]) -> str | None:
    for result in reversed(find_tool_results(state, tool_name=ToolName.EDIT, status="succeeded")):
        if result.artifact_ids:
            return result.artifact_ids[-1]
    return None


def _run_graph_smoke(graph: Any, input_state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not _is_enabled("REAL_AGENT_SMOKE_STOP_AFTER_FIRST_EDIT", True):
        return graph.invoke(input_state), "graph_terminal"

    latest_state: dict[str, Any] | None = None
    for update in graph.stream(input_state, stream_mode="values"):
        latest_state = update
        if _find_latest_edit_output_id(update):
            return update, "first_edit_candidate"
    if latest_state is None:
        raise RuntimeError("agent graph produced no states")
    return latest_state, "graph_terminal"

def _summarize_tool_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in find_tool_results(state):
        rows.append(
            {
                "id": result.tool_call_id,
                "task_id": result.task_id,
                "loop_index": result.loop_index,
                "tool_name": _enum_value(result.tool_name),
                "status": result.status,
                "output_refs": list(result.artifact_ids),
                "error": result.error,
            }
        )
    return rows


def _summarize_artifact(state: dict[str, Any], artifact_id: str | None) -> dict[str, Any] | None:
    if not artifact_id:
        return None
    artifact = state.get("artifacts", {}).get(artifact_id)
    if artifact is None:
        return None
    return {
        "id": artifact.id,
        "kind": _enum_value(artifact.kind),
        "uri": artifact.uri,
        "created_by": artifact.created_by,
        "source_ids": list(artifact.source_ids),
        "payload": artifact.payload,
    }


def _latest_image_artifact_id(state: dict[str, Any]) -> str | None:
    session = state["session"]
    indexed_images = []
    if session.artifact_index is not None:
        indexed_images = list(session.artifact_index.by_type.get(ArtifactKind.IMAGE, []))
    for artifact_id in reversed(indexed_images):
        if not artifact_id.startswith("art_img_input_"):
            return artifact_id
    return None


def main() -> int:
    _require_opt_in()
    _require_llm_config()
    image_paths = _resolve_image_paths()
    instruction = os.getenv("REAL_AGENT_SMOKE_INSTRUCTION", DEFAULT_INSTRUCTION)
    os.environ.setdefault("FIRERED_HEIGHT", "768")
    os.environ.setdefault("FIRERED_WIDTH", "352")
    os.environ.setdefault("FIRERED_NUM_INFERENCE_STEPS", "8")
    os.environ.setdefault("FIRERED_FUSE_LORA", "false")

    graph = create_agent()
    input_state = {
        "input": {
            "session_id": os.getenv("REAL_AGENT_SMOKE_SESSION_ID", "real-agent-smoke"),
            "image_uri": str(image_paths[0]),
            "image_uris": [str(image_path) for image_path in image_paths],
            "instruction_text": instruction,
            "desired_decision_route": os.getenv("REAL_AGENT_SMOKE_DESIRED_ROUTE", "pass"),
            "use_llm": True,
        },
        "max_task_loops": int(os.getenv("REAL_AGENT_SMOKE_MAX_TASK_LOOPS", "2")),
        "max_execute_acts": int(
            os.getenv(
                "REAL_AGENT_SMOKE_MAX_EXECUTE_ACTS",
                str(default_max_execute_acts()),
            )
        ),
        "max_evaluator_checkpoints": int(os.getenv("REAL_AGENT_SMOKE_MAX_EVALUATOR_CHECKPOINTS", "1")),
        "max_tool_failures": int(os.getenv("REAL_AGENT_SMOKE_MAX_TOOL_FAILURES", "1")),
    }
    try:
        result, stop_reason = _run_graph_smoke(graph, input_state)

        session = result["session"]
        decision = result.get("decision")
        final_artifact_id = session.final_result_id or _find_latest_edit_output_id(result) or _latest_image_artifact_id(result)
        summary = {
            "run_id": result.get("run_id"),
            "run_log_uri": result.get("run_log_uri"),
            "message_log_uri": result.get("message_log_uri"),
            "stop_reason": stop_reason,
            "session_phase": _enum_value(session.phase),
            "current_plan_id": session.current_plan_id,
            "current_task_id": session.current_task_id,
            "latest_decision_id": session.latest_decision_id,
            "final_result_id": session.final_result_id,
            "fallback_final_image_id": final_artifact_id,
            "decision": None
            if decision is None
            else {
                "id": decision.id,
                "route": _enum_value(decision.route),
                "task_id": decision.task_id,
                "summary": decision.summary,
                "issues": list(decision.issues),
            },
            "final_artifact": _summarize_artifact(result, final_artifact_id),
            "tool_results": _summarize_tool_results(result),
        }
        print(format_final_summary(summary))

        final_artifact = result.get("artifacts", {}).get(final_artifact_id) if final_artifact_id else None
        if stop_reason == "graph_terminal" and session.phase == SessionPhase.FAILED:
            print("agent smoke ended in failed phase", file=sys.stderr)
            return 1
        if final_artifact is None or not final_artifact.uri or not Path(final_artifact.uri).is_file():
            print("agent smoke did not produce a readable final image artifact", file=sys.stderr)
            return 1
        return 0
    finally:
        unload_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
