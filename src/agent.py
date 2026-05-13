"""Entry point for the minimal fig edit agent.

This module provides a stable top-level runtime entry:

- `create_agent()` returns a compiled LangGraph runtime
- `agent` exposes a default compiled graph instance
- `python src/agent.py ...` runs the agent as a CLI

This keeps higher-level integrations from importing deep runtime internals
directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from runtime.graph import build_runtime_graph
from runtime.config import default_max_execute_acts, default_max_tool_failures
from runtime.message_query import find_latest_tool_result, find_tool_results
from schema import ArtifactKind, SessionPhase, ToolName
from vision_backends.firered_edit_backend import unload_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SUMMARY_TEXT = 240


def create_agent() -> CompiledStateGraph:
    """Create and return the minimal fig edit runtime graph."""

    return build_runtime_graph()


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _short_text(value: Any, *, limit: int = MAX_SUMMARY_TEXT) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_refs(values: Any, *, limit: int = 6) -> str:
    if values is None:
        return "无"
    if isinstance(values, str):
        items = [values]
    else:
        try:
            items = [str(item) for item in values]
        except TypeError:
            items = [str(values)]
    if not items:
        return "无"
    visible = items[:limit]
    suffix = f" 等 {len(items)} 项" if len(items) > limit else ""
    return ", ".join(visible) + suffix


def _tool_label(tool_name: Any) -> str:
    labels = {
        "understand": "图片理解",
        "grounding": "目标定位",
        "segment": "图像分割",
        "crop": "裁剪预览",
        "collage": "拼图参考",
        "prompt_reconstruct": "指令重写",
        "edit": "图像编辑",
        "evaluate": "结果评估",
    }
    tool_text = str(_enum_value(tool_name))
    return labels.get(tool_text, tool_text)


def _stop_reason_label(stop_reason: Any) -> str:
    labels = {
        "first_edit_candidate": "已生成第一张编辑候选图后停止",
        "graph_terminal": "运行到图终止节点",
    }
    text = str(stop_reason)
    return labels.get(text, text)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _resolve_image_paths(image_values: Sequence[str]) -> list[Path]:
    resolved = []
    for image_value in image_values:
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        if not image_path.is_file():
            raise SystemExit(f"image not found: {image_path}")
        resolved.append(image_path)
    if not resolved:
        raise SystemExit("at least one --images path is required")
    return resolved


def _find_latest_edit_output_id(state: dict[str, Any]) -> str | None:
    result = find_latest_tool_result(
        state,
        tool_name=ToolName.EDIT,
        status="succeeded",
    )
    if result is not None and result.artifact_ids:
        return result.artifact_ids[-1]
    return None


def _latest_image_artifact_id(state: dict[str, Any]) -> str | None:
    session = state["session"]
    indexed_images = []
    if session.artifact_index is not None:
        indexed_images = list(session.artifact_index.by_type.get(ArtifactKind.IMAGE, []))
    for artifact_id in reversed(indexed_images):
        if not artifact_id.startswith("art_img_input_"):
            return artifact_id
    return None


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


def _summarize_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    decision = state.get("decision")
    if decision is None:
        return None
    return {
        "id": decision.id,
        "route": _enum_value(decision.route),
        "task_id": decision.task_id,
        "summary": decision.summary,
        "issues": list(decision.issues),
    }


def _run_graph(graph: Any, input_state: dict[str, Any], *, stop_after_first_edit: bool) -> tuple[dict[str, Any], str]:
    if not stop_after_first_edit:
        return graph.invoke(input_state), "graph_terminal"

    latest_state: dict[str, Any] | None = None
    for update in graph.stream(input_state, stream_mode="values"):
        latest_state = update
        if _find_latest_edit_output_id(update):
            return update, "first_edit_candidate"
    if latest_state is None:
        raise RuntimeError("agent graph produced no states")
    return latest_state, "graph_terminal"


def _build_input_state(args: argparse.Namespace, image_paths: list[Path]) -> dict[str, Any]:
    return {
        "input": {
            "session_id": args.session_id,
            "image_uri": str(image_paths[0]),
            "image_uris": [str(image_path) for image_path in image_paths],
            "instruction_text": args.instruction,
            "desired_decision_route": args.desired_decision_route,
            "use_llm": args.use_llm,
        },
        "max_task_loops": args.max_task_loops,
        "max_execute_acts": args.max_execute_acts,
        "max_evaluator_checkpoints": args.max_evaluator_checkpoints,
        "max_tool_failures": args.max_tool_failures,
    }


def _build_summary(state: dict[str, Any], stop_reason: str) -> dict[str, Any]:
    session = state["session"]
    final_artifact_id = session.final_result_id or _find_latest_edit_output_id(state) or _latest_image_artifact_id(state)
    return {
        "run_id": state.get("run_id"),
        "output_dir": state.get("output_dir"),
        "run_log_uri": state.get("run_log_uri"),
        "message_log_uri": state.get("message_log_uri"),
        "stop_reason": stop_reason,
        "session_phase": _enum_value(session.phase),
        "current_plan_id": session.current_plan_id,
        "current_task_id": session.current_task_id,
        "latest_decision_id": session.latest_decision_id,
        "final_result_id": session.final_result_id,
        "fallback_final_image_id": final_artifact_id,
        "decision": _summarize_decision(state),
        "final_artifact": _summarize_artifact(state, final_artifact_id),
        "tool_results": _summarize_tool_results(state),
    }


def format_final_summary(summary: dict[str, Any]) -> str:
    lines = [
        "",
        "运行摘要",
        f"- Run ID: {summary.get('run_id') or '未知'}",
        f"- 输出目录: {summary.get('output_dir') or '未创建'}",
        f"- 日志文件: {summary.get('run_log_uri') or '未写入'}",
        f"- 停止原因: {_stop_reason_label(summary.get('stop_reason'))}",
        f"- 最终阶段: {summary.get('session_phase') or '未知'}",
    ]

    decision = summary.get("decision")
    if decision:
        lines.extend(
            [
                "",
                "评估结论",
                f"- 路由: {decision.get('route') or '未知'}",
                f"- 说明: {_short_text(decision.get('summary')) or '暂无'}",
                f"- 问题: {_format_refs(decision.get('issues'))}",
            ]
        )

    final_artifact = summary.get("final_artifact")
    lines.append("")
    lines.append("最终图片")
    if final_artifact:
        payload = final_artifact.get("payload") or {}
        lines.extend(
            [
                f"- Artifact: {final_artifact.get('id')}",
                f"- 类型: {final_artifact.get('kind')}",
                f"- 路径: {final_artifact.get('uri') or '无'}",
                f"- 来源工具: {final_artifact.get('created_by') or '未知'}",
                f"- 源产物: {_format_refs(final_artifact.get('source_ids'))}",
            ]
        )
        role = payload.get("role")
        instruction = payload.get("task_instruction")
        if role:
            lines.append(f"- 角色: {role}")
        if instruction:
            lines.append(f"- 对应任务: {_short_text(instruction)}")
    else:
        lines.append("- 未生成最终图片产物")

    tool_results = summary.get("tool_results") or []
    lines.append("")
    lines.append("操作流水")
    if not tool_results:
        lines.append("- 无工具调用记录")
    else:
        for index, tool_result in enumerate(tool_results, start=1):
            status = tool_result.get("status") or "unknown"
            refs = _format_refs(tool_result.get("output_refs"))
            line = (
                f"- {index}. {_tool_label(tool_result.get('tool_name'))} "
                f"[{status}] task={tool_result.get('task_id')} loop={tool_result.get('loop_index')} -> {refs}"
            )
            error = tool_result.get("error")
            if error:
                line += f"；错误={error.get('type')}: {_short_text(error.get('message'))}"
            lines.append(line)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fig edit agent runtime.")
    parser.add_argument("--images", nargs="+", required=True, help="Input/reference image paths.")
    parser.add_argument("--instruction", required=True, help="User edit instruction.")
    parser.add_argument("--session-id", default="agent-cli", help="Session id used in logs and state.")
    parser.add_argument(
        "--desired-decision-route",
        default="pass",
        choices=["pass", "needs_revision", "replan", "fail"],
        help="Optional target route for deterministic/fallback evaluators.",
    )
    parser.add_argument("--max-task-loops", type=_positive_int, default=2, help="Maximum execute/evaluate loops per task.")
    parser.add_argument(
        "--max-execute-acts",
        type=_positive_int,
        default=default_max_execute_acts(),
        help="Maximum thinking-act-observe rounds per execute node.",
    )
    parser.add_argument(
        "--max-evaluator-checkpoints",
        type=_positive_int,
        default=1,
        help="Maximum evaluator checkpoints before replanning/finalizing.",
    )
    parser.add_argument(
        "--max-tool-failures",
        type=_positive_int,
        default=default_max_tool_failures(),
        help="Maximum tolerated tool failures.",
    )
    parser.add_argument("--stop-after-first-edit", action="store_true", help="Return after the first edit artifact is created.")
    use_llm_group = parser.add_mutually_exclusive_group()
    use_llm_group.add_argument("--use-llm", dest="use_llm", action="store_true", help="Use configured LLM clients.")
    use_llm_group.add_argument("--no-use-llm", dest="use_llm", action="store_false", help="Use rule-based fallback paths.")
    parser.set_defaults(use_llm=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agent from the command line."""

    load_dotenv(REPO_ROOT / ".env")
    parser = _build_parser()
    args = parser.parse_args(argv)
    image_paths = _resolve_image_paths(args.images)
    graph = create_agent()
    try:
        result, stop_reason = _run_graph(
            graph,
            _build_input_state(args, image_paths),
            stop_after_first_edit=args.stop_after_first_edit,
        )
        summary = _build_summary(result, stop_reason)
        print(format_final_summary(summary))

        session = result["session"]
        if stop_reason == "graph_terminal" and session.phase == SessionPhase.FAILED:
            print("agent ended in failed phase", file=sys.stderr)
            return 1
        if summary["final_artifact"] is None:
            print("agent did not produce a final image artifact", file=sys.stderr)
            return 1
        return 0
    finally:
        unload_pipeline()


agent = create_agent()
"""Default compiled runtime graph for the fig edit agent."""


if __name__ == "__main__":
    raise SystemExit(main())
