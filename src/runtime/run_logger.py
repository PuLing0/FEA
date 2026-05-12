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
MAX_CONSOLE_TEXT = 240


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _short_text(value: Any, *, limit: int = MAX_CONSOLE_TEXT) -> str:
    text = " ".join(_as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_refs(values: Any, *, limit: int = 5) -> str:
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


def _format_score(value: Any) -> str:
    if value is None:
        return "未知"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _display_time(record: dict[str, Any]) -> str:
    raw_timestamp = _as_text(record.get("timestamp"))
    try:
        return datetime.fromisoformat(raw_timestamp).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return raw_timestamp or "-"


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
            print(_format_console_line(record, state), file=sys.stdout, flush=True)


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
    summary = {
        "id": artifact.id,
        "kind": getattr(artifact.kind, "value", artifact.kind),
        "uri": artifact.uri,
        "created_by": artifact.created_by,
        "source_ids": list(artifact.source_ids),
        "role": getattr(artifact, "role", None),
    }
    if getattr(artifact.kind, "value", artifact.kind) == ArtifactKind.EVALUATION.value:
        payload = _as_dict(getattr(artifact, "payload", {}))
        summary["payload"] = {
            "verdict": payload.get("verdict"),
            "reason": payload.get("reason"),
            "candidate_ref": payload.get("candidate_ref"),
        }
        summary["summary"] = getattr(artifact, "summary", None)
    return summary


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


def summarize_task(task: Any | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "id": getattr(task, "id", None),
        "type": getattr(task, "type", None),
        "instruction": getattr(task, "instruction", None),
        "input_artifact_ids": list(getattr(task, "input_artifact_ids", [])),
        "depends_on": list(getattr(task, "depends_on", [])),
        "acceptance_criteria": list(getattr(task, "acceptance_criteria", [])),
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


def _format_console_line(record: dict[str, Any], state: dict[str, Any] | None = None) -> str:
    record = _enrich_console_record(record, state)
    prefix = f"[agent:{record['run_id']} {_display_time(record)}]"
    body = _format_console_body(record)
    context = _format_console_context(record)
    return f"{prefix} {body}{context}"


def _enrich_console_record(record: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return record
    enriched = dict(record)
    payload = dict(_as_dict(record.get("payload")))
    event = _as_text(record.get("event"))
    if event == "run_start":
        runtime_input = _as_dict(state.get("input"))
        if "instruction_text" not in payload:
            payload["instruction_text"] = runtime_input.get("instruction_text")
    elif event == "artifact_created":
        artifact = _as_dict(state.get("artifacts")).get(payload.get("id"))
        if artifact is not None:
            if "summary" not in payload:
                payload["summary"] = getattr(artifact, "summary", None)
            if "payload" not in payload:
                payload["payload"] = getattr(artifact, "payload", {})
    elif event in {"operation_succeeded", "operation_failed"}:
        operation_id = payload.get("id")
        for operation in state.get("operations", []):
            if getattr(operation, "id", None) == operation_id:
                if "result_payload" not in payload:
                    payload["result_payload"] = getattr(operation, "result_payload", None)
                break
    current_task_id = record.get("current_task_id")
    if current_task_id and "current_task" not in payload:
        task = _as_dict(state.get("tasks")).get(current_task_id)
        task_summary = summarize_task(task)
        if task_summary is not None:
            payload["current_task"] = task_summary
    enriched["payload"] = payload
    return enriched


def _format_console_context(record: dict[str, Any]) -> str:
    context = []
    phase = record.get("phase")
    task_id = record.get("current_task_id")
    payload = _as_dict(record.get("payload"))
    current_task = _as_dict(payload.get("current_task"))
    if phase:
        context.append(f"阶段={phase}")
    if task_id:
        task_instruction = _short_text(current_task.get("instruction"), limit=120)
        task_type = _as_text(current_task.get("type"))
        if task_instruction:
            type_suffix = f" [{task_type}]" if task_type else ""
            context.append(f"任务={task_id}{type_suffix}: {task_instruction}")
        else:
            context.append(f"任务={task_id}")
    if not context:
        return ""
    return f" ({'，'.join(context)})"


def _format_console_body(record: dict[str, Any]) -> str:
    event = _as_text(record.get("event"))
    payload = _as_dict(record.get("payload"))
    if event == "run_start":
        return _format_run_start(payload)
    if event == "node_start":
        return f"进入节点：{_node_label(payload.get('node'))}"
    if event == "node_end":
        return f"完成节点：{_node_label(payload.get('node'))}"
    if event == "artifact_created":
        return _format_artifact_created(payload)
    if event == "operation_succeeded":
        return _format_operation(payload, failed=False)
    if event == "operation_failed":
        return _format_operation(payload, failed=True)
    if event == "plan_created":
        return _format_plan_created(payload)
    if event == "execute_checkpoint":
        return _format_execute_checkpoint(payload)
    if event == "evaluate_decision":
        return _format_evaluate_decision(payload)
    if event == "route_next":
        return f"下一步：{_route_label(payload.get('route'))}"
    if event == "run_end":
        return f"运行结束：{_route_label(payload.get('route'))}"
    return _format_generic_event(event, payload)


def _format_run_start(payload: dict[str, Any]) -> str:
    image_uris = payload.get("image_uris") or []
    use_llm = "开启" if payload.get("use_llm") else "关闭"
    log_uri = payload.get("log_uri") or "未写入文件"
    instruction = _short_text(payload.get("instruction_text"))
    pieces = [f"开始运行：收到 {len(image_uris)} 张图片", f"LLM={use_llm}", f"日志={log_uri}"]
    if instruction:
        pieces.append(f"指令：{instruction}")
    return "；".join(pieces)


def _node_label(node: Any) -> str:
    labels = {
        "register_and_understand": "注册输入并理解图片",
        "plan": "规划任务",
        "execute": "执行当前任务",
        "evaluate": "评估候选结果",
    }
    node_text = _as_text(node)
    return labels.get(node_text, node_text or "未知节点")


def _route_label(route: Any) -> str:
    labels = {
        "plan": "回到规划",
        "execute": "继续执行",
        "evaluate": "进入评估",
        "end": "结束",
    }
    route_text = _as_text(route)
    return labels.get(route_text, route_text or "未知")


def _format_plan_created(payload: dict[str, Any]) -> str:
    plan_ids = _format_refs(payload.get("plan_ids"))
    task_ids = _format_refs(payload.get("task_ids"))
    current_task = payload.get("current_task_id") or "无"
    task_summaries = [
        task for task in (_as_dict(item) for item in payload.get("task_summaries") or []) if task
    ]
    pieces = [f"创建计划：计划={plan_ids}", f"任务={task_ids}", f"当前任务={current_task}"]
    if task_summaries:
        task_details = []
        for task in task_summaries:
            task_id = task.get("id") or "unknown_task"
            task_type = _as_text(task.get("type"))
            instruction = _short_text(task.get("instruction"), limit=140)
            detail = f"{task_id}"
            if task_type:
                detail += f"[{task_type}]"
            if instruction:
                detail += f": {instruction}"
            criteria = _format_refs(task.get("acceptance_criteria"), limit=2)
            if criteria != "无":
                detail += f"；验收={criteria}"
            task_details.append(detail)
        pieces.append("任务内容=" + " | ".join(task_details))
    return "；".join(pieces)


def _format_execute_checkpoint(payload: dict[str, Any]) -> str:
    task_state = _as_dict(payload.get("task_state"))
    if not task_state:
        return "执行检查点：当前没有任务状态"
    checkpoint = task_state.get("latest_execute_checkpoint") or "未知"
    status = task_state.get("status") or "未知"
    latest_refs = _format_refs(task_state.get("latest_artifact_ids"))
    instruction = _short_text(task_state.get("active_instruction_preview"))
    pieces = [
        f"执行检查点：任务 {task_state.get('task_id')} 状态={status}",
        f"结论={checkpoint}",
        f"最新产物={latest_refs}",
    ]
    if instruction:
        pieces.append(f"当前指令：{instruction}")
    return "；".join(pieces)


def _format_evaluate_decision(payload: dict[str, Any]) -> str:
    decision = _as_dict(payload.get("decision"))
    route = payload.get("decision_route") or decision.get("route") or "未知"
    evaluation_verdict = payload.get("evaluation_verdict")
    summary = _short_text(decision.get("summary"))
    issues = _format_refs(decision.get("issues"))
    pieces = [f"评估决策：{route}"]
    if evaluation_verdict:
        pieces.append(f"结论：{evaluation_verdict}")
    if summary:
        pieces.append(f"原因：{summary}")
    if issues != "无":
        pieces.append(f"问题：{issues}")
    return "；".join(pieces)


def _format_artifact_created(payload: dict[str, Any]) -> str:
    artifact_id = payload.get("id") or "unknown_artifact"
    kind = _as_text(payload.get("kind"))
    role = payload.get("role") or _as_dict(payload.get("payload")).get("role")
    created_by = _as_text(payload.get("created_by"))
    uri = payload.get("uri")
    artifact_payload = _as_dict(payload.get("payload"))
    summary = payload.get("summary") or artifact_payload.get("summary")

    if kind == "image" and role == "initial_input":
        slot = artifact_payload.get("slot_index")
        slot_text = f"第 {slot} 张" if slot else "一张"
        return f"读取输入图片：{slot_text}图片 {artifact_id}，路径={uri}"
    if kind == "understanding":
        image_ref = artifact_payload.get("image_ref") or _format_refs(payload.get("source_ids"))
        return f"形成图片理解：{artifact_id}，图片={image_ref}，理解={_short_text(summary) or '暂无摘要'}"
    if kind == "instruction":
        text = summary or artifact_payload.get("instruction_text") or artifact_payload.get("task_instruction")
        return f"记录指令：{artifact_id}，内容={_short_text(text) or '暂无内容'}"
    if kind == "geometry":
        return _format_geometry_artifact(artifact_id, artifact_payload)
    if kind == "mask":
        return _format_mask_artifact(artifact_id, uri, artifact_payload)
    if kind == "evaluation":
        return _format_evaluation_artifact(artifact_id, artifact_payload, summary)
    if kind == "image":
        return _format_image_artifact(artifact_id, uri, role, created_by, artifact_payload, payload)
    return f"创建产物：{artifact_id}，类型={kind or '未知'}，来源={created_by or '未知'}"


def _format_geometry_artifact(artifact_id: str, artifact_payload: dict[str, Any]) -> str:
    candidates = artifact_payload.get("candidates") or []
    first = _as_dict(candidates[0]) if candidates else {}
    bbox = first.get("bbox")
    score = _format_score(first.get("score"))
    query = _short_text(artifact_payload.get("grounding_query"))
    image_ref = artifact_payload.get("image_artifact_id") or "未知图片"
    details = f"bbox={bbox}, score={score}" if bbox else "暂无候选框"
    return f"定位目标：{artifact_id}，图片={image_ref}，目标={query or '未命名'}，{details}"


def _format_mask_artifact(artifact_id: str, uri: Any, artifact_payload: dict[str, Any]) -> str:
    prompt = _short_text(artifact_payload.get("prompt") or artifact_payload.get("text_prompt"))
    score = _format_score(artifact_payload.get("mask_score"))
    image_ref = artifact_payload.get("image_ref") or "未知图片"
    return f"生成分割蒙版：{artifact_id}，图片={image_ref}，提示={prompt or '无'}，score={score}，路径={uri}"


def _format_evaluation_artifact(artifact_id: str, artifact_payload: dict[str, Any], summary: Any) -> str:
    verdict = artifact_payload.get("verdict") or "未知"
    reason = _short_text(summary or artifact_payload.get("reason"))
    return f"完成视觉评估：{artifact_id}，结论={verdict}，原因={reason or '暂无'}"


def _format_image_artifact(
    artifact_id: str,
    uri: Any,
    role: Any,
    created_by: str,
    artifact_payload: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    if role == "candidate_image" or created_by == "edit":
        refs = artifact_payload.get("primary_image_ref") or _format_refs(payload.get("source_ids"))
        instruction = _short_text(artifact_payload.get("task_instruction"))
        pieces = [f"生成候选图片：{artifact_id}，路径={uri}", f"输入={refs}"]
        if instruction:
            pieces.append(f"任务={instruction}")
        return "；".join(pieces)
    if role == "cropped_preview" or created_by == "crop":
        return (
            f"生成裁剪预览：{artifact_id}，路径={uri}，"
            f"bbox={artifact_payload.get('bbox')}，padding={artifact_payload.get('padding')}"
        )
    if role == "collage_reference" or created_by == "collage":
        canvas = _as_dict(artifact_payload.get("canvas"))
        canvas_text = f"{canvas.get('width')}x{canvas.get('height')}" if canvas else "未知尺寸"
        blocks = _format_refs(artifact_payload.get("block_artifact_ids"))
        return f"生成拼图参考图：{artifact_id}，路径={uri}，画布={canvas_text}，输入={blocks}"
    source_ids = _format_refs(payload.get("source_ids"))
    return f"生成图片产物：{artifact_id}，路径={uri}，来源工具={created_by or '未知'}，源={source_ids}"


def _format_operation(payload: dict[str, Any], *, failed: bool) -> str:
    tool_name = _as_text(payload.get("tool_name")) or "unknown_tool"
    if failed:
        error = _as_dict(payload.get("error"))
        error_type = error.get("type") or "Error"
        message = _short_text(error.get("message"))
        return f"工具失败：{_tool_label(tool_name)}，错误={error_type}: {message or '未知错误'}"

    args = _as_dict(payload.get("args"))
    result_payload = _as_dict(payload.get("result_payload"))
    output_refs = payload.get("output_refs")
    if tool_name == "understand":
        summary = _short_text(result_payload.get("summary"))
        return (
            f"工具完成：理解图片 {result_payload.get('image_ref') or args.get('image_ref')} "
            f"-> {_format_refs(output_refs)}，理解={summary or '暂无摘要'}"
        )
    if tool_name == "grounding":
        query = _short_text(result_payload.get("grounding_query") or args.get("grounding_query"))
        return f"工具完成：定位目标，图片={result_payload.get('image_ref') or args.get('image_ref')}，目标={query}，输出={_format_refs(output_refs)}"
    if tool_name == "segment":
        prompt = _short_text(result_payload.get("prompt") or args.get("prompt"))
        return (
            f"工具完成：分割图片 {result_payload.get('image_ref') or args.get('image_ref')}，"
            f"提示={prompt}，mask={result_payload.get('mask_ref') or _format_refs(output_refs)}，"
            f"score={_format_score(result_payload.get('mask_score'))}"
        )
    if tool_name == "crop":
        return (
            f"工具完成：裁剪图片 {result_payload.get('image_ref') or args.get('image_ref')}，"
            f"输出={result_payload.get('crop_ref') or _format_refs(output_refs)}"
        )
    if tool_name == "collage":
        goal = _short_text(result_payload.get("layout_goal") or args.get("layout_goal"))
        output_path = result_payload.get("output_path") or "见产物记录"
        return f"工具完成：生成拼图参考图，目标={goal}，输出={result_payload.get('collage_artifact_id') or _format_refs(output_refs)}，路径={output_path}"
    if tool_name == "prompt_reconstruct":
        return f"工具完成：重写任务指令，输出={_format_refs(output_refs)}"
    if tool_name == "edit":
        instruction = _short_text(result_payload.get("instruction") or args.get("instruction"))
        refs = _format_refs(result_payload.get("image_refs") or args.get("image_refs"))
        return f"工具完成：生成编辑候选图，输入={refs}，输出={result_payload.get('output_image_ref') or _format_refs(output_refs)}，指令={instruction}"
    if tool_name == "evaluate":
        verdict = result_payload.get("verdict") or "未知"
        reason = _short_text(result_payload.get("reason"))
        candidate_ref = result_payload.get("candidate_ref") or args.get("candidate_ref")
        return f"工具完成：评估候选图 {candidate_ref}，结论={verdict}，原因={reason or '暂无'}"
    return f"工具完成：{_tool_label(tool_name)}，输出={_format_refs(output_refs)}"


def _tool_label(tool_name: str) -> str:
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
    return labels.get(tool_name, tool_name)


def _format_generic_event(event: str, payload: dict[str, Any]) -> str:
    if not payload:
        return f"事件：{event}"
    details = []
    for key, value in payload.items():
        if value is None:
            continue
        details.append(f"{key}={_short_text(value, limit=80)}")
        if len(details) >= 4:
            break
    return f"事件：{event}；" + "；".join(details)
