from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_runtime_run_logger_writes_jsonl(tmp_path, monkeypatch, capsys) -> None:
    import json

    monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LOG_ENABLED", "true")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "true")

    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "log-smoke",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "记录一次日志",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    captured = capsys.readouterr()
    assert "[agent:" in captured.out
    assert "开始运行" in captured.out
    assert "读取输入图片" in captured.out
    assert "形成图片理解" in captured.out
    assert "创建计划" in captured.out
    assert "任务内容=" in captured.out
    assert "task_001[compose_subject]" in captured.out
    assert result["run_id"]
    assert result["output_dir"] is not None
    assert result["run_log_uri"] is not None
    assert result["message_log_uri"] is not None

    output_dir = Path(result["output_dir"])
    assert output_dir.is_dir()
    assert output_dir.parent == tmp_path
    log_path = Path(result["run_log_uri"])
    message_log_path = Path(result["message_log_uri"])
    assert log_path == output_dir / "logs" / "run.jsonl"
    assert message_log_path == output_dir / "logs" / "messages.jsonl"
    assert log_path.is_file()
    assert message_log_path.is_file()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["event"] == "run_start"
    assert records[0]["payload"]["output_dir"] == str(output_dir)
    assert any(record["event"] == "node_start" and record["payload"]["node"] == "plan" for record in records)
    assert any(record["event"] == "plan_created" for record in records)
    assert all(record["run_id"] == result["run_id"] for record in records)


def test_runtime_run_logger_formats_structured_events_for_humans() -> None:
    from runtime.run_logger import _format_console_line

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "artifact_created",
        "phase": "understanding",
        "current_plan_id": None,
        "current_task_id": "task_001",
        "payload": {
            "id": "art_understanding_001",
            "kind": "understanding",
            "uri": None,
            "created_by": "understand",
            "source_ids": ["art_img_input_001"],
            "role": "image_understanding",
            "summary": "图片里是一名穿白色上衣的人。",
            "payload": {
                "image_ref": "art_img_input_001",
                "summary": "图片里是一名穿白色上衣的人。",
            },
        },
    }

    line = _format_console_line(record)

    assert "形成图片理解" in line
    assert "art_img_input_001" in line
    assert "图片里是一名穿白色上衣的人" in line


def test_runtime_run_logger_formats_plan_tasks_for_humans() -> None:
    from runtime.run_logger import _format_console_line

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "plan_created",
        "phase": "executing",
        "current_plan_id": "plan_001",
        "current_task_id": "task_001",
        "payload": {
            "plan_ids": ["plan_001"],
            "task_ids": ["task_001", "task_002"],
            "current_task_id": "task_001",
            "task_summaries": [
                {
                    "id": "task_001",
                    "type": "prepare_subject_mask",
                    "instruction": "分割人物主体并生成干净蒙版。",
                    "input_artifact_ids": ["art_img_input_001"],
                    "depends_on": [],
                    "acceptance_criteria": ["人物主体完整", "背景不被误选"],
                },
                {
                    "id": "task_002",
                    "type": "edit_image",
                    "instruction": "把人物换成蓝色外套。",
                    "input_artifact_ids": ["art_img_input_001"],
                    "depends_on": ["task_001"],
                    "acceptance_criteria": ["外套颜色正确"],
                },
            ],
        },
    }

    line = _format_console_line(record)

    assert "创建计划" in line
    assert "task_001[prepare_subject_mask]: 分割人物主体并生成干净蒙版" in line
    assert "task_002[edit_image]: 把人物换成蓝色外套" in line
    assert "验收=人物主体完整, 背景不被误选" in line


def test_runtime_run_logger_summarizes_evaluation_artifact_payload() -> None:
    from runtime.run_logger import summarize_artifact

    artifact = EvaluationArtifact(
        id="art_evaluation_001",
        payload={
            "verdict": "pass_with_issues",
            "reason": "核心目标已完成，但仍有轻微问题。",
            "candidate_ref": "art_image_001",
        },
        summary="核心目标已完成，但仍有轻微问题。",
        created_by=ToolName.EVALUATE.value,
        source_ids=["art_image_001"],
        role="evaluation_feedback",
    )

    summary = summarize_artifact(artifact)

    assert summary["payload"]["verdict"] == "pass_with_issues"
    assert summary["payload"]["reason"] == "核心目标已完成，但仍有轻微问题。"
    assert summary["payload"]["candidate_ref"] == "art_image_001"


def test_runtime_run_logger_formats_evaluation_verdict_for_humans() -> None:
    from runtime.run_logger import _format_console_line

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "evaluate_decision",
        "phase": "done",
        "current_plan_id": "plan_001",
        "current_task_id": None,
        "payload": {
            "decision_route": "pass",
            "evaluation_verdict": "pass_with_issues",
            "decision": {
                "route": "pass",
                "summary": "核心目标已完成，但仍有轻微问题。",
                "issues": ["minor artifact"],
            },
        },
    }

    line = _format_console_line(record)

    assert "评估决策：pass" in line
    assert "结论：pass_with_issues" in line
    assert "原因：核心目标已完成，但仍有轻微问题。" in line


def test_runtime_run_logger_context_includes_current_task_instruction() -> None:
    from runtime.run_logger import _format_console_line
    from schema import Task

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "node_start",
        "phase": "executing",
        "current_plan_id": "plan_001",
        "current_task_id": "task_001",
        "payload": {"node": "execute"},
    }
    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="edit_image",
                instruction="把人物换成蓝色外套，并保持背景不变。",
                input_artifact_ids=["art_img_input_001"],
                acceptance_criteria=["外套颜色正确"],
            )
        }
    }

    line = _format_console_line(record, state)

    assert "进入节点：执行当前任务" in line
    assert "任务=task_001 [edit_image]: 把人物换成蓝色外套，并保持背景不变" in line


def test_runtime_run_logger_formats_failed_tool_event_for_humans() -> None:
    from runtime.run_logger import _format_console_line

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "operation_failed",
        "phase": "executing",
        "current_plan_id": "plan_001",
        "current_task_id": "task_001",
        "payload": {
            "id": "op_edit_001",
            "tool_name": "edit",
            "status": "failed",
            "args": {"image_refs": ["art_img_input_001"], "instruction": "换衣服"},
            "output_refs": [],
            "error": {"type": "RuntimeError", "message": "backend unavailable"},
        },
    }

    line = _format_console_line(record)

    assert "工具失败：图像编辑" in line
    assert "RuntimeError: backend unavailable" in line


def test_runtime_run_logger_formats_understand_tool_result_with_summary() -> None:
    from runtime.message_query import summarize_tool_result
    from runtime.run_logger import _format_console_line
    from schema.messages import ToolResultBlock

    payload = summarize_tool_result(
        ToolResultBlock(
            tool_call_id="call_understand_001",
            tool_name=ToolName.UNDERSTAND,
            task_id="bootstrap",
            loop_index=0,
            status="succeeded",
            args={"image_ref": "art_img_input_002"},
            artifact_ids=["art_understanding_002"],
            result_payload={
                "image_ref": "art_img_input_002",
                "summary": "图片里是一件黑色上衣。",
                "understanding_ref": "art_understanding_002",
            },
        )
    )
    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "operation_succeeded",
        "phase": "understanding",
        "current_plan_id": None,
        "current_task_id": None,
        "payload": payload,
    }

    line = _format_console_line(record)

    assert payload["output_refs"] == ["art_understanding_002"]
    assert payload["result_payload"]["summary"] == "图片里是一件黑色上衣。"
    assert "工具完成：理解图片 art_img_input_002 -> art_understanding_002" in line
    assert "理解=图片里是一件黑色上衣。" in line


def test_runtime_run_logger_uses_artifact_ids_fallback_for_tool_outputs() -> None:
    from runtime.run_logger import _format_console_line

    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "operation_succeeded",
        "phase": "understanding",
        "current_plan_id": None,
        "current_task_id": None,
        "payload": {
            "tool_call_id": "call_understand_001",
            "tool_name": "understand",
            "status": "succeeded",
            "args": {"image_ref": "art_img_input_002"},
            "artifact_ids": ["art_understanding_002"],
            "summary": "图片里是一件黑色上衣。",
        },
    }

    line = _format_console_line(record)

    assert "-> art_understanding_002" in line
    assert "理解=图片里是一件黑色上衣。" in line


def test_runtime_run_logger_formats_evaluate_tool_result_with_verdict() -> None:
    from runtime.message_query import summarize_tool_result
    from runtime.run_logger import _format_console_line
    from schema.messages import ToolResultBlock

    payload = summarize_tool_result(
        ToolResultBlock(
            tool_call_id="call_evaluate_001",
            tool_name=ToolName.EVALUATE,
            task_id="task_001",
            loop_index=1,
            status="succeeded",
            args={"candidate_ref": "art_image_007"},
            artifact_ids=["art_evaluation_001"],
            result_payload={
                "candidate_ref": "art_image_007",
                "verdict": "needs_revision",
                "reason": "Restore full head-to-toe framing.",
            },
        )
    )
    record = {
        "timestamp": "2026-05-07T01:02:03.000+00:00",
        "run_id": "run123",
        "session_id": "sess",
        "event": "operation_succeeded",
        "phase": "evaluating",
        "current_plan_id": "plan_001",
        "current_task_id": "task_001",
        "payload": payload,
    }

    line = _format_console_line(record)

    assert payload["output_refs"] == ["art_evaluation_001"]
    assert payload["result_payload"]["verdict"] == "needs_revision"
    assert "工具完成：评估候选图 art_image_007" in line
    assert "结论=needs_revision" in line
    assert "原因=Restore full head-to-toe framing." in line


def test_summarize_task_state_includes_active_instruction_fields() -> None:
    state = _make_instruction_resolution_state(
        artifacts={
            "art_instruction_task_001_001": InstructionArtifact(
                id="art_instruction_task_001_001",
                payload={"instruction_text": "使用任务局部指令"},
                role="task_instruction",
                scope="task",
            ),
            "art_instruction_session_root_001": InstructionArtifact(
                id="art_instruction_session_root_001",
                payload={"instruction_text": "使用全局根指令"},
                role="session_root_instruction",
                scope="session",
            ),
        },
        task_artifact_ids=[
            "art_instruction_task_001_001",
            "art_instruction_session_root_001",
        ],
    )

    summary = summarize_task_state(state["session"].task_states["task_001"], state)

    assert summary is not None
    assert summary["active_instruction_artifact_id"] == "art_instruction_task_001_001"
    assert summary["active_instruction_role"] == "task_instruction"
    assert summary["active_instruction_preview"] == "使用任务局部指令"


def test_runtime_run_logger_can_disable_file_and_console(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LOG_ENABLED", "false")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "false")

    result = build_runtime_graph(stop_after_plan=True).invoke(
        {
            "input": {
                "session_id": "log-disabled",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "不要写日志",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert result["run_id"]
    assert result["output_dir"] is not None
    assert result["run_log_uri"] is None
    assert Path(result["message_log_uri"]).is_file()
    assert Path(result["message_log_uri"]).parent == Path(result["output_dir"]) / "logs"
