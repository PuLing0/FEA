from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_instruction_resolver_prefers_task_instruction_over_later_session_root() -> None:
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

    artifact = resolve_active_instruction_artifact(state, "task_001")

    assert artifact is not None
    assert artifact.id == "art_instruction_task_001_001"
    assert resolve_active_instruction_text(state, "task_001") == "使用任务局部指令"


def test_instruction_resolver_prefers_latest_rewritten_instruction_across_roles() -> None:
    state = _make_instruction_resolution_state(
        artifacts={
            "art_instruction_rewrite_001": InstructionArtifact(
                id="art_instruction_rewrite_001",
                payload={"instruction_text": "旧改写指令", "task_id": "task_001"},
                role="rewritten_instruction",
                scope="task",
            ),
            "art_instruction_session_root_001": InstructionArtifact(
                id="art_instruction_session_root_001",
                payload={"instruction_text": "全局根指令"},
                role="session_root_instruction",
                scope="session",
            ),
            "art_instruction_task_001_001": InstructionArtifact(
                id="art_instruction_task_001_001",
                payload={"instruction_text": "任务局部指令"},
                role="task_instruction",
                scope="task",
            ),
            "art_instruction_rewrite_002": InstructionArtifact(
                id="art_instruction_rewrite_002",
                payload={"instruction_text": "最新改写指令", "task_id": "task_001"},
                role="rewritten_instruction",
                scope="task",
            ),
        },
        task_artifact_ids=[
            "art_instruction_rewrite_001",
            "art_instruction_session_root_001",
            "art_instruction_task_001_001",
            "art_instruction_rewrite_002",
        ],
    )

    artifact = resolve_active_instruction_artifact(state, "task_001")

    assert artifact is not None
    assert artifact.id == "art_instruction_rewrite_002"
    assert resolve_active_instruction_text(state, "task_001") == "最新改写指令"


def test_instruction_resolver_ignores_other_task_instruction_artifacts() -> None:
    state = _make_instruction_resolution_state(
        artifacts={
            "art_instruction_task_001_001": InstructionArtifact(
                id="art_instruction_task_001_001",
                payload={"instruction_text": "当前任务指令", "task_id": "task_001"},
                role="task_instruction",
                scope="task",
            ),
            "art_instruction_task_002_001": InstructionArtifact(
                id="art_instruction_task_002_001",
                payload={"instruction_text": "其他任务指令", "task_id": "task_002"},
                role="task_instruction",
                scope="task",
            ),
        },
        task_artifact_ids=[
            "art_instruction_task_001_001",
            "art_instruction_task_002_001",
        ],
    )

    artifact = resolve_active_instruction_artifact(state, "task_001")

    assert artifact is not None
    assert artifact.id == "art_instruction_task_001_001"
    assert resolve_active_instruction_text(state, "task_001") == "当前任务指令"


def test_instruction_resolver_ignores_unknown_roles_and_falls_back_to_task_instruction() -> None:
    state = _make_instruction_resolution_state(
        artifacts={
            "art_instruction_note_001": InstructionArtifact(
                id="art_instruction_note_001",
                payload={"instruction_text": "这不是有效执行指令"},
                role="note",
                scope="task",
            )
        },
        task_artifact_ids=["art_instruction_note_001"],
        task_instruction="回退到 Task.instruction",
    )

    assert resolve_active_instruction_artifact(state, "task_001") is None
    assert resolve_active_instruction_text(state, "task_001") == "回退到 Task.instruction"
