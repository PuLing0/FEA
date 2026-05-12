from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_build_candidate_images_text_uses_compact_scheme_b() -> None:
    state = {
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                summary="上衣单品图",
                scope="session",
            ),
            "art_image_001": ImageArtifact(
                id="art_image_001",
                uri="store://generated/task_001.png",
                payload={"role": "candidate_image"},
                summary="已合成的人物主体图",
                scope="task",
            ),
        },
        "operations": [],
    }
    text = build_candidate_images_text(state, ["art_img_input_001", "art_image_001"])
    assert "[art_img_input_001] 上衣单品图 | 原始输入图片" in text
    assert "[art_image_001] 已合成的人物主体图 | source=generated_result" in text
    assert "uri:" not in text


def test_build_candidate_image_pool_includes_current_plan_inputs() -> None:
    state = {
        "session": SessionState(
            session_id="sess_plan_candidates",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="测试 plan 输入池",
                task_ids=["task_001"],
                input_artifact_ids=[
                    "art_plan_visible_001",
                    "art_not_image_001",
                    "art_img_input_001",
                ],
            )
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="edit",
                instruction="执行当前任务",
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_plan_visible_001": ImageArtifact(
                id="art_plan_visible_001",
                uri="store://images/retained.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
            "art_not_image_001": InstructionArtifact(
                id="art_not_image_001",
                payload={"instruction_text": "not an image"},
                scope="task",
            ),
        },
    }

    assert build_candidate_image_pool(state) == [
        "art_img_input_001",
        "art_plan_visible_001",
    ]


def test_prepare_task_inputs_prefers_retry_input_artifact_ids() -> None:
    state = {
        "input": {"use_llm": False},
        "session": SessionState(
            session_id="sess_retry_inputs",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=["art_img_input_001", "art_image_candidate_001", "art_inst_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正",
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://images/candidate.png",
                payload={"role": "candidate_image"},
            ),
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={"instruction_text": "继续修正当前结果"},
            ),
        },
        "operations": [],
    }

    selection = TaskInputSelectionOutput.model_validate(
        {"selected_artifact_ids": ["placeholder"]}
    )
    from runtime.input_selector import prepare_task_inputs

    result = prepare_task_inputs(state, "task_001")

    assert result.selected_artifact_ids == ["art_img_input_001", "art_image_candidate_001", "art_inst_001"]
    assert state["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_image_candidate_001",
        "art_inst_001",
    ]
    assert state["session"].task_states["task_001"].input_selection_reasoning == "resolved from evaluator retry context"


def test_prepare_task_inputs_uses_llm_selector_and_caps_image_count(mocker) -> None:
    from runtime.input_selector import prepare_task_inputs

    state = {
        "input": {"use_llm": True},
        "session": SessionState(
            session_id="sess_initial_selector",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            session_working_set=[
                WorkingSetEntry(artifact_id="art_instruction_session_root_001", usage="root"),
                *[
                    WorkingSetEntry(artifact_id=f"art_img_input_{index:03d}", usage="input")
                    for index in range(1, 6)
                ],
            ],
            artifact_index=ArtifactIndex(
                by_type={
                    ArtifactKind.IMAGE: [f"art_img_input_{index:03d}" for index in range(1, 6)]
                }
            ),
        ),
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="选择最相关的三张图进行编辑",
            )
        },
        "artifacts": {
            "art_instruction_session_root_001": InstructionArtifact(
                id="art_instruction_session_root_001",
                payload={"instruction_text": "root"},
                role="session_root_instruction",
            ),
            **{
                f"art_img_input_{index:03d}": ImageArtifact(
                    id=f"art_img_input_{index:03d}",
                    uri=f"store://images/{index}.png",
                    payload={"role": "input"},
                    scope="session",
                )
                for index in range(1, 6)
            },
        },
        "operations": [],
    }

    mocker.patch("runtime.input_selector.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    selector_mock = mocker.patch(
        "runtime.input_selector.invoke_structured_llm",
        return_value=TaskInputSelectionOutput(
            selected_artifact_ids=[
                "art_instruction_session_root_001",
                "art_img_input_001",
                "art_img_input_002",
                "art_img_input_003",
                "art_img_input_004",
            ],
            working_set_entries=[
                WorkingSetEntry(artifact_id="art_instruction_session_root_001", usage="root"),
                WorkingSetEntry(artifact_id="art_img_input_001", usage="person"),
                WorkingSetEntry(artifact_id="art_img_input_002", usage="top"),
                WorkingSetEntry(artifact_id="art_img_input_003", usage="background"),
                WorkingSetEntry(artifact_id="art_img_input_004", usage="extra"),
            ],
        ),
    )

    result = prepare_task_inputs(state, "task_001")

    selector_mock.assert_called_once()
    assert result.selected_artifact_ids == [
        "art_instruction_session_root_001",
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]
    assert state["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]


def test_prepare_task_inputs_fallback_prioritizes_static_task_inputs() -> None:
    from runtime.input_selector import prepare_task_inputs

    state = {
        "input": {"use_llm": False},
        "session": SessionState(
            session_id="sess_static_inputs",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            session_working_set=[
                WorkingSetEntry(artifact_id="art_img_input_001", usage="session"),
                WorkingSetEntry(artifact_id="art_img_input_002", usage="session"),
                WorkingSetEntry(artifact_id="art_img_input_003", usage="session"),
                WorkingSetEntry(artifact_id="art_img_input_004", usage="session"),
            ],
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: []}),
        ),
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="place_subject_in_background",
                instruction="把人物放到背景中",
                input_artifact_ids=["art_img_input_004"],
            )
        },
        "artifacts": {
            f"art_img_input_{index:03d}": ImageArtifact(
                id=f"art_img_input_{index:03d}",
                uri=f"store://images/{index}.png",
                payload={"role": "input"},
                scope="session",
            )
            for index in range(1, 5)
        },
        "operations": [],
    }

    result = prepare_task_inputs(state, "task_001")

    assert result.selected_artifact_ids[:1] == ["art_img_input_004"]
    assert state["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_004",
        "art_img_input_001",
        "art_img_input_002",
    ]


def test_prepare_task_inputs_caps_existing_working_set_image_count() -> None:
    from runtime.input_selector import prepare_task_inputs

    state = {
        "input": {"use_llm": False},
        "session": SessionState(
            session_id="sess_existing_working_set",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_working_set=[
                        WorkingSetEntry(artifact_id=f"art_img_input_{index:03d}", usage="image")
                        for index in range(1, 6)
                    ],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: []}),
        ),
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="编辑图片",
            )
        },
        "artifacts": {
            f"art_img_input_{index:03d}": ImageArtifact(
                id=f"art_img_input_{index:03d}",
                uri=f"store://images/{index}.png",
                payload={"role": "input"},
                scope="session",
            )
            for index in range(1, 6)
        },
        "operations": [],
    }

    result = prepare_task_inputs(state, "task_001")

    assert result.selected_artifact_ids == [
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]
    assert state["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]
