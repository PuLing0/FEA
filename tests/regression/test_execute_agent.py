from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_execute_agent_uses_llm_strategy_when_enabled(mocker) -> None:
    from PIL import Image

    state = {
        "input": {
            "session_id": "exec_llm",
            "instruction_text": "global recolor",
            "desired_decision_route": "pass",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="global recolor",
                input_artifact_ids=["art_img_input_001"],
                acceptance_criteria=["candidate exists"],
            )
        },
        "session": SessionState(
            session_id="exec_llm",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(
                by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}
            ),
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="global recolor",
                task_ids=["task_001"],
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="global edit is enough",
            selected_tools=["edit"],
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )
    mocker.patch(
        "runtime.input_selector.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )
    mocker.patch(
        "runtime.input_selector.invoke_structured_llm",
        return_value=TaskInputSelectionOutput(
            selected_artifact_ids=["art_img_input_001"],
        ),
    )
    mocker.patch(
        "runtime.input_selector.invoke_llm",
        return_value=mocker.Mock(content="Need one base image and no dependency result."),
    )
    mocker.patch("runtime.input_selector.ensure_understanding_for_images")
    mocker.patch(
        "tools.edit_tool.edit_images",
        return_value=Image.new("RGB", (16, 16), color=(0, 128, 255)),
    )
    mocker.patch(
        "tools.edit_tool.backend_config_snapshot",
        return_value={"model_path": "mock-firered"},
    )

    result = ExecuteAgent().run(state)

    latest_edit = latest_message_tool_result(result, tool_name=ToolName.EDIT)
    assert latest_edit is not None
    assert latest_edit.tool_name == ToolName.EDIT
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
    ]
    assert result["session"].task_states["task_001"].input_selection_reasoning is not None
    latest_refs = result["session"].task_states["task_001"].latest_artifact_ids
    assert len(latest_refs) == 1
    assert latest_refs[0].startswith("art_image_")
    assert latest_edit.args == {
        "instruction": "global recolor",
        "image_refs": ["art_img_input_001"],
    }
    assert message_observations(result, task_id="task_001")


def test_execute_agent_accepts_grounding_and_collage_strategy(mocker) -> None:
    from PIL import Image
    from tools.collage_tool import CollageLayoutItem, CollageLayoutResult
    from tools.grounding_tool import GroundingTool

    import tempfile

    tmpdir = tempfile.TemporaryDirectory()
    img1 = Path(tmpdir.name) / "1.png"
    img2 = Path(tmpdir.name) / "2.png"
    img3 = Path(tmpdir.name) / "3.png"
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img1)
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img2)
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img3)

    state = {
        "input": {
            "instruction_text": "先定位人物，再整理参考图，最后编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="完成图像编辑",
                input_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
            )
        },
        "session": SessionState(
            session_id="sess_exec_new_tools",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002", "art_img_input_003"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(img1),
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri=str(img2),
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_003": ImageArtifact(
                id="art_img_input_003",
                uri=str(img3),
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch("runtime.input_selector.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "runtime.input_selector.invoke_llm",
        return_value=mocker.Mock(content="Need base image and two references."),
    )
    mocker.patch(
        "runtime.input_selector.invoke_structured_llm",
        return_value=TaskInputSelectionOutput(
            selected_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
        ),
    )

    invoke_structured = mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        side_effect=[
            ExecuteLLMOutput(reasoning="Need coarse region first.", selected_tools=["grounding"]),
            ExecuteLLMOutput(reasoning="Need one combined reference image.", selected_tools=["collage"]),
            ExecuteLLMOutput(reasoning="Now edit with the prepared references.", selected_tools=["edit"]),
        ],
    )
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=32,
            canvas_height=16,
            background="transparent",
            items=[
                CollageLayoutItem(
                    artifact_id="art_img_input_002",
                    x=0,
                    y=0,
                    width=8,
                    height=8,
                    z_index=0,
                ),
                CollageLayoutItem(
                    artifact_id="art_img_input_003",
                    x=12,
                    y=0,
                    width=8,
                    height=8,
                    z_index=1,
                ),
                CollageLayoutItem(
                    artifact_id="art_img_input_001",
                    x=24,
                    y=0,
                    width=8,
                    height=8,
                    z_index=2,
                ),
            ],
        ),
    )

    def fake_grounding_execute(state, *, task_id, loop_index, args):
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.GROUNDING
        fake.invocation.args = args.model_dump()
        fake.invocation.status = "succeeded"
        fake.invocation.output_refs = ["art_geometry_001"]
        fake.artifacts = [
            GeometryArtifact(
                id="art_geometry_001",
                payload={
                    "image_artifact_id": "art_img_input_001",
                    "grounding_query": args.grounding_query,
                    "candidates": [
                        {
                            "label": "subject",
                            "bbox": [1, 1, 6, 6],
                            "score": 0.95,
                            "positive_points": [],
                            "negative_points": [],
                        }
                    ],
                },
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(GroundingTool, "execute", side_effect=fake_grounding_execute)

    def fake_edit_execute(state, *, task_id, loop_index, args):
        fake = mocker.Mock()
        fake.invocation = ToolInvocationRecord(
            id="op_edit_001",
            task_id=task_id,
            loop_index=loop_index,
            tool_name=ToolName.EDIT,
            args=args.model_dump(),
            status="succeeded",
            output_refs=["art_image_edit_001"],
        )
        fake.artifacts = [
            ImageArtifact(
                id="art_image_edit_001",
                uri=str(img1),
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch("tools.edit_tool.EditTool.execute", side_effect=fake_edit_execute)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        side_effect=[
            ObserveLLMOutput(outcome="continue", observation="grounding is useful", artifact_summaries=[]),
            ObserveLLMOutput(outcome="continue", observation="collage is useful", artifact_summaries=[]),
            ObserveLLMOutput(outcome="success", observation="candidate is ready", artifact_summaries=[]),
        ],
    )

    result = ExecuteAgent().run(state)
    tmpdir.cleanup()

    tool_names = [block.tool_name for block in message_tool_results(result)]
    assert ToolName.GROUNDING in tool_names
    assert ToolName.COLLAGE in tool_names
    assert message_tool_results(result)[-1].tool_name == ToolName.EDIT
    assert invoke_structured.call_count == 3


def test_execute_agent_crop_accepts_grounding_without_mask(mocker) -> None:
    state = {
        "artifacts": {},
        "session": SessionState(
            session_id="sess_crop_from_grounding",
            phase=SessionPhase.EXECUTING,
            task_states={},
        ),
    }

    captured = {}

    def fake_crop_execute(state, *, task_id, loop_index, args):
        captured["args"] = args
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.CROP
        fake.invocation.args = args.model_dump()
        fake.invocation.status = "succeeded"
        fake.invocation.output_refs = ["art_image_crop_001"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_crop_001",
                uri="examples/fig1.jpg",
                payload={"role": "cropped_preview"},
                scope="task",
            )
        ]
        return fake

    mocker.patch("tools.crop_tool.CropTool.execute", side_effect=fake_crop_execute)

    ExecuteAgent()._run_tool_step(
        state=state,
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.CROP,
        runtime_ctx={
            "initial_base_image_ref": "art_img_input_001",
            "mask_ref": None,
            "grounding_ref": "art_geometry_001",
        },
    )

    assert captured["args"].mask_ref is None
    assert captured["args"].grounding_ref == "art_geometry_001"


def test_execute_agent_segment_passes_grounding_ref(mocker) -> None:
    from tools.segment_tool import SegmentTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="segment the target object",
            )
        },
        "artifacts": {},
        "session": SessionState(
            session_id="sess_segment_from_grounding",
            phase=SessionPhase.EXECUTING,
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=[],
                )
            },
        ),
    }

    captured = {}

    def fake_segment_execute(state, *, task_id, loop_index, args):
        captured["args"] = args
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.SEGMENT
        fake.invocation.args = args.model_dump()
        fake.invocation.status = "succeeded"
        fake.invocation.output_refs = ["art_mask_001"]
        fake.artifacts = [
            MaskArtifact(
                id="art_mask_001",
                uri="generated/segment/task_001_001_mask.png",
                payload={"image_ref": "art_img_input_001", "grounding_ref": "art_geometry_001", "target": "primary_edit_region", "positive_points": [], "negative_points": [], "mask_score": 0.9},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(SegmentTool, "execute", side_effect=fake_segment_execute)

    ExecuteAgent()._run_tool_step(
        state=state,
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.SEGMENT,
        runtime_ctx={
            "base_image_ref": "art_img_input_001",
            "grounding_ref": "art_geometry_001",
        },
    )

    assert captured["args"].image_ref == "art_img_input_001"
    assert captured["args"].prompt


def test_execute_agent_exhausted_budget_goes_directly_to_replan() -> None:
    state = {
        "input": {
            "instruction_text": "先处理局部，再编辑",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="只修改上衣",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_fail",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 2,
    }

    result = ExecuteAgent().run(state)

    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].latest_execute_checkpoint == "failed"
    assert result["decision"].route == DecisionRoute.REPLAN
    assert result["decision"].replan is not None


def test_execute_agent_caps_oversized_edit_inputs_instead_of_retrying() -> None:
    state = {
        "input": {
            "instruction_text": "把人物、服饰和背景组合起来",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物、服饰和背景组合起来",
                input_artifact_ids=[
                    "art_img_input_001",
                    "art_img_input_002",
                    "art_img_input_003",
                    "art_img_input_004",
                ],
            )
        },
        "session": SessionState(
            session_id="sess_execute_budget_retry",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=[
                        "art_img_input_001",
                        "art_img_input_002",
                        "art_img_input_003",
                        "art_img_input_004",
                    ],
                )
            },
            artifact_index=ArtifactIndex(
                by_type={
                    ArtifactKind.IMAGE: [
                        "art_img_input_001",
                        "art_img_input_002",
                        "art_img_input_003",
                        "art_img_input_004",
                    ]
                }
            ),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(id="art_img_input_001", uri="store://images/1.png", payload={"role": "input"}, scope="session"),
            "art_img_input_002": ImageArtifact(id="art_img_input_002", uri="store://images/2.png", payload={"role": "input"}, scope="session"),
            "art_img_input_003": ImageArtifact(id="art_img_input_003", uri="store://images/3.png", payload={"role": "input"}, scope="session"),
            "art_img_input_004": ImageArtifact(id="art_img_input_004", uri="store://images/4.png", payload={"role": "input"}, scope="session"),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
    }

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert result["session"].current_task_id == "task_001"
    assert task_state.status == TaskStatus.WAITING_EVALUATION
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.edit_input_budget_overflow_count == 0
    latest_edit = latest_message_tool_result(result, tool_name=ToolName.EDIT)
    assert latest_edit is not None
    assert latest_edit.status == "succeeded"
    assert latest_edit.args["image_refs"] == [
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]
    assert len(latest_edit.args["image_refs"]) == 3


def test_execute_agent_caps_retry_inputs_and_resets_previous_budget_overflow() -> None:
    state = {
        "input": {
            "instruction_text": "把人物、服饰和背景组合起来",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物、服饰和背景组合起来",
                input_artifact_ids=[],
            )
        },
        "session": SessionState(
            session_id="sess_execute_budget_replan",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=[
                        "art_img_input_001",
                        "art_img_input_002",
                        "art_img_input_003",
                        "art_img_input_004",
                    ],
                    edit_input_budget_overflow_count=1,
                )
            },
            artifact_index=ArtifactIndex(
                by_type={
                    ArtifactKind.IMAGE: [
                        "art_img_input_001",
                        "art_img_input_002",
                        "art_img_input_003",
                        "art_img_input_004",
                    ]
                }
            ),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(id="art_img_input_001", uri="store://images/1.png", payload={"role": "input"}, scope="session"),
            "art_img_input_002": ImageArtifact(id="art_img_input_002", uri="store://images/2.png", payload={"role": "input"}, scope="session"),
            "art_img_input_003": ImageArtifact(id="art_img_input_003", uri="store://images/3.png", payload={"role": "input"}, scope="session"),
            "art_img_input_004": ImageArtifact(id="art_img_input_004", uri="store://images/4.png", payload={"role": "input"}, scope="session"),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
    }

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert result["session"].current_task_id == "task_001"
    assert task_state.status == TaskStatus.WAITING_EVALUATION
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.edit_input_budget_overflow_count == 0
    latest_edit = latest_message_tool_result(result, tool_name=ToolName.EDIT)
    assert latest_edit is not None
    assert latest_edit.status == "succeeded"
    assert latest_edit.args["image_refs"] == [
        "art_img_input_001",
        "art_img_input_002",
        "art_img_input_003",
    ]


def test_execute_agent_successful_edit_resets_budget_overflow_counter(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "保持图片内容不变",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="保持图片内容不变",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_budget_reset",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=["art_img_input_001"],
                    edit_input_budget_overflow_count=1,
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
    }

    fake_execution = mocker.Mock()
    fake_execution.invocation = ToolInvocationRecord(
        id="op_edit_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "保持图片内容不变", "image_refs": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_image_candidate_001"],
    )
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            created_by=ToolName.EDIT.value,
            scope="task",
        )
    ]
    mocker.patch("tools.edit_tool.EditTool.execute", return_value=fake_execution)

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.edit_input_budget_overflow_count == 0
    assert result["session"].phase == SessionPhase.EVALUATING
    assert task_state.latest_artifact_ids == ["art_image_candidate_001"]


def test_execute_agent_prompt_reconstruct_success_is_not_evaluable_candidate(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "先改写提示，再编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_prompt_not_candidate",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="rewrite prompt first",
            selected_tools=["prompt_reconstruct"],
        ),
    )
    fake_execution = mocker.Mock()
    fake_execution.invocation = ToolInvocationRecord(
        id="op_prompt_reconstruct_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.PROMPT_RECONSTRUCT,
        args={"input_artifact_ids": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_instruction_001"],
    )
    fake_execution.artifacts = [
        InstructionArtifact(
            id="art_instruction_001",
            payload={"instruction_text": "更清晰的编辑提示"},
            created_by=ToolName.PROMPT_RECONSTRUCT.value,
            scope="task",
        )
    ]
    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", return_value=fake_execution)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="prompt is ready",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert task_state.latest_execute_checkpoint == "failed"
    assert task_state.latest_artifact_ids == []
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["decision"].route == DecisionRoute.REPLAN
    assert "art_instruction_001" in task_state.task_artifact_ids


def test_execute_agent_prompt_reconstruct_then_edit_uses_only_image_candidate(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "先改写提示，再编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_prompt_then_edit",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 2,
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    strategies = [
        ExecuteLLMOutput(reasoning="rewrite prompt first", selected_tools=["prompt_reconstruct"]),
        ExecuteLLMOutput(reasoning="now edit", selected_tools=["edit"]),
    ]
    mocker.patch("agents.execute_agent.invoke_structured_llm", side_effect=strategies)
    executions = []
    prompt_execution = mocker.Mock()
    prompt_execution.invocation = ToolInvocationRecord(
        id="op_prompt_reconstruct_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.PROMPT_RECONSTRUCT,
        args={"input_artifact_ids": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_instruction_001"],
    )
    prompt_execution.artifacts = [
        InstructionArtifact(
            id="art_instruction_001",
            payload={"instruction_text": "更清晰的编辑提示"},
            created_by=ToolName.PROMPT_RECONSTRUCT.value,
            scope="task",
        )
    ]
    edit_execution = mocker.Mock()
    edit_execution.invocation = ToolInvocationRecord(
        id="op_edit_002",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "更清晰的编辑提示", "image_refs": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_image_candidate_001"],
    )
    edit_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            created_by=ToolName.EDIT.value,
            scope="task",
        )
    ]
    executions.extend([prompt_execution, edit_execution])
    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", side_effect=executions)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="step done",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.latest_artifact_ids == ["art_image_candidate_001"]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert [block.tool_name for block in message_tool_results(result, task_id="task_001")[-2:]] == [
        ToolName.PROMPT_RECONSTRUCT.value,
        ToolName.EDIT.value,
    ]


def test_execute_agent_edit_uses_task_local_instruction_instead_of_root(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    source_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
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
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(source_path),
                payload={"role": "input"},
                scope="session",
            ),
        },
        task_artifact_ids=[
            "art_instruction_task_001_001",
            "art_instruction_session_root_001",
        ],
    )
    state["session"].task_states["task_001"].resolved_input_artifact_ids = ["art_img_input_001"]

    captured: dict[str, object] = {}

    class FakeEditTool(BaseTool):
        name = ToolName.EDIT
        args_schema = EditArgs

        def execute(self, state, *, task_id: str, loop_index: int, args: EditArgs):
            captured["instruction"] = args.instruction
            captured["image_refs"] = list(args.image_refs)
            artifact = ImageArtifact(
                id="art_image_candidate_001",
                uri=str(source_path),
                payload={"role": "candidate_image"},
                created_by=ToolName.EDIT.value,
                scope="task",
            )
            invocation = ToolInvocationRecord(
                id="op_edit_001",
                task_id=task_id,
                loop_index=loop_index,
                tool_name=ToolName.EDIT,
                args=args.model_dump(),
                status="succeeded",
                output_refs=[artifact.id],
            )
            return ToolExecutionResult(invocation=invocation, artifacts=[artifact])

    registry = build_default_tool_registry()
    registry._tools[ToolName.EDIT] = FakeEditTool()

    ExecuteAgent(registry=registry)._select_and_run_tool(
        state=state,
        task=state["tasks"]["task_001"],
        task_id="task_001",
        loop_index=1,
        selected_tool=ToolName.EDIT,
        resolved_inputs=["art_img_input_001"],
        base_image_ref="art_img_input_001",
    )

    assert captured["instruction"] == "使用任务局部指令"
    assert captured["image_refs"] == ["art_img_input_001"]


def test_execute_agent_prompt_reconstruct_context_uses_shared_active_instruction() -> None:
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
    state["session"].task_states["task_001"].resolved_input_artifact_ids = ["art_img_input_001"]

    context = ExecuteAgent()._build_prompt_reconstruct_context(state, "task_001")

    assert "Current active instruction: 使用任务局部指令" in context


def test_execute_agent_tool_failure_still_fails_after_explicit_failure_limit(mocker) -> None:
    image_path = Path("examples/fig1.jpg")
    state = {
        "input": {
            "instruction_text": "保持图片内容不变",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="保持图片内容不变",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_tool_failure",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(image_path),
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
        "max_tool_failures": 1,
    }

    mocker.patch("tools.edit_tool.EditTool.execute", side_effect=RuntimeError("backend unavailable"))

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["session"].phase == SessionPhase.FAILED
    assert result["session"].current_task_id is None
    assert task_state.latest_execute_checkpoint == "failed"
    assert task_state.latest_execution_outcome == ExecutionOutcome.FAILURE
    latest_edit = latest_message_tool_result(result, tool_name=ToolName.EDIT)
    assert latest_edit is not None
    latest_observation = message_observations(result, task_id="task_001")[-1]
    assert latest_edit.args == {
        "instruction": "保持图片内容不变",
        "image_refs": ["art_img_input_001"],
    }
    assert "Tool failure 1/1" in latest_observation.text
    assert "backend unavailable" in latest_observation.text
    assert latest_edit.status == "failed"
    assert latest_edit.error == {
        "type": "RuntimeError",
        "message": "backend unavailable",
    }
    assert result["decision"].route == DecisionRoute.FAIL
    assert "execute_tool_failed" in result["decision"].issues
    assert "backend unavailable" in result["decision"].summary


def test_execute_agent_retries_after_tool_failure_with_error_context(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "保持图片内容不变",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="保持图片内容不变",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_tool_failure_retry",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 2,
        "max_tool_failures": 2,
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    captured_prompts: list[str] = []

    def fake_strategy(*, user_prompt, system_prompt, output_schema):
        captured_prompts.append(user_prompt)
        return ExecuteLLMOutput(reasoning="try edit", selected_tools=["edit"])

    mocker.patch("agents.execute_agent.invoke_structured_llm", side_effect=fake_strategy)

    failed_execution = mocker.Mock()
    failed_execution.invocation = ToolInvocationRecord(
        id="op_edit_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "保持图片内容不变", "image_refs": ["art_img_input_001"]},
        status="failed",
        output_refs=[],
        error={"type": "RuntimeError", "message": "backend unavailable"},
    )
    failed_execution.artifacts = []

    success_execution = mocker.Mock()
    success_execution.invocation = ToolInvocationRecord(
        id="op_edit_002",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "保持图片内容不变", "image_refs": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_image_candidate_001"],
    )
    success_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            created_by=ToolName.EDIT.value,
            scope="task",
        )
    ]
    mocker.patch.object(
        ExecuteAgent,
        "_select_and_run_tool",
        side_effect=[failed_execution, success_execution],
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="candidate is ready",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert [block.status for block in message_tool_results(result, task_id="task_001")] == [
        "failed",
        "succeeded",
    ]
    assert len(captured_prompts) == 2
    assert "Last tool failure:" in captured_prompts[1]
    assert "tool=edit" in captured_prompts[1]
    assert "backend unavailable" in captured_prompts[1]
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.latest_artifact_ids == ["art_image_candidate_001"]
    assert task_state.retry_context_text is None
    assert result["session"].phase == SessionPhase.EVALUATING


def test_execute_agent_llm_prompt_includes_retry_context(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "继续修图",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_retry_prompt",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=["art_img_input_001", "art_image_candidate_001"],
                    retry_context_text="Evaluator feedback:\n- Reason: 边界不干净\n- Fix focuses:\n  1. 修边界",
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://images/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    captured = {}

    def fake_structured(*, user_prompt, system_prompt, output_schema):
        captured["user_prompt"] = user_prompt
        return ExecuteLLMOutput(
            reasoning="retry with edit",
            selected_tools=["edit"],
        )

    mocker.patch("agents.execute_agent.invoke_structured_llm", side_effect=fake_structured)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    assert "Retry context:" in captured["user_prompt"]
    assert "边界不干净" in captured["user_prompt"]
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_image_candidate_001",
    ]


def test_execute_agent_observe_injects_artifact_summary(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                input_artifact_ids=["art_img_input_001", "art_img_input_002"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_summary",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/person.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="store://images/background.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.tool_name = ToolName.EDIT
    fake_execution.invocation.args = {
        "instruction": "把人物放到背景里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.invocation.output_refs = ["art_image_candidate_001"]
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", return_value=fake_execution)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="This candidate is ready for evaluator checkpoint.",
            artifact_summaries=[
                ObserveArtifactSummary(
                    artifact_id="art_image_candidate_001",
                    summary="人物已放入背景的候选图，可交给 evaluator 评估。",
                )
            ],
        ),
    )

    result = ExecuteAgent().run(state)

    artifact = result["artifacts"]["art_image_candidate_001"]
    assert artifact.summary == "人物已放入背景的候选图，可交给 evaluator 评估。"
    assert message_observations(result, task_id="task_001")[-1].text == "This candidate is ready for evaluator checkpoint."


def test_execute_agent_observe_uses_multimodal_when_image_artifact_exists(mocker, tmp_path) -> None:
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    state = {
        "input": {"instruction_text": "把人物放到背景里", "use_llm": True},
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                acceptance_criteria=["人物位于背景中"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_mm",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "artifacts": {},
    }
    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.args = {
        "instruction": "将人物主体放到背景图里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_001",
            uri=str(image_path),
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]
    mocker.patch(
        "agents.execute_agent.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )
    multimodal_call = mocker.patch(
        "agents.execute_agent.invoke_structured_multimodal_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="candidate looks good",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent()._observe_with_llm(
        state=state,
        task=state["tasks"]["task_001"],
        task_id="task_001",
        selected_tool=ToolName.EDIT,
        execution=fake_execution,
    )

    assert result.outcome == "success"
    multimodal_call.assert_called_once()


def test_execute_agent_observe_rejects_non_local_image_uri(mocker) -> None:
    state = {
        "input": {"instruction_text": "把人物放到背景里", "use_llm": True},
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                acceptance_criteria=["人物位于背景中"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_bad_uri",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "artifacts": {},
    }
    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.args = {
        "instruction": "将人物主体放到背景图里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]
    mocker.patch(
        "agents.execute_agent.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )

    with pytest.raises(ValueError, match="not a local file path"):
        ExecuteAgent()._observe_with_llm(
            state=state,
            task=state["tasks"]["task_001"],
            task_id="task_001",
            selected_tool=ToolName.EDIT,
            execution=fake_execution,
        )


def test_execute_agent_uses_llm_selected_base_image_artifact_id(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                input_artifact_ids=["art_img_input_001", "art_img_input_002"],
            )
        },
        "session": SessionState(
            session_id="sess_base_select",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002"],
                    task_artifact_ids=["art_under_001", "art_under_002"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/person.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="store://images/background.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_under_001": UnderstandingArtifact(
                id="art_under_001",
                payload={"image_ref": "art_img_input_001", "summary": "人物主体图"},
                scope="task",
            ),
            "art_under_002": UnderstandingArtifact(
                id="art_under_002",
                payload={"image_ref": "art_img_input_002", "summary": "拍照背景图"},
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="use the background as base and the person as reference",
            selected_tools=["edit"],
            base_image_artifact_id="art_img_input_002",
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    captured = {}

    def fake_select_and_run_tool(self, *, state, task, task_id, loop_index, selected_tool, resolved_inputs, base_image_ref):
        captured["base_image_ref"] = base_image_ref
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.EDIT
        fake.invocation.args = {
            "instruction": "把人物放到背景里",
            "image_refs": ["art_img_input_002", "art_img_input_001"],
        }
        fake.invocation.output_refs = ["art_image_candidate_001"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", fake_select_and_run_tool)

    ExecuteAgent().run(state)

    assert captured["base_image_ref"] == "art_img_input_002"


def test_execute_agent_retry_candidate_has_priority_as_base_image(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "继续修正失败结果",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正当前结果",
            )
        },
        "session": SessionState(
            session_id="sess_retry_base",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_image_candidate_001"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_001",
            route=DecisionRoute.CONTINUE_EXECUTE,
            task_id="task_001",
            summary="继续修正",
            task_retry=TaskRetryAdvice(
                reason="继续沿失败结果修正",
                base_candidate_artifact_id="art_image_candidate_001",
                fix_focuses=["修边界"],
            ),
        ),
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="even if another image exists, continue from failed candidate",
            selected_tools=["edit"],
            base_image_artifact_id="art_img_input_001",
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    captured = {}

    def fake_select_and_run_tool(self, *, state, task, task_id, loop_index, selected_tool, resolved_inputs, base_image_ref):
        captured["base_image_ref"] = base_image_ref
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.EDIT
        fake.invocation.args = {
            "instruction": "继续修正当前结果",
            "image_refs": ["art_image_candidate_001", "art_img_input_001"],
        }
        fake.invocation.output_refs = ["art_image_candidate_002"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_candidate_002",
                uri="store://generated/task_001/candidate_retry.png",
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", fake_select_and_run_tool)

    ExecuteAgent().run(state)

    assert captured["base_image_ref"] == "art_image_candidate_001"


def test_execute_agent_build_edit_image_refs_caps_inputs_to_budget() -> None:
    state = {
        "session": SessionState(
            session_id="sess_edit_refs",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_collage_001", "art_crop_001"],
                )
            },
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="examples/fig2.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_003": ImageArtifact(
                id="art_img_input_003",
                uri="examples/fig3.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_collage_001": ImageArtifact(
                id="art_collage_001",
                uri="examples/fig4.jpg",
                payload={"role": "collage_reference"},
                scope="task",
            ),
            "art_crop_001": ImageArtifact(
                id="art_crop_001",
                uri="examples/fig1.jpg",
                payload={"role": "cropped_preview", "source": "crop_preview"},
                created_by=ToolName.CROP.value,
                scope="task",
            ),
        },
    }

    refs = ExecuteAgent()._build_edit_image_refs(
        state=state,
        task_id="task_001",
        runtime_ctx={
            "base_image_ref": "art_img_input_001",
            "initial_base_image_ref": "art_img_input_001",
            "reference_refs": ["art_img_input_002", "art_img_input_003"],
            "grounding_ref": None,
            "mask_ref": None,
            "crop_ref": "art_crop_001",
        },
    )

    assert refs == [
        "art_img_input_001",
        "art_collage_001",
        "art_crop_001",
    ]


def test_execute_agent_repairs_repeated_crop_to_next_tool() -> None:
    state = {
        "session": SessionState(
            session_id="sess_repair_crop",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_mask_001", "art_crop_001"],
                )
            },
        ),
        "artifacts": {
            "art_mask_001": MaskArtifact(
                id="art_mask_001",
                uri="generated/segment/mask.png",
                payload={},
            ),
            "art_crop_001": ImageArtifact(
                id="art_crop_001",
                uri="examples/fig1.jpg",
                payload={"role": "cropped_preview", "source": "crop_preview"},
                created_by=ToolName.CROP.value,
                scope="task",
            ),
        },
    }

    assert ExecuteAgent()._repair_next_tool(state, "task_001", ToolName.CROP) == ToolName.UNDERSTAND

    state["artifacts"]["art_understanding_001"] = UnderstandingArtifact(
        id="art_understanding_001",
        payload={"image_ref": "art_crop_001", "summary": "crop summary"},
    )
    state["session"].task_states["task_001"].task_artifact_ids.append("art_understanding_001")

    assert ExecuteAgent()._repair_next_tool(state, "task_001", ToolName.CROP) == ToolName.EDIT


def test_execute_agent_finds_crop_preview_after_observe_relabels_role() -> None:
    state = {
        "session": SessionState(
            session_id="sess_relabelled_crop",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_crop_001"],
                )
            },
        ),
        "artifacts": {
            "art_crop_001": ImageArtifact(
                id="art_crop_001",
                uri="examples/fig1.jpg",
                payload={
                    "role": "cropped_preview",
                    "source": "crop_preview",
                    "crop_mode": "mask_cutout",
                },
                created_by=ToolName.CROP.value,
                role="person_identity_crop",
                scope="task",
            ),
        },
    }

    assert ExecuteAgent()._find_latest_crop_ref(state, "task_001") == "art_crop_001"
    assert ExecuteAgent()._repair_next_tool(state, "task_001", ToolName.CROP) == ToolName.UNDERSTAND
