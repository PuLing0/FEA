from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel, ValidationError

from agent import agent, create_agent
from agents import EvaluatorAgent, ExecuteAgent, PlanAgent
from llm import invoke_llm, invoke_structured_llm, load_llm_config
from runtime import build_runtime_graph
from runtime.input_selector import (
    TaskInputSelectionOutput,
    build_candidate_image_pool,
    build_candidate_images_text,
)
from runtime.scheduler import select_next_runnable_task
from schema import (
    ArtifactIndex,
    ArtifactKind,
    ArtifactStage,
    CollageArgs,
    CropArgs,
    Decision,
    DecisionLLMOutput,
    DecisionRoute,
    EditArgs,
    EvaluationArtifact,
    ExecuteLLMOutput,
    ExecutionOutcome,
    GeometryArtifact,
    GlobalEditArgs,
    GroundingArgs,
    ImageArtifact,
    InstructionArtifact,
    LocalEditArgs,
    MaskArtifact,
    ObserveArtifactSummary,
    ObserveLLMOutput,
    Plan,
    PlanLLMOutput,
    PlanTaskSpec,
    ReplanMode,
    ReplanRequest,
    SegmentArgs,
    SessionPhase,
    SessionState,
    Task,
    TaskActRecord,
    TaskLoop,
    TaskRetryAdvice,
    TaskState,
    TaskStatus,
    ToolInvocationRecord,
    ToolName,
    UnderstandArgs,
    UnderstandingArtifact,
)
from tools import build_default_tool_registry


def test_public_imports_construct_minimal_objects() -> None:
    image = ImageArtifact(
        id="art_img_001",
        uri="store://images/input.png",
        payload={"role": "input"},
        summary="原始输入图",
        stage=ArtifactStage.COMMITTED,
    )
    mask = MaskArtifact(
        id="art_mask_001",
        uri="store://masks/object.png",
        payload={"target": "shirt"},
    )
    geometry = GeometryArtifact(
        id="art_geo_001",
        payload={"image_artifact_id": "art_img_001", "candidates": [{"bbox": [1, 2, 3, 4], "score": 0.9}]},
    )
    instruction = InstructionArtifact(
        id="art_inst_001",
        payload={"raw_instruction": "replace the shirt"},
        created_by="user",
    )
    understanding = UnderstandingArtifact(
        id="art_under_001",
        payload={"image_ref": "art_img_001", "summary": "person in shirt"},
        source_ids=["art_img_001"],
    )
    evaluation = EvaluationArtifact(
        id="art_eval_001",
        payload={"verdict": "good", "issues": []},
        source_ids=["art_img_001"],
    )

    assert image.kind == ArtifactKind.IMAGE
    assert mask.kind == ArtifactKind.MASK
    assert geometry.kind == ArtifactKind.GEOMETRY
    assert instruction.kind == ArtifactKind.INSTRUCTION
    assert understanding.kind == ArtifactKind.UNDERSTANDING
    assert evaluation.kind == ArtifactKind.EVALUATION
    assert image.summary == "原始输入图"


def test_instruction_artifact_returns_instruction_text() -> None:
    artifact = InstructionArtifact(
        id="art_inst_001",
        payload={"instruction_text": "把人物放入背景中，并保持人物身份不变。"},
    )

    assert artifact.get_instruction_text() == "把人物放入背景中，并保持人物身份不变。"


def test_runtime_models_construct() -> None:
    plan = Plan(
        id="plan_001",
        instruction="edit the shirt color",
        task_ids=["task_001"],
        input_artifact_ids=["art_img_001"],
        understanding_artifact_ids=["art_under_001"],
    )
    task = Task(
        id="task_001",
        plan_id="plan_001",
        type="local_edit",
        instruction="segment shirt then edit it",
        input_artifact_ids=["art_img_001"],
        acceptance_criteria=["shirt color changed", "face unchanged"],
    )
    task_state = TaskState(
        task_id="task_001",
        status=TaskStatus.RUNNING,
        latest_execution_outcome=ExecutionOutcome.SUCCESS,
        latest_execute_checkpoint="passed",
        latest_evaluate_checkpoint="passed",
        loop_count=1,
    )
    session = SessionState(
        session_id="sess_001",
        phase=SessionPhase.EXECUTING,
        current_plan_id="plan_001",
        current_task_id="task_001",
        task_states={"task_001": task_state},
        artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_001"]}),
    )

    assert plan.task_ids == ["task_001"]
    assert task.id == "task_001"
    assert session.task_states["task_001"].status == TaskStatus.RUNNING


def test_task_loop_construct() -> None:
    loop = TaskLoop(
        id="loop_task_001_001",
        task_id="task_001",
        loop_index=1,
        thinking="Need to segment before local edit.",
        selected_tools=["segment", "crop", "understand", "edit"],
        output_artifact_ids=["art_image_001"],
        observation="Produced one candidate image.",
    )
    assert loop.loop_index == 1
    assert loop.selected_tools == ["segment", "crop", "understand", "edit"]


def test_task_act_record_construct() -> None:
    record = TaskActRecord(
        task_id="task_001",
        loop_index=1,
        act_index=1,
        thinking_text="Need to rewrite prompt before editing.",
        tool_name="prompt_reconstruct",
        tool_args={"input_artifact_ids": ["art_img_input_001"]},
        output_artifact_ids=["art_instruction_001"],
        observation_text="Produced one rewritten instruction artifact.",
    )
    assert record.tool_name == "prompt_reconstruct"


def test_observe_llm_output_construct() -> None:
    out = ObserveLLMOutput(
        outcome="continue",
        observation="This crop preview is useful but not final yet.",
        artifact_summaries=[
            ObserveArtifactSummary(
                artifact_id="art_image_001",
                summary="局部裁剪预览图，用于下一步判断编辑区域。",
            )
        ],
    )
    assert out.outcome == "continue"
    assert out.artifact_summaries[0].artifact_id == "art_image_001"


def test_tool_registry_contains_prompt_reconstruct() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.PROMPT_RECONSTRUCT) is not None


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


def test_select_next_runnable_task_respects_dependencies() -> None:
    state = {
        "session": SessionState(
            session_id="sched_001",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id=None,
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.PENDING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="dag test",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="compose_subject",
                instruction="first",
            ),
            "task_002": Task(
                id="task_002",
                plan_id="plan_001",
                type="place_subject_in_background",
                instruction="second",
                depends_on=["task_001"],
            ),
        },
    }
    assert select_next_runnable_task(state) == "task_002"


def test_tool_args_construct() -> None:
    understand = UnderstandArgs(
        image_ref="art_img_001",
    )
    grounding = GroundingArgs(
        image_ref="art_img_001",
        target_description="shirt area",
        top_k=2,
    )
    segment = SegmentArgs(
        image_ref="art_img_001",
        target="shirt",
    )
    crop = CropArgs(
        image_ref="art_img_001",
        mask_ref="art_mask_001",
    )
    collage = CollageArgs(
        block_artifact_ids=["art_img_001", "art_img_002"],
        layout_goal="identity and clothing are primary",
    )
    global_edit = GlobalEditArgs(
        image_ref="art_img_001",
    )

    assert understand.image_ref == "art_img_001"
    assert grounding.target_description == "shirt area"
    assert segment.target == "shirt"
    assert crop.mask_ref == "art_mask_001"
    assert collage.block_artifact_ids == ["art_img_001", "art_img_002"]
    assert global_edit.mode == "global_edit"


def test_tool_registry_contains_grounding_and_collage() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.GROUNDING) is not None
    assert registry.get(ToolName.COLLAGE) is not None


class EditArgsHolder(BaseModel):
    value: EditArgs


def test_edit_args_discriminated_union_accepts_valid_local() -> None:
    payload = {
        "value": {
            "mode": "local_edit",
            "image_ref": "art_img_001",
            "mask_ref": "art_mask_001",
        }
    }
    parsed = EditArgsHolder.model_validate(payload)
    assert isinstance(parsed.value, LocalEditArgs)


def test_edit_args_discriminated_union_rejects_invalid_local() -> None:
    payload = {
        "value": {
            "mode": "local_edit",
            "image_ref": "art_img_001",
        }
    }
    with pytest.raises(ValidationError):
        EditArgsHolder.model_validate(payload)


def test_decision_continue_execute_requires_task_retry() -> None:
    decision = Decision(
        id="dec_001",
        route=DecisionRoute.CONTINUE_EXECUTE,
        task_id="task_001",
        summary="Need one more pass",
        source_execution_outcome=ExecutionOutcome.FAILURE,
        task_retry=TaskRetryAdvice(
            reason="mask too loose",
            base_candidate_artifact_id="art_img_001",
            fix_focuses=["tighten mask"],
        ),
    )
    assert decision.task_retry is not None


def test_decision_replan_requires_replan_request() -> None:
    decision = Decision(
        id="dec_002",
        route=DecisionRoute.REPLAN,
        plan_id="plan_001",
        summary="Need to split the task",
        replan=ReplanRequest(
            mode=ReplanMode.SPLIT_TASK,
            reason="single task is too coarse",
        ),
    )
    assert decision.replan is not None


def test_decision_pass_forbids_retry_payloads() -> None:
    with pytest.raises(ValidationError):
        Decision(
            id="dec_003",
            route=DecisionRoute.PASS,
            summary="Looks good",
            task_retry=TaskRetryAdvice(
                reason="should not exist",
                fix_focuses=["none"],
            ),
        )


def test_tool_invocation_record_links_outputs() -> None:
    record = ToolInvocationRecord(
        id="op_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"mode": "local_edit", "image_ref": "art_img_001"},
        status="succeeded",
        output_refs=["art_img_002"],
        raw_output_uri="runs/sess_001/task_001/loop_01/edit/result.json",
    )
    assert record.output_refs == ["art_img_002"]


def test_minimal_runtime_graph_pass_path() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_001",
                "image_uri": "store://images/input.png",
                "image_uris": ["store://images/input.png"],
                "instruction_text": "change the shirt color",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_plan_id == "plan_001"
    assert result["session"].current_task_id == "task_001"
    assert len(result["operations"]) == 1
    assert [op.tool_name for op in result["operations"]] == [ToolName.UNDERSTAND]
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]
    assert result["tasks"]["task_001"].type == "compose_subject"
    assert result["tasks"]["task_002"].type == "place_subject_in_background"
    understand_ops = [op for op in result["operations"] if op.tool_name == ToolName.UNDERSTAND]
    assert len(understand_ops) == 1


def test_minimal_runtime_graph_replan_path() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_002",
                "image_uri": "store://images/input.png",
                "image_uris": ["store://images/input.png"],
                "instruction_text": "change the shirt color",
                "desired_decision_route": "replan",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_001"
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]


def test_four_images_generate_task_todo_array() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_003",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "Generate a photo of this person wearing the provided top and skirt in the provided background.",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert len(result["session"].artifact_index.by_type[ArtifactKind.IMAGE]) == 4
    understand_ops = [op for op in result["operations"] if op.tool_name == ToolName.UNDERSTAND]
    assert len(understand_ops) == 4
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]
    assert result["tasks"]["task_001"].instruction.startswith("Combine the face")
    assert result["tasks"]["task_002"].instruction.startswith("Place the composed subject")
    assert result["tasks"]["task_001"].depends_on == []
    assert result["tasks"]["task_002"].depends_on == ["task_001"]
    assert not any(artifact_id.startswith("art_image_") for artifact_id in result["tasks"]["task_002"].input_artifact_ids)
    assert result["session"].current_task_id == "task_001"
    assert result["session"].task_states["task_001"].status == TaskStatus.RUNNING
    assert result["session"].task_states["task_002"].status == TaskStatus.PENDING
    assert set(result["session"].artifact_index.by_type.keys()) == {
        ArtifactKind.IMAGE,
    }


def test_single_task_loop_runs_until_success() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "single-task-loop",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.DONE
    assert result["decision"].route == DecisionRoute.PASS
    assert result["session"].final_result_id is not None
    assert len(result["task_loops"]) >= 2
    assert all(loop.selected_tools for loop in result["task_loops"])


def test_all_tasks_run_until_session_done() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "all-task-loop",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].status == TaskStatus.PASSED
    assert result["session"].task_states["task_002"].status == TaskStatus.PASSED
    loop_task_ids = [loop.task_id for loop in result["task_loops"]]
    assert "task_001" in loop_task_ids
    assert "task_002" in loop_task_ids


def test_task_one_result_is_resolved_into_task_two_runtime_inputs() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "inject-task-result",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    resolved_inputs = result["session"].task_states["task_002"].resolved_input_artifact_ids
    assert any(artifact_id.startswith("art_image_") for artifact_id in resolved_inputs)
    assert result["session"].task_states["task_002"].input_selection_reasoning is not None
    assert result["session"].task_states["task_002"].input_validation_summary is not None


def test_task_two_uses_reference_edit_after_task_switch() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "reference-edit-switch",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    task_two_edit_ops = [
        op for op in result["operations"]
        if op.task_id == "task_002" and op.tool_name == ToolName.EDIT
    ]
    assert task_two_edit_ops
    assert "reference_refs" in task_two_edit_ops[0].args


def test_agent_entrypoint_exports_compiled_graph() -> None:
    compiled = create_agent()
    assert type(compiled).__name__ == "CompiledStateGraph"
    assert type(agent).__name__ == "CompiledStateGraph"


def test_load_llm_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    config = load_llm_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.invalid/v1"
    assert config.model_name == "test-model"
    assert config.temperature == 0.2


def test_invoke_llm_uses_langchain_model(mocker) -> None:
    fake_response = mocker.Mock()
    fake_response.__class__.__name__ = "AIMessage"

    fake_model = mocker.Mock()
    fake_model.invoke.return_value = fake_response

    result = invoke_llm(
        user_prompt="hello",
        system_prompt="you are helpful",
        model=fake_model,
    )

    assert result is fake_response
    fake_model.invoke.assert_called_once()


def test_invoke_structured_llm_uses_structured_output(mocker) -> None:
    class OutputSchema(BaseModel):
        answer: str

    parsed = OutputSchema(answer="done")
    fake_result = mocker.Mock()
    fake_result.output = parsed
    fake_agent = mocker.Mock()
    fake_agent.run_sync.return_value = fake_result
    agent_cls = mocker.patch("llm.client.Agent", return_value=fake_agent)

    result = invoke_structured_llm(
        user_prompt="return structured",
        system_prompt="system",
        output_schema=OutputSchema,
        model="openai:gpt-5.4",
    )

    assert result == parsed
    agent_cls.assert_called_once()
    fake_agent.run_sync.assert_called_once_with("return structured")


def test_plan_agent_uses_llm_when_enabled(mocker) -> None:
    state = build_runtime_graph().invoke(
        {
            "input": {
                "session_id": "plan_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "make the jacket red",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.PLANNING
    state["session"].current_plan_id = None
    state["session"].current_task_id = None
    state["plans"] = {}
    state["tasks"] = {}
    state["session"].task_states = {}
    state["input"]["use_llm"] = True

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction="llm plan",
            tasks=[
                PlanTaskSpec(
                    id="task_002",
                    type="global_edit",
                    instruction="run one global edit",
                    input_artifact_ids=["art_img_input_001"],
                    depends_on=[],
                    acceptance_criteria=["candidate exists"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["plans"]["plan_001"].instruction == "llm plan"
    assert result["tasks"]["task_002"].type == "global_edit"
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.INSTRUCTION
        for artifact_id in result["session"].task_states["task_002"].task_artifact_ids
    )


def test_execute_agent_uses_llm_strategy_when_enabled(mocker) -> None:
    graph = build_runtime_graph()
    state = graph.invoke(
        {
            "input": {
                "session_id": "exec_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "global recolor",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.EXECUTING
    state["session"].current_task_id = "task_001"
    state["session"].task_states["task_001"].status = TaskStatus.RUNNING
    state["session"].task_states["task_001"].latest_artifact_ids = []
    state["session"].task_states["task_001"].loop_count = 0
    state["session"].task_states["task_001"].resolved_input_artifact_ids = []
    state["operations"] = state["operations"][:1]
    state["task_act_records"] = []
    state["input"]["use_llm"] = True

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="global edit is enough",
            selected_tools=["edit"],
            edit_mode="global_edit",
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

    result = ExecuteAgent().run(state)

    assert result["operations"][-1].tool_name == ToolName.EDIT
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
    ]
    assert result["session"].task_states["task_001"].input_selection_reasoning is not None
    latest_refs = result["session"].task_states["task_001"].latest_artifact_ids
    assert len(latest_refs) == 1
    assert latest_refs[0].startswith("art_image_")
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.INSTRUCTION
        for artifact_id in result["session"].task_states["task_001"].task_artifact_ids
    )
    assert result["task_act_records"]


def test_grounding_tool_returns_geometry_artifact() -> None:
    from tools.grounding_tool import GroundingTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="定位人物区域",
            )
        },
        "session": SessionState(
            session_id="sess_grounding",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri="store://images/input.png",
                payload={"role": "input"},
            )
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = GroundingTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=GroundingArgs(
            image_ref="art_img_001",
            target_description="person torso",
            top_k=2,
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.GEOMETRY
    assert len(artifact.payload["candidates"]) == 2
    assert execution.invocation.result_payload["geometry_artifact_id"] == artifact.id


def test_collage_tool_returns_image_artifact() -> None:
    from tools.collage_tool import CollageTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="整理参考图",
            )
        },
        "session": SessionState(
            session_id="sess_collage",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {},
        "operations": [],
        "task_act_records": [],
    }

    execution = CollageTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="identity and clothing are primary",
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert artifact.payload["role"] == "collage_reference"
    assert len(artifact.payload["layers"]) == 2
    assert execution.invocation.result_payload["collage_artifact_id"] == artifact.id


def test_execute_agent_accepts_grounding_and_collage_strategy(mocker) -> None:
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
                uri="store://images/1.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="store://images/2.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_003": ImageArtifact(
                id="art_img_input_003",
                uri="store://images/3.png",
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

    tool_names = [op.tool_name for op in result["operations"]]
    assert ToolName.GROUNDING in tool_names
    assert ToolName.COLLAGE in tool_names
    assert result["operations"][-1].tool_name == ToolName.EDIT
    assert invoke_structured.call_count == 3


def test_edit_tool_prefers_instruction_artifact_human_text() -> None:
    from tools.edit_tool import EditTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="原始任务指令",
            )
        },
        "session": SessionState(
            session_id="sess_instruction_override",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_inst_001"],
                )
            },
        ),
        "artifacts": {
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={
                    "instruction_text": "将 art_img_input_004 作为背景，将人物放入其中。 Relevant artifacts: art_img_input_004"
                },
            ),
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri="store://images/input.png",
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = EditTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=GlobalEditArgs(image_ref="art_img_001"),
    )

    assert "Relevant artifacts: art_img_input_004" in execution.artifacts[0].payload["task_instruction"]


def test_local_edit_flow_materializes_task_inputs_and_intermediate_task_artifacts() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    state = graph.invoke(
        {
            "input": {
                "session_id": "local-edit-chain",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "只修改上衣区域。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["tasks"]["task_001"].type = "local_edit"
    state["tasks"]["task_001"].input_artifact_ids = ["art_img_input_001"]
    state["session"].current_task_id = "task_001"
    state["session"].task_states["task_001"].status = TaskStatus.RUNNING

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.INSTRUCTION
        for artifact_id in task_state.task_artifact_ids
    )
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.MASK
        for artifact_id in task_state.task_artifact_ids
    )
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.UNDERSTANDING
        for artifact_id in task_state.task_artifact_ids
    )
    assert [loop.selected_tools for loop in result["task_loops"]] == [
        ["segment", "crop", "understand", "edit"],
    ]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert result["session"].task_states["task_001"].latest_execute_checkpoint == "passed"
    assert len(result["task_act_records"]) == 4


def test_evaluator_agent_uses_llm_decision_when_enabled(mocker) -> None:
    graph = build_runtime_graph()
    state = graph.invoke(
        {
            "input": {
                "session_id": "eval_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "change the shirt color",
                "desired_decision_route": "fail",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.EVALUATING
    state["session"].current_task_id = "task_001"
    state["session"].final_result_id = None
    state["session"].task_states["task_001"].latest_artifact_ids = ["art_image_candidate_001"]
    state["session"].task_states["task_001"].latest_execute_checkpoint = "passed"
    state["input"]["use_llm"] = True

    mocker.patch("agents.evaluator_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.evaluator_agent.invoke_structured_llm",
        return_value=DecisionLLMOutput(
            route=DecisionRoute.PASS,
            summary="llm says pass",
            issues=[],
        ),
    )

    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.PASS
    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_002"


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


def test_evaluator_agent_failed_candidate_can_continue_same_task() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "continue_execute",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                acceptance_criteria=["subject appears in background"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_continue",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_image_candidate_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    evaluator_checkpoint_count=0,
                    task_artifact_ids=["art_inst_001", "art_image_candidate_001"],
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
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={"instruction_text": "把人物放到背景里"},
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
            ),
        },
        "operations": [],
        "task_act_records": [],
        "max_evaluator_checkpoints": 3,
    }

    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.CONTINUE_EXECUTE
    assert result["decision"].task_retry is not None
    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_001"
    assert result["session"].task_states["task_001"].latest_evaluate_checkpoint == "failed"
    assert "art_img_input_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert "art_image_candidate_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert "art_inst_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert result["session"].task_states["task_001"].retry_context_text is not None
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == []


def test_evaluator_agent_budget_exhausted_upgrades_to_replan() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "continue_execute",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                acceptance_criteria=["subject appears in background"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_budget",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_image_candidate_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    evaluator_checkpoint_count=2,
                    task_artifact_ids=["art_image_candidate_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
            ),
        },
        "operations": [],
        "task_act_records": [],
        "max_evaluator_checkpoints": 3,
    }

    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.REPLAN
    assert result["decision"].replan is not None
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None


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
    fake_execution.invocation.args = {"mode": "reference_edit"}
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
    assert result["task_act_records"][-1].observation_text == "This candidate is ready for evaluator checkpoint."


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
        fake.invocation.args = {"mode": "reference_edit"}
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
        fake.invocation.args = {"mode": "reference_edit"}
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


def test_plan_agent_replan_creates_new_plan_version_and_marks_old_tasks() -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_replan",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            latest_decision_id="dec_001",
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
                "task_003": TaskState(task_id="task_003", status=TaskStatus.PENDING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002", "task_003"],
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
            "task_003": Task(id="task_003", plan_id="plan_001", type="future", instruction="未来 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                summary="原始输入人物图",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                summary="已保留的人物主体候选图",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_001",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002", "task_003"]
    assert result["session"].task_states["task_001"].status == TaskStatus.PASSED
    assert result["session"].task_states["task_002"].status == TaskStatus.REPLANNED
    assert result["session"].task_states["task_003"].status == TaskStatus.ABANDONED
    assert "task_001" in result["plans"]["plan_002"].task_ids
    assert any(task_id.startswith("task_00") and task_id not in {"task_001", "task_002", "task_003"} for task_id in result["plans"]["plan_002"].task_ids)
    assert "art_keep_001" in result["plans"]["plan_002"].input_artifact_ids
    new_task_ids = [
        task_id
        for task_id in result["plans"]["plan_002"].task_ids
        if task_id != "task_001"
    ]
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_selects_first_runnable_new_task() -> None:
    state = {
        "input": {
            "instruction_text": "重排路线",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_replan_runnable",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="done", instruction="旧已完成"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="旧失败"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "decision": Decision(
            id="dec_002",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="重排顺序",
            replan=ReplanRequest(
                mode=ReplanMode.REROUTE_PLAN,
                reason="后续路线需要重排",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=[],
            ),
        ),
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["session"].current_task_id is not None
    assert result["session"].task_states[result["session"].current_task_id].status == TaskStatus.RUNNING


def test_plan_agent_initial_planning_uses_incrementing_plan_id() -> None:
    state = {
        "input": {
            "instruction_text": "新建计划",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_initial_increment",
            phase=SessionPhase.PLANNING,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001"]),
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="old", instruction="旧 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "task_act_records": [],
        "task_loops": [],
        "operations": [],
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"


def test_plan_agent_invalid_llm_output_falls_back_to_replan_template(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_invalid_replan_llm",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_003",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction='[{"id":"task_999","type":"bad"}]',
            tasks=[
                PlanTaskSpec(
                    id="task_001",
                    type="bad",
                    instruction="bad",
                    input_artifact_ids=[],
                    depends_on=[],
                    acceptance_criteria=["bad"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["plans"]["plan_002"].instruction.startswith("Replan mode:")
    new_task_ids = [task_id for task_id in result["plans"]["plan_002"].task_ids if task_id != "task_001"]
    assert len(new_task_ids) == 2
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_rejects_illegal_llm_input_artifact_id(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_replan_illegal_input",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001", "task_002"])
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "decision": Decision(
            id="dec_illegal_input",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要重做",
            replan=ReplanRequest(
                mode=ReplanMode.REROUTE_PLAN,
                reason="输入绑定不应由 planner 决定",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=[],
            ),
        ),
    }

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction="补全后续任务",
            tasks=[
                PlanTaskSpec(
                    id="task_003",
                    type="bad_input_binding",
                    instruction="错误绑定不存在的输入",
                    input_artifact_ids=["art_future_999"],
                    depends_on=["task_001"],
                    acceptance_criteria=["完成"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["plans"]["plan_002"].instruction.startswith("Replan mode:")
    new_task_ids = [task_id for task_id in result["plans"]["plan_002"].task_ids if task_id != "task_001"]
    assert len(new_task_ids) == 2
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_prompt_includes_available_artifact_summaries(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_replan_artifact_catalog",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001", "task_002"])
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                summary="原始输入人物图",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                summary="已保留的人物主体候选图",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_004",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    captured = {}

    def fake_structured(*, user_prompt, system_prompt, output_schema):
        captured["user_prompt"] = user_prompt
        return PlanLLMOutput(
            plan_instruction="补全后续任务",
            tasks=[
                PlanTaskSpec(
                    id="task_003",
                    type="prepare_refined_inputs",
                    instruction="准备后续输入",
                    input_artifact_ids=[],
                    depends_on=["task_001"],
                    acceptance_criteria=["输入准备完成"],
                )
            ],
        )

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch("agents.plan_agent.invoke_structured_llm", side_effect=fake_structured)

    result = PlanAgent().run(state)

    assert "Available artifacts for planning:" in captured["user_prompt"]
    assert "art_keep_001" in captured["user_prompt"]
    assert "已保留的人物主体候选图" in captured["user_prompt"]
    assert result["plans"]["plan_002"].task_ids == ["task_001", "task_003"]
    assert result["tasks"]["task_003"].input_artifact_ids == []
