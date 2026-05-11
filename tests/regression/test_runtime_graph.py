from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


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


def test_register_and_understand_reads_runtime_budgets_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.graph import register_and_understand

    monkeypatch.setenv("AGENT_LOG_ENABLED", "false")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "false")
    monkeypatch.setenv("AGENT_MAX_EXECUTE_ACTS", "6")
    monkeypatch.setenv("AGENT_MAX_TOOL_FAILURES", "5")

    result = register_and_understand(
        {
            "input": {
                "session_id": "env-execute-acts",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "Keep the image unchanged.",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["max_execute_acts"] == 6
    assert result["max_tool_failures"] == 5


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


def test_route_after_execute_retries_same_task_on_budget_overflow() -> None:
    from runtime.graph import _route_after_execute

    state = {
        "session": SessionState(
            session_id="sess_route_retry",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    latest_execute_checkpoint="retry",
                )
            },
        )
    }

    assert _route_after_execute(state) == "execute"


def test_runtime_graph_retry_route_does_not_raise_key_error(mocker) -> None:
    from runtime import graph as runtime_graph

    execute_calls = {"count": 0}

    def fake_register_and_understand(_state):
        return {
            "input": {"session_id": "sess_graph_retry", "instruction_text": "retry", "use_llm": False},
            "session": SessionState(
                session_id="sess_graph_retry",
                phase=SessionPhase.PLANNING,
                current_plan_id=None,
                current_task_id=None,
                task_states={},
                artifact_index=ArtifactIndex(by_type={}),
            ),
            "artifacts": {},
            "plans": {},
            "tasks": {},
            "task_act_records": [],
            "task_loops": [],
            "operations": [],
        }

    def fake_plan(state):
        session = state["session"]
        if "task_001" not in session.task_states:
            state["tasks"]["task_001"] = Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="retry",
            )
            session.task_states["task_001"] = TaskState(
                task_id="task_001",
                status=TaskStatus.RUNNING,
            )
        session.current_plan_id = "plan_001"
        session.current_task_id = "task_001"
        session.phase = SessionPhase.EXECUTING
        state["plans"]["plan_001"] = Plan(id="plan_001", instruction="retry", task_ids=["task_001"])
        return state

    def fake_execute(state):
        execute_calls["count"] += 1
        task_state = state["session"].task_states["task_001"]
        state["session"].current_task_id = "task_001"
        if execute_calls["count"] == 1:
            task_state.latest_execute_checkpoint = "retry"
            task_state.status = TaskStatus.RUNNING
            state["session"].phase = SessionPhase.EXECUTING
        else:
            task_state.latest_execute_checkpoint = "passed"
            task_state.latest_artifact_ids = ["art_image_candidate_001"]
            task_state.status = TaskStatus.WAITING_EVALUATION
            state["session"].phase = SessionPhase.EVALUATING
            state["artifacts"]["art_image_candidate_001"] = ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/candidate.png",
                payload={"role": "candidate_image"},
                created_by=ToolName.EDIT.value,
                scope="task",
            )
        return state

    def fake_evaluate(state):
        task_state = state["session"].task_states["task_001"]
        task_state.latest_evaluate_checkpoint = "passed"
        task_state.status = TaskStatus.PASSED
        task_state.final_artifact_id = "art_image_candidate_001"
        state["session"].phase = SessionPhase.DONE
        state["session"].current_task_id = None
        return state

    mocker.patch.object(runtime_graph, "register_and_understand", side_effect=fake_register_and_understand)
    mocker.patch.object(runtime_graph, "plan", side_effect=fake_plan)
    mocker.patch.object(runtime_graph, "execute_current_task", side_effect=fake_execute)
    mocker.patch.object(runtime_graph, "evaluate_checkpoint", side_effect=fake_evaluate)
    mocker.patch.object(runtime_graph, "log_event")

    result = runtime_graph.build_runtime_graph().invoke({"input": {"session_id": "sess_graph_retry"}})

    assert execute_calls["count"] == 2
    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].task_states["task_001"].status == TaskStatus.PASSED
