from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


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
