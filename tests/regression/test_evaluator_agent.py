from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_evaluator_agent_uses_structured_evaluation_verdict() -> None:
    graph = build_runtime_graph()
    state = graph.invoke(
        {
            "input": {
                "session_id": "eval_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "change the shirt color",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.EVALUATING
    state["session"].current_task_id = "task_001"
    state["session"].final_result_id = None
    candidate_id = state["session"].task_states["task_001"].final_artifact_id or state["session"].final_result_id
    assert candidate_id is not None
    state["session"].task_states["task_001"].latest_artifact_ids = [candidate_id]
    state["session"].task_states["task_001"].latest_execute_checkpoint = "passed"
    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.PASS
    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].current_task_id is None


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


def test_evaluator_agent_invalid_candidate_replans_without_tool_call() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "pass",
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
            session_id="sess_eval_invalid_candidate",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_instruction_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    task_artifact_ids=["art_instruction_001"],
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
            "art_instruction_001": InstructionArtifact(
                id="art_instruction_001",
                payload={"instruction_text": "更清晰的编辑提示"},
                created_by=ToolName.PROMPT_RECONSTRUCT.value,
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    result = EvaluatorAgent().run(state)

    assert result["operations"] == []
    assert result["decision"].route == DecisionRoute.REPLAN
    assert "invalid_evaluation_candidate" in result["decision"].issues
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].status == TaskStatus.REPLANNED


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
                    evaluator_checkpoint_count=3,
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
