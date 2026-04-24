from __future__ import annotations

from schema import (
    ArtifactIndex,
    ArtifactKind,
    DecisionRoute,
    EvaluationArtifact,
    ExecutionOutcome,
    ImageArtifact,
    Plan,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
    ToolInvocationRecord,
    ToolName,
)
from agents import EvaluatorAgent
from tools.base import ToolExecutionResult


class FakeEvaluateTool:
    name = ToolName.EVALUATE

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def run(self, state, *, task_id: str, loop_index: int, args) -> ToolExecutionResult:
        artifact = EvaluationArtifact(
            id=f"art_eval_{self.verdict}",
            payload={
                "input_refs": list(args.input_refs),
                "candidate_ref": args.candidate_ref,
                "candidate_refs": list(args.candidate_refs),
                "instruction": args.instruction,
                "checks": list(args.checks),
                "is_satisfied": self.verdict == "pass",
                "verdict": self.verdict,
                "reason": f"fake {self.verdict} evaluation",
                "issues": ["revise visible mismatch"] if self.verdict == "needs_revision" else [],
                "new_rewritten_prompt": "refine the edit" if self.verdict == "needs_revision" else None,
                "scores": {
                    "instruction_success": 4,
                    "reference_consistency": 4,
                    "overediting": 4,
                    "naturalness": 4,
                    "artifacts": 4,
                    "semantic_score": 4,
                    "quality_score": 4,
                    "weighted_score": 4,
                    "overall_score": 4,
                },
            },
            source_ids=[*args.input_refs, args.candidate_ref],
            created_by=ToolName.EVALUATE.value,
            scope="task",
        )
        invocation = ToolInvocationRecord(
            id=f"op_eval_{self.verdict}",
            task_id=task_id,
            loop_index=loop_index,
            tool_name=ToolName.EVALUATE,
            args=args.model_dump(),
            status="succeeded",
            output_refs=[artifact.id],
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])


class FakeRegistry:
    def __init__(self, verdict: str) -> None:
        self.evaluate_tool = FakeEvaluateTool(verdict)

    def get(self, name: ToolName):
        assert name == ToolName.EVALUATE
        return self.evaluate_tool


def _make_evaluator_state() -> dict:
    return {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "pass",
            "use_llm": False,
        },
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="把人物放到背景里",
                task_ids=["task_001"],
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                input_artifact_ids=["art_img_input_001"],
                acceptance_criteria=["人物应自然出现在背景中"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_route",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    resolved_input_artifact_ids=["art_img_input_001"],
                    latest_artifact_ids=["art_image_candidate_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    loop_count=1,
                    task_artifact_ids=["art_image_candidate_001"],
                )
            },
            artifact_index=ArtifactIndex(
                by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}
            ),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "operations": [
            ToolInvocationRecord(
                id="op_edit_001",
                task_id="task_001",
                loop_index=1,
                tool_name=ToolName.EDIT,
                args={
                    "instruction": "把人物放到背景里，并保持自然融合",
                    "image_refs": ["art_img_input_001"],
                },
                status="succeeded",
                output_refs=["art_image_candidate_001"],
            )
        ],
        "task_act_records": [],
        "task_loops": [],
        "max_evaluator_checkpoints": 3,
    }


def test_evaluator_routes_pass_verdict_to_task_pass() -> None:
    state = _make_evaluator_state()

    result = EvaluatorAgent(registry=FakeRegistry("pass")).run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["decision"].route == DecisionRoute.PASS
    assert task_state.status == TaskStatus.PASSED
    assert task_state.latest_evaluate_checkpoint == "passed"
    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].final_result_id == "art_image_candidate_001"


def test_evaluator_routes_needs_revision_verdict_to_continue_execute() -> None:
    state = _make_evaluator_state()

    result = EvaluatorAgent(registry=FakeRegistry("needs_revision")).run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["decision"].route == DecisionRoute.CONTINUE_EXECUTE
    assert result["decision"].task_retry is not None
    assert task_state.status == TaskStatus.RUNNING
    assert task_state.latest_evaluate_checkpoint == "failed"
    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_001"
    assert "art_image_candidate_001" in task_state.retry_input_artifact_ids
    assert task_state.retry_context_text is not None


def test_evaluator_routes_replan_verdict_to_replan() -> None:
    state = _make_evaluator_state()

    result = EvaluatorAgent(registry=FakeRegistry("replan")).run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["decision"].route == DecisionRoute.REPLAN
    assert result["decision"].replan is not None
    assert task_state.status == TaskStatus.REPLANNED
    assert task_state.latest_evaluate_checkpoint == "failed"
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
