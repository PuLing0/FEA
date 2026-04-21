"""Minimal evaluator agent."""

from __future__ import annotations

from llm import invoke_structured_llm, load_llm_config
from runtime.scheduler import select_next_runnable_task
from runtime.state import RuntimeState
from schema import (
    ArtifactKind,
    Decision,
    DecisionLLMOutput,
    DecisionRoute,
    EvaluateArgs,
    ReplanMode,
    ReplanRequest,
    SessionPhase,
    TaskRetryAdvice,
    TaskStatus,
    ToolName,
)
from tools import ToolRegistry, build_default_tool_registry


class EvaluatorAgent:
    """Evaluates successful task outputs and routes the next step."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or build_default_tool_registry()

    def run(self, state: RuntimeState) -> RuntimeState:
        session = state["session"]
        current_task_id = session.current_task_id
        if current_task_id is None:
            raise ValueError("current_task_id is required during evaluate")

        task = state["tasks"][current_task_id]
        task_state = session.task_states[current_task_id]
        if task_state.latest_execute_checkpoint != "passed" or not task_state.latest_artifact_ids:
            raise ValueError("EvaluatorAgent only accepts successful execute checkpoint outputs")

        task_state.evaluator_checkpoint_count += 1
        evaluate_execution = self._registry.get(ToolName.EVALUATE).run(
            state,
            task_id=current_task_id,
            loop_index=task_state.loop_count,
            args=EvaluateArgs(
                candidate_refs=list(task_state.latest_artifact_ids),
                checks=task.acceptance_criteria,
            ),
        )
        state["operations"].append(evaluate_execution.invocation)
        for artifact in evaluate_execution.artifacts:
            state["artifacts"][artifact.id] = artifact
            if artifact.id not in task_state.task_artifact_ids:
                task_state.task_artifact_ids.append(artifact.id)

        llm_decision = (
            self._build_decision_with_llm(state, current_task_id)
            if self._use_llm(state)
            else None
        )
        route = (
            llm_decision.route
            if llm_decision
            else self._fallback_route(state, current_task_id)
        )

        common = {
            "id": f"dec_eval_{current_task_id}_{task_state.evaluator_checkpoint_count:03d}",
            "route": route,
            "task_id": current_task_id,
            "plan_id": session.current_plan_id,
            "source_execution_outcome": task_state.latest_execution_outcome,
            "candidate_artifact_ids": list(task_state.latest_artifact_ids),
            "summary": llm_decision.summary if llm_decision else "minimal evaluator checkpoint decision",
            "issues": llm_decision.issues if llm_decision else [],
        }

        if route == DecisionRoute.CONTINUE_EXECUTE:
            task_state.latest_evaluate_checkpoint = "failed"
            task_retry = TaskRetryAdvice(
                reason="the task output is not good enough yet, but it is still worth refining in the same task",
                base_candidate_artifact_id=task_state.latest_artifact_ids[-1],
                reuse_artifact_ids=self._build_reuse_artifact_ids(task_state),
                fix_focuses=["improve the current candidate based on evaluator feedback"],
                avoid_changes=["do not discard the original input images"],
            )
            decision = Decision(
                **common,
                task_retry=task_retry,
            )
            self._apply_task_retry_context(state, current_task_id, task_retry)
            task_state.status = TaskStatus.RUNNING
            session.phase = SessionPhase.EXECUTING
        elif route == DecisionRoute.REPLAN:
            task_state.latest_evaluate_checkpoint = "failed"
            self._clear_task_retry_context(task_state)
            decision = Decision(
                **common,
                replan=ReplanRequest(
                    mode=self._fallback_replan_mode(task.type),
                    reason="the current task should not continue as-is after evaluator review",
                    preserve_artifact_ids=list(task_state.task_artifact_ids[-6:]),
                ),
            )
            task_state.status = TaskStatus.REPLANNED
            session.current_task_id = None
            session.phase = SessionPhase.PLANNING
        elif route == DecisionRoute.FAIL:
            task_state.latest_evaluate_checkpoint = "failed"
            self._clear_task_retry_context(task_state)
            decision = Decision(**common)
            task_state.status = TaskStatus.FAILED
            session.current_task_id = None
            session.phase = SessionPhase.FAILED
        else:
            task_state.latest_evaluate_checkpoint = "passed"
            self._clear_task_retry_context(task_state)
            decision = Decision(**common)
            task_state.status = TaskStatus.PASSED
            if task_state.latest_artifact_ids:
                task_state.final_artifact_id = task_state.latest_artifact_ids[-1]
                if task_state.final_artifact_id not in state["session"].artifact_index.by_type.setdefault(ArtifactKind.IMAGE, []):
                    state["session"].artifact_index.by_type[ArtifactKind.IMAGE].append(task_state.final_artifact_id)
            next_task_id = select_next_runnable_task(state)
            if next_task_id is None:
                session.final_result_id = task_state.final_artifact_id
                session.current_task_id = None
                session.phase = SessionPhase.DONE
            else:
                session.current_task_id = next_task_id
                session.task_states[next_task_id].status = TaskStatus.RUNNING
                session.phase = SessionPhase.EXECUTING

        session.latest_decision_id = decision.id
        state["decision"] = decision
        state["session"] = session
        return state

    def _apply_task_retry_context(
        self,
        state: RuntimeState,
        task_id: str,
        task_retry: TaskRetryAdvice,
    ) -> None:
        task_state = state["session"].task_states[task_id]
        retry_ids: list[str] = []

        for artifact_id in state["session"].artifact_index.by_type.get(ArtifactKind.IMAGE, []):
            if artifact_id.startswith("art_img_input_") and artifact_id not in retry_ids:
                retry_ids.append(artifact_id)

        if task_retry.base_candidate_artifact_id and task_retry.base_candidate_artifact_id in state["artifacts"]:
            retry_ids.append(task_retry.base_candidate_artifact_id)

        latest_instruction_id = self._find_latest_instruction_artifact_id(state, task_id)
        if latest_instruction_id:
            retry_ids.append(latest_instruction_id)

        for artifact_id in task_retry.reuse_artifact_ids:
            if artifact_id in state["artifacts"] and artifact_id not in retry_ids:
                retry_ids.append(artifact_id)

        task_state.retry_input_artifact_ids = retry_ids
        task_state.retry_context_text = self._build_retry_context_text(task_retry)
        task_state.resolved_input_artifact_ids = []
        task_state.input_selection_reasoning = None
        task_state.input_validation_summary = None

    def _clear_task_retry_context(self, task_state) -> None:
        task_state.retry_input_artifact_ids = []
        task_state.retry_context_text = None

    def _find_latest_instruction_artifact_id(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> str | None:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = state["artifacts"].get(artifact_id)
            if artifact is not None and artifact.kind == "instruction":
                return artifact_id
        return None

    def _build_retry_context_text(self, task_retry: TaskRetryAdvice) -> str:
        lines = [
            "Evaluator feedback:",
            f"- Reason: {task_retry.reason}",
        ]
        if task_retry.base_candidate_artifact_id:
            lines.append(f"- Base candidate: {task_retry.base_candidate_artifact_id}")
        if task_retry.reuse_artifact_ids:
            lines.append(f"- Reuse artifacts: {', '.join(task_retry.reuse_artifact_ids)}")
        if task_retry.fix_focuses:
            lines.append("- Fix focuses:")
            for index, item in enumerate(task_retry.fix_focuses, start=1):
                lines.append(f"  {index}. {item}")
        if task_retry.avoid_changes:
            lines.append("- Avoid changes:")
            for index, item in enumerate(task_retry.avoid_changes, start=1):
                lines.append(f"  {index}. {item}")
        return "\n".join(lines)

    def _build_reuse_artifact_ids(self, task_state) -> list[str]:
        return [
            artifact_id
            for artifact_id in task_state.task_artifact_ids[-8:]
            if artifact_id != task_state.latest_artifact_ids[-1]
        ]

    def _fallback_route(self, state: RuntimeState, task_id: str) -> DecisionRoute:
        max_evaluator_checkpoints = state.get("max_evaluator_checkpoints", 3)
        task_state = state["session"].task_states[task_id]
        desired_route = DecisionRoute(state["input"]["desired_decision_route"])
        if desired_route == DecisionRoute.FAIL:
            return DecisionRoute.FAIL
        if task_state.evaluator_checkpoint_count >= max_evaluator_checkpoints:
            return DecisionRoute.REPLAN
        return desired_route

    def _fallback_replan_mode(self, task_type: str) -> ReplanMode:
        if task_type in {"local_edit", "compose_subject", "reference_edit"}:
            return ReplanMode.SPLIT_TASK
        return ReplanMode.REROUTE_PLAN

    def _use_llm(self, state: RuntimeState) -> bool:
        if not state["input"].get("use_llm", False):
            return False
        try:
            config = load_llm_config()
        except Exception:
            return False
        return bool(config.api_key and config.base_url and config.model_name)

    def _build_decision_with_llm(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> DecisionLLMOutput:
        task = state["tasks"][task_id]
        task_state = state["session"].task_states[task_id]
        return invoke_structured_llm(
            system_prompt=(
                "You are an evaluator agent for image editing. "
                "You only evaluate a successful execute checkpoint output. "
                "Return only one route from continue_execute, pass, replan, fail."
            ),
            user_prompt=(
                f"User instruction: {state['input']['instruction_text']}\n"
                f"Task type: {task.type}\n"
                f"Task instruction: {task.instruction}\n"
                f"Acceptance criteria: {task.acceptance_criteria}\n"
                f"Candidate refs: {task_state.latest_artifact_ids}\n"
                f"Evaluator checkpoint count: {task_state.evaluator_checkpoint_count}\n"
                f"Execution outcome: {task_state.latest_execution_outcome}\n"
                "Choose the next route and summarize why."
            ),
            output_schema=DecisionLLMOutput,
        )
