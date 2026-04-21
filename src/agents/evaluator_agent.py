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
    ReplanMode,
    ReplanRequest,
    SessionPhase,
    EvaluateArgs,
    TaskRetryAdvice,
    TaskStatus,
    ToolName,
)
from tools import ToolRegistry, build_default_tool_registry


class EvaluatorAgent:
    """Converts a desired route into a typed Decision and updates state."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or build_default_tool_registry()

    def run(self, state: RuntimeState) -> RuntimeState:
        session = state["session"]
        current_task_id = session.current_task_id
        if current_task_id is None:
            raise ValueError("current_task_id is required during evaluate")

        task_state = session.task_states[current_task_id]
        if not task_state.latest_artifact_ids:
            task_state.status = TaskStatus.RUNNING
            session.phase = SessionPhase.EXECUTING
            state["session"] = session
            return state
        task_state.evaluator_checkpoint_count += 1
        evaluate_execution = self._registry.get(ToolName.EVALUATE).run(
            state,
            task_id=current_task_id,
            loop_index=task_state.loop_count,
            args=EvaluateArgs(
                candidate_refs=list(task_state.latest_artifact_ids),
                checks=state["tasks"][current_task_id].acceptance_criteria,
            ),
        )
        state["operations"].append(evaluate_execution.invocation)
        for artifact in evaluate_execution.artifacts:
            state["artifacts"][artifact.id] = artifact
            if artifact.id not in task_state.task_artifact_ids:
                task_state.task_artifact_ids.append(artifact.id)

        llm_decision = self._build_decision_with_llm(state, current_task_id) if self._use_llm(state) else None
        route = llm_decision.route if llm_decision else self._fallback_route(state, current_task_id)

        common = {
            "id": "dec_001",
            "route": route,
            "task_id": current_task_id,
            "plan_id": session.current_plan_id,
            "source_execution_outcome": task_state.latest_execution_outcome,
            "candidate_artifact_ids": list(task_state.latest_artifact_ids),
            "summary": llm_decision.summary if llm_decision else "minimal evaluator checkpoint decision",
            "issues": llm_decision.issues if llm_decision else [],
        }

        if route == DecisionRoute.CONTINUE_EXECUTE:
            decision = Decision(
                **common,
                task_retry=TaskRetryAdvice(
                    reason="continue current task for one more refinement pass",
                    base_candidate_artifact_id=task_state.latest_artifact_ids[0],
                    fix_focuses=["improve candidate quality"],
                ),
            )
            task_state.status = TaskStatus.RUNNING
            session.phase = SessionPhase.EXECUTING
        elif route == DecisionRoute.REPLAN:
            decision = Decision(
                **common,
                replan=ReplanRequest(
                    mode=ReplanMode.REROUTE_PLAN,
                    reason="current task should be replanned",
                    preserve_artifact_ids=list(task_state.latest_artifact_ids),
                ),
            )
            task_state.status = TaskStatus.REPLANNED
            session.current_task_id = None
            session.phase = SessionPhase.PLANNING
        elif route == DecisionRoute.FAIL:
            decision = Decision(**common)
            task_state.status = TaskStatus.FAILED
            session.current_task_id = None
            session.phase = SessionPhase.FAILED
        else:
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

    def _fallback_route(self, state: RuntimeState, task_id: str) -> DecisionRoute:
        max_task_loops = state.get("max_task_loops", 1)
        task_state = state["session"].task_states[task_id]
        if task_state.loop_count < max_task_loops:
            return DecisionRoute.CONTINUE_EXECUTE
        return DecisionRoute(state["input"]["desired_decision_route"])

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
        return invoke_structured_llm(
            system_prompt=(
                "You are an evaluator agent for image editing. "
                "Return only one route from continue_execute, pass, replan, fail."
            ),
            user_prompt=(
                f"User instruction: {state['input']['instruction_text']}\n"
                f"Task type: {task.type}\n"
                f"Task instruction: {task.instruction}\n"
                f"Candidate refs: {state['session'].task_states[task_id].latest_artifact_ids}\n"
                f"Execution outcome: {state['session'].task_states[task_id].latest_execution_outcome}\n"
                "Choose the next route and summarize why."
            ),
            output_schema=DecisionLLMOutput,
        )
