"""Minimal execute agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import (
    invoke_llm,
    invoke_structured_llm,
    invoke_structured_multimodal_llm,
    load_llm_config,
)
from runtime.artifact_context import (
    build_task_working_set_summary,
    preserve_from_task_working_set,
    register_task_artifact,
)
from runtime.instruction_resolver import (
    resolve_active_instruction_artifact,
    resolve_active_instruction_text,
)
from runtime.input_selector import prepare_task_inputs
from runtime.prompts import (
    EXECUTE_COLLAGE_LAYOUT_GOAL,
    EXECUTE_GROUNDING_QUERY,
    EXECUTE_OBSERVE_SYSTEM_PROMPT,
    EXECUTE_STRATEGY_SYSTEM_PROMPT,
    EXECUTE_UNDERSTAND_QUESTION,
    EXECUTION_HISTORY_SUMMARY_SYSTEM_PROMPT,
    build_execute_observe_user_prompt,
    build_execute_strategy_user_prompt,
    build_execution_history_summary_user_prompt,
)
from runtime.config import default_max_execute_acts
from runtime.state import RuntimeState
from runtime.tool_runner import ToolRunner
from schema import (
    ArtifactKind,
    CollageArgs,
    CropArgs,
    Decision,
    DecisionRoute,
    EditArgs,
    ExecuteLLMOutput,
    ExecutionOutcome,
    GroundingArgs,
    InstructionArtifact,
    ObserveLLMOutput,
    PromptReconstructArgs,
    ReplanMode,
    ReplanRequest,
    SegmentArgs,
    SessionPhase,
    TaskActRecord,
    TaskLoop,
    TaskRetryAdvice,
    TaskStatus,
    ToolInvocationRecord,
    ToolName,
    UnderstandArgs,
)
from tools.registry import ToolRegistry, build_default_tool_registry


EDIT_IMAGE_INPUT_LIMIT = 3


class EditInputBudgetExceeded(RuntimeError):
    """Raised when one edit round needs more images than the edit tool allows."""

    def __init__(self, attempted_refs: list[str]) -> None:
        self.attempted_refs = list(attempted_refs)
        self.attempted_count = len(self.attempted_refs)
        super().__init__(
            f"edit step requires {self.attempted_count} image refs, exceeding the "
            f"{EDIT_IMAGE_INPUT_LIMIT}-image limit; reduce the edit goal and retry "
            "with a smaller step"
        )


class ExecuteAgent:
    """Runs one execute loop containing multiple thinking-act-observe rounds."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or build_default_tool_registry()
        self._tool_runner = ToolRunner(self._registry)

    def run(self, state: RuntimeState) -> RuntimeState:
        session = state["session"]
        current_task_id = session.current_task_id
        if current_task_id is None:
            raise ValueError("current_task_id is required during execute")

        task = state["tasks"][current_task_id]
        selection = prepare_task_inputs(state, current_task_id)
        resolved_inputs = selection.selected_artifact_ids
        loop_index = session.task_states[current_task_id].loop_count + 1
        task_state = session.task_states[current_task_id]
        act_records: list[tuple[str, str]] = []
        latest_refs: list[str] = []
        previous_latest_refs = list(task_state.latest_artifact_ids)
        max_acts = state.get("max_execute_acts", default_max_execute_acts())
        overflow_retry: EditInputBudgetExceeded | None = None
        last_base_image_ref: str | None = None

        for act_index in range(1, max_acts + 1):
            strategy = (
                self._build_execution_strategy_with_llm(state, task)
                if self._use_llm(state)
                else None
            )
            thinking = (
                strategy.reasoning
                if strategy
                else self._build_rule_based_thinking(state, task, current_task_id, act_index)
            )
            selected_tool = self._resolve_next_tool(
                state=state,
                task=task,
                task_id=current_task_id,
                act_index=act_index,
                strategy=strategy,
            )
            base_image_ref = self._resolve_base_image_ref(
                state=state,
                task=task,
                task_id=current_task_id,
                resolved_inputs=resolved_inputs,
                strategy=strategy,
            )
            last_base_image_ref = base_image_ref
            execution = self._select_and_run_tool(
                state=state,
                task=task,
                task_id=current_task_id,
                loop_index=loop_index,
                selected_tool=selected_tool,
                resolved_inputs=resolved_inputs,
                base_image_ref=base_image_ref,
            )
            if execution.invocation.status == "failed":
                state["operations"].append(execution.invocation)
                error = execution.invocation.error or {}
                if error.get("type") == "EditInputBudgetExceeded":
                    observation = (
                        "Planned edit step exceeded the 3-image input budget: "
                        f"{error.get('message', '')}. Narrow the next edit goal."
                    )
                    overflow_retry = EditInputBudgetExceeded(
                        list(error.get("attempted_refs", []))
                    )
                else:
                    observation = (
                        f"Tool {selected_tool.value} failed: "
                        f"{error.get('message', 'unknown error')}"
                    )
                act_records.append((thinking, observation))
                state["task_act_records"].append(
                    TaskActRecord(
                        task_id=current_task_id,
                        loop_index=loop_index,
                        act_index=act_index,
                        thinking_text=thinking,
                        tool_name=selected_tool.value,
                        tool_args={},
                        output_artifact_ids=[],
                        observation_text=observation,
                    )
                )
                break

            state["operations"].append(execution.invocation)
            observe_result = self._observe_with_llm(
                state=state,
                task=task,
                task_id=current_task_id,
                selected_tool=selected_tool,
                execution=execution,
            )
            self._apply_observe_artifact_summaries(execution.artifacts, observe_result)
            for artifact in execution.artifacts:
                register_task_artifact(
                    state,
                    current_task_id,
                    artifact,
                    usage=artifact.role or f"artifact produced by {selected_tool.value}",
                    selection_reason=f"produced in the current execute cycle via {selected_tool.value}",
                )
            observation = observe_result.observation
            act_records.append((thinking, observation))
            state["task_act_records"].append(
                TaskActRecord(
                    task_id=current_task_id,
                    loop_index=loop_index,
                    act_index=act_index,
                    thinking_text=thinking,
                    tool_name=selected_tool.value,
                    tool_args=dict(execution.invocation.args),
                    output_artifact_ids=list(execution.invocation.output_refs),
                    observation_text=observation,
                )
            )

            if observe_result.outcome == "success":
                candidate_refs = [
                    artifact.id
                    for artifact in execution.artifacts
                    if self._is_evaluable_candidate_artifact(artifact)
                ]
                if candidate_refs:
                    latest_refs = candidate_refs
                    break

        task_state.loop_count += 1
        if overflow_retry is not None:
            task_state.latest_execution_outcome = ExecutionOutcome.FAILURE
            task_state.latest_artifact_ids = previous_latest_refs
            task_state.edit_input_budget_overflow_count += 1
            if task_state.edit_input_budget_overflow_count >= 2:
                task_state.latest_execute_checkpoint = "failed"
                self._clear_retry_context(task_state)
                task_state.status = TaskStatus.REPLANNED
                task_state.edit_input_budget_overflow_count = 0
                session.current_task_id = None
                session.phase = SessionPhase.PLANNING
                decision = self._build_edit_input_budget_replan_decision(
                    state=state,
                    task_id=current_task_id,
                    overflow_error=overflow_retry,
                )
            else:
                task_state.latest_execute_checkpoint = "retry"
                task_state.status = TaskStatus.RUNNING
                session.current_task_id = current_task_id
                session.phase = SessionPhase.EXECUTING
                decision = self._build_edit_input_budget_retry_decision(
                    state=state,
                    task_id=current_task_id,
                    overflow_error=overflow_retry,
                    base_image_ref=previous_latest_refs[-1] if previous_latest_refs else last_base_image_ref,
                )
                self._apply_execute_retry_context(
                    state=state,
                    task_id=current_task_id,
                    task_retry=decision.task_retry,
                )
            state["decision"] = decision
            session.latest_decision_id = decision.id
        elif latest_refs:
            task_state.latest_artifact_ids = latest_refs
            task_state.latest_execution_outcome = ExecutionOutcome.SUCCESS
            task_state.latest_execute_checkpoint = "passed"
            task_state.edit_input_budget_overflow_count = 0
            task_state.status = TaskStatus.WAITING_EVALUATION
            session.phase = SessionPhase.EVALUATING
        else:
            task_state.latest_artifact_ids = latest_refs
            task_state.latest_execution_outcome = ExecutionOutcome.FAILURE
            task_state.latest_execute_checkpoint = "failed"
            task_state.edit_input_budget_overflow_count = 0
            self._clear_retry_context(task_state)
            task_state.status = TaskStatus.REPLANNED
            session.current_task_id = None
            session.phase = SessionPhase.PLANNING
            decision = self._build_execute_failure_replan_decision(
                state=state,
                task_id=current_task_id,
            )
            state["decision"] = decision
            session.latest_decision_id = decision.id
            if decision.route == DecisionRoute.FAIL:
                task_state.status = TaskStatus.FAILED
                session.phase = SessionPhase.FAILED

        state["task_loops"].append(
            TaskLoop(
                id=f"loop_{current_task_id}_{loop_index:03d}",
                task_id=current_task_id,
                loop_index=loop_index,
                thinking="\n\n".join(
                    f"Act {index}: {thinking}"
                    for index, (thinking, _) in enumerate(act_records, start=1)
                ),
                selected_tools=[
                    record.tool_name
                    for record in state["task_act_records"]
                    if record.task_id == current_task_id and record.loop_index == loop_index
                ],
                output_artifact_ids=list(latest_refs),
                observation="\n".join(
                    f"{observation}" for _, observation in act_records
                ),
            )
        )
        state["session"] = session
        return state

    def _observe_with_llm(
        self,
        *,
        state: RuntimeState,
        task: Any,
        task_id: str,
        selected_tool: ToolName,
        execution,
    ) -> ObserveLLMOutput:
        if not self._use_llm(state):
            return ObserveLLMOutput(
                outcome="success" if selected_tool == ToolName.EDIT else "continue",
                observation=(
                    f"Act executed {selected_tool.value} and produced "
                    f"{len(execution.artifacts)} artifact(s)."
                ),
                artifact_summaries=[
                    {
                        "artifact_id": artifact.id,
                        "summary": artifact.summary
                        or f"{selected_tool.value} produced {artifact.kind} {artifact.id}",
                        "role": artifact.role or artifact.kind,
                    }
                    for artifact in execution.artifacts
                ],
            )
        image_paths = self._resolve_observe_image_paths(execution.artifacts)
        source_lines = []
        for artifact in execution.artifacts:
            if artifact.source_ids:
                for source_id in artifact.source_ids:
                    source_artifact = state["artifacts"].get(source_id)
                    if source_artifact is None:
                        continue
                    source_summary = source_artifact.summary
                    if source_summary is None and source_artifact.kind == "understanding":
                        source_summary = source_artifact.payload.get("summary")
                    source_lines.append(
                        f"- {source_id} | kind={source_artifact.kind} | summary={source_summary or '(no summary)'}"
                    )
        new_artifact_lines = [
            f"- {artifact.id} | kind={artifact.kind} | payload={artifact.payload} | source_ids={artifact.source_ids}"
            for artifact in execution.artifacts
        ]
        prompt = build_execute_observe_user_prompt(
            task=task,
            active_instruction=self._resolve_active_instruction_text_from_artifacts(state, task_id),
            retry_context=state["session"].task_states[task_id].retry_context_text,
            selected_tool=selected_tool.value,
            tool_args=execution.invocation.args,
            source_lines=source_lines,
            new_artifact_lines=new_artifact_lines,
        )
        if image_paths:
            return invoke_structured_multimodal_llm(
                system_prompt=EXECUTE_OBSERVE_SYSTEM_PROMPT,
                user_prompt=prompt,
                image_paths=image_paths,
                output_schema=ObserveLLMOutput,
            )
        return invoke_structured_llm(
            system_prompt=EXECUTE_OBSERVE_SYSTEM_PROMPT,
            user_prompt=prompt,
            output_schema=ObserveLLMOutput,
        )

    def _resolve_observe_image_paths(self, artifacts) -> list[str]:
        image_paths: list[str] = []
        for artifact in artifacts:
            if artifact.kind != "image" or not artifact.uri:
                continue
            if "://" in artifact.uri:
                raise ValueError(
                    f"Observe image artifact uri is not a local file path: {artifact.uri}"
                )
            path = Path(artifact.uri)
            if not path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {artifact.uri}")
            image_paths.append(str(path))
        return image_paths

    def _apply_observe_artifact_summaries(self, artifacts, observe_result: ObserveLLMOutput) -> None:
        annotations_by_id = {
            item.artifact_id: item
            for item in observe_result.artifact_summaries
        }
        for artifact in artifacts:
            annotation = annotations_by_id.get(artifact.id)
            if annotation is None:
                continue
            artifact.summary = annotation.summary
            if annotation.role:
                artifact.role = annotation.role

    def _build_execute_failure_replan_decision(
        self,
        *,
        state: RuntimeState,
        task_id: str,
    ) -> Decision:
        task = state["tasks"][task_id]
        task_state = state["session"].task_states[task_id]
        failed_operation = self._find_latest_failed_operation(state, task_id)
        failure_summary = "execute checkpoint reached the max loop budget before a successful output was produced"
        failure_issues = [
            "execute_loop_budget_exceeded",
            "current_task_did_not_reach_success_state",
        ]
        failure_reason = (
            "the current task kept requesting another loop until the execute budget was "
            "exhausted"
        )
        if failed_operation is not None:
            error = failed_operation.error or {}
            error_message = error.get("message", "tool execution failed")
            failure_count = self._count_failed_operations(state)
            if failure_count >= state.get("max_tool_failures", 3):
                return Decision(
                    id=f"dec_exec_fail_{task_id}_{task_state.loop_count + 1:03d}",
                    route=DecisionRoute.FAIL,
                    task_id=task_id,
                    plan_id=state["session"].current_plan_id,
                    source_execution_outcome=ExecutionOutcome.FAILURE,
                    candidate_artifact_ids=[],
                    summary=f"execute tool failures exceeded retry limit: {error_message}",
                    issues=[
                        "execute_tool_failed",
                        "tool_failure_retry_limit_exceeded",
                        f"tool={failed_operation.tool_name}",
                        f"error_type={error.get('type', 'unknown')}",
                    ],
                )
            failure_summary = f"execute tool {failed_operation.tool_name} failed: {error_message}"
            failure_issues = [
                "execute_tool_failed",
                f"tool={failed_operation.tool_name}",
                f"error_type={error.get('type', 'unknown')}",
            ]
            failure_reason = failure_summary
        return Decision(
            id=f"dec_exec_fail_{task_id}_{task_state.loop_count + 1:03d}",
            route=DecisionRoute.REPLAN,
            task_id=task_id,
            plan_id=state["session"].current_plan_id,
            source_execution_outcome=ExecutionOutcome.FAILURE,
            candidate_artifact_ids=[],
            summary=failure_summary,
            issues=failure_issues,
            replan=ReplanRequest(
                mode=self._fallback_replan_mode(task.type),
                reason=failure_reason,
                preserve_artifact_ids=preserve_from_task_working_set(state, task_id),
            ),
        )

    def _build_edit_input_budget_retry_decision(
        self,
        *,
        state: RuntimeState,
        task_id: str,
        overflow_error: EditInputBudgetExceeded,
        base_image_ref: str | None,
    ) -> Decision:
        task_state = state["session"].task_states[task_id]
        task_retry = TaskRetryAdvice(
            reason=(
                "The planned edit step required more than 3 input images. "
                "Shrink the next edit goal and continue the same task with a smaller step."
            ),
            base_candidate_artifact_id=base_image_ref,
            reuse_artifact_ids=[],
            fix_focuses=[
                "Reduce the next edit round so it uses at most 3 image inputs.",
                "Do not try to satisfy all references in one edit call.",
            ],
            avoid_changes=[
                "Do not silently drop reference images.",
                "Do not expand the edit goal before the input budget fits.",
            ],
        )
        return Decision(
            id=f"dec_exec_retry_{task_id}_{task_state.loop_count + 1:03d}",
            route=DecisionRoute.CONTINUE_EXECUTE,
            task_id=task_id,
            plan_id=state["session"].current_plan_id,
            source_execution_outcome=ExecutionOutcome.FAILURE,
            candidate_artifact_ids=list(task_state.latest_artifact_ids),
            summary=(
                f"planned edit step exceeded the {EDIT_IMAGE_INPUT_LIMIT}-image input "
                f"budget with {overflow_error.attempted_count} refs"
            ),
            issues=[
                "edit_input_budget_exceeded",
                f"attempted_count={overflow_error.attempted_count}",
                f"attempted_refs={overflow_error.attempted_refs}",
            ],
            task_retry=task_retry,
        )

    def _build_edit_input_budget_replan_decision(
        self,
        *,
        state: RuntimeState,
        task_id: str,
        overflow_error: EditInputBudgetExceeded,
    ) -> Decision:
        task = state["tasks"][task_id]
        task_state = state["session"].task_states[task_id]
        return Decision(
            id=f"dec_exec_fail_{task_id}_{task_state.loop_count + 1:03d}",
            route=DecisionRoute.REPLAN,
            task_id=task_id,
            plan_id=state["session"].current_plan_id,
            source_execution_outcome=ExecutionOutcome.FAILURE,
            candidate_artifact_ids=list(task_state.latest_artifact_ids),
            summary=(
                "the same task repeatedly planned edit steps that exceeded the "
                f"{EDIT_IMAGE_INPUT_LIMIT}-image input budget"
            ),
            issues=[
                "edit_input_budget_exceeded",
                "edit_input_budget_retry_limit_exceeded",
                f"attempted_count={overflow_error.attempted_count}",
                f"attempted_refs={overflow_error.attempted_refs}",
            ],
            replan=ReplanRequest(
                mode=self._fallback_replan_mode(task.type),
                reason=(
                    "the task repeatedly required more than 3 images in a single "
                    "edit step, so it should be replanned into smaller edit goals"
                ),
                preserve_artifact_ids=preserve_from_task_working_set(state, task_id),
            ),
        )

    def _find_latest_failed_operation(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> ToolInvocationRecord | None:
        for operation in reversed(state.get("operations", [])):
            if operation.task_id == task_id and operation.status == "failed":
                return operation
        return None

    def _count_failed_operations(self, state: RuntimeState) -> int:
        return sum(
            1
            for operation in state.get("operations", [])
            if operation.status == "failed"
            and (operation.error or {}).get("type") != "EditInputBudgetExceeded"
        )

    def _fallback_replan_mode(self, task_type: str) -> ReplanMode:
        if task_type in {"local_edit", "compose_subject", "reference_edit"}:
            return ReplanMode.SPLIT_TASK
        return ReplanMode.REROUTE_PLAN

    def _select_and_run_tool(
        self,
        *,
        state: RuntimeState,
        task: Any,
        task_id: str,
        loop_index: int,
        selected_tool: ToolName,
        resolved_inputs: list[str],
        base_image_ref: str,
    ):
        if selected_tool == ToolName.PROMPT_RECONSTRUCT:
            state["_prompt_reconstruct_context"] = self._build_prompt_reconstruct_context(
                state, task_id
            )
            return self._tool_runner.run(
                state,
                ToolName.PROMPT_RECONSTRUCT,
                task_id=task_id,
                loop_index=loop_index,
                args=PromptReconstructArgs(
                    input_artifact_ids=list(resolved_inputs),
                ),
            )

        reference_refs = self._resolve_reference_refs(state, resolved_inputs, base_image_ref)
        previous_candidate = (
            state["session"].task_states[task_id].latest_artifact_ids[-1]
            if state["session"].task_states[task_id].latest_artifact_ids
            else None
        )
        runtime_ctx = {
            "base_image_ref": previous_candidate or base_image_ref,
            "initial_base_image_ref": base_image_ref,
            "reference_refs": reference_refs,
            "grounding_ref": self._find_latest_grounding_ref(state, task_id),
            "mask_ref": self._find_latest_mask_ref(state, task_id),
            "crop_ref": self._find_latest_crop_ref(state, task_id),
        }
        return self._run_tool_step(
            state=state,
            task_id=task_id,
            loop_index=loop_index,
            tool_name=selected_tool,
            runtime_ctx=runtime_ctx,
        )

    def _resolve_base_image_ref(
        self,
        state: RuntimeState,
        task: Any,
        task_id: str,
        resolved_inputs: list[str],
        strategy: ExecuteLLMOutput | None,
    ) -> str:
        retry_candidate = self._resolve_retry_base_candidate_ref(state, task_id)
        if retry_candidate is not None:
            return retry_candidate
        if strategy and strategy.base_image_artifact_id and self._is_image_artifact(state, strategy.base_image_artifact_id):
            if strategy.base_image_artifact_id in resolved_inputs:
                return strategy.base_image_artifact_id
        for artifact_id in resolved_inputs:
            artifact = self._get_artifact(state, artifact_id)
            if artifact is not None and artifact.kind == "image":
                return artifact_id
        return resolved_inputs[0]

    def _resolve_reference_refs(
        self,
        state: RuntimeState,
        resolved_inputs: list[str],
        base_image_ref: str,
    ) -> list[str]:
        return [
            artifact_id
            for artifact_id in resolved_inputs
            if self._is_image_artifact(state, artifact_id)
            and artifact_id != base_image_ref
        ]

    def _resolve_retry_base_candidate_ref(self, state: RuntimeState, task_id: str) -> str | None:
        decision = state.get("decision")
        if (
            decision is not None
            and decision.route == DecisionRoute.CONTINUE_EXECUTE
            and decision.task_id == task_id
            and decision.task_retry is not None
            and decision.task_retry.base_candidate_artifact_id in state["artifacts"]
        ):
            return decision.task_retry.base_candidate_artifact_id
        return None

    def _get_artifact(self, state: RuntimeState, artifact_id: str):
        return state["artifacts"].get(artifact_id)

    def _is_evaluable_candidate_artifact(self, artifact) -> bool:
        if artifact.kind != ArtifactKind.IMAGE:
            return False
        return artifact.created_by == ToolName.EDIT.value or artifact.role == "candidate_image"

    def _is_image_artifact(self, state: RuntimeState, artifact_id: str) -> bool:
        artifact = self._get_artifact(state, artifact_id)
        return artifact is not None and artifact.kind == "image"

    def _resolve_next_tool(
        self,
        *,
        state: RuntimeState,
        task: Any,
        task_id: str,
        act_index: int,
        strategy: ExecuteLLMOutput | None,
    ) -> ToolName:
        if strategy and strategy.selected_tools:
            normalized = self._normalize_next_tool(strategy.selected_tools)
            if normalized is not None:
                return self._repair_next_tool(state, task_id, normalized)
        return self._fallback_next_tool(state, task, task_id, act_index)

    def _normalize_next_tool(self, selected_tools: list[str]) -> ToolName | None:
        for tool_name in selected_tools:
            try:
                tool = ToolName(tool_name)
            except ValueError:
                continue
            if tool == ToolName.EVALUATE:
                continue
            return tool
        return None

    def _repair_next_tool(
        self,
        state: RuntimeState,
        task_id: str,
        selected_tool: ToolName,
    ) -> ToolName:
        mask_ref = self._find_latest_mask_ref(state, task_id)
        crop_ref = self._find_latest_crop_ref(state, task_id)
        if selected_tool == ToolName.CROP and mask_ref is None:
            return ToolName.SEGMENT
        if selected_tool == ToolName.UNDERSTAND and crop_ref is None and mask_ref is not None:
            return ToolName.CROP
        if selected_tool == ToolName.UNDERSTAND and crop_ref is None:
            return ToolName.SEGMENT
        if selected_tool == ToolName.COLLAGE:
            block_ids = self._collect_collage_blocks(state, task_id, [])
            if len(block_ids) < 2:
                return ToolName.EDIT
        return selected_tool

    def _fallback_next_tool(
        self,
        state: RuntimeState,
        task: Any,
        task_id: str,
        act_index: int,
    ) -> ToolName:
        if task.type != "local_edit":
            return ToolName.EDIT

        mask_ref = self._find_latest_mask_ref(state, task_id)
        crop_ref = self._find_latest_crop_ref(state, task_id)
        if mask_ref is None:
            return ToolName.SEGMENT
        if crop_ref is None:
            return ToolName.CROP
        if not self._has_understanding_for_image(state, task_id, crop_ref):
            return ToolName.UNDERSTAND
        return ToolName.EDIT

    def _build_rule_based_thinking(
        self,
        state: RuntimeState,
        task: Any,
        task_id: str,
        act_index: int,
    ) -> str:
        retry_context = state["session"].task_states[task_id].retry_context_text
        if retry_context:
            return (
                "This is a retry for the same task. "
                f"Use the retry context below to decide the next single tool.\n{retry_context}"
            )
        if task.type == "local_edit":
            return (
                "This task may benefit from a weak chained template: "
                "segment, then crop, then understand, then edit. "
                "Choose only the next single tool."
            )
        if self._find_latest_mask_ref(state, task_id) is not None:
            return "A mask already exists. Choose the next single tool instead of restarting the whole chain."
        return "This task can start with a direct edit pass. Choose only the next single tool."

    def _run_tool_step(
        self,
        *,
        state: RuntimeState,
        task_id: str,
        loop_index: int,
        tool_name: ToolName,
        runtime_ctx: dict[str, Any],
    ):
        if tool_name == ToolName.GROUNDING:
            return self._tool_runner.run(
                state,
                ToolName.GROUNDING,
                task_id=task_id,
                loop_index=loop_index,
                args=GroundingArgs(
                    image_ref=runtime_ctx["base_image_ref"],
                    grounding_query=EXECUTE_GROUNDING_QUERY,
                    top_k=1,
                ),
            )

        if tool_name == ToolName.SEGMENT:
            return self._tool_runner.run(
                state,
                ToolName.SEGMENT,
                task_id=task_id,
                loop_index=loop_index,
                args=SegmentArgs(
                    image_ref=runtime_ctx["base_image_ref"],
                    prompt=self._resolve_active_instruction_text_from_artifacts(state, task_id),
                ),
            )

        if tool_name == ToolName.CROP:
            mask_ref = runtime_ctx["mask_ref"]
            grounding_ref = runtime_ctx["grounding_ref"]
            if mask_ref is None and grounding_ref is None:
                return self._tool_runner.failure_result(
                    state,
                    ToolName.CROP,
                    task_id=task_id,
                    loop_index=loop_index,
                    args={},
                    error=ValueError("crop requires mask_ref or grounding_ref from earlier task-local results"),
                )
            return self._tool_runner.run(
                state,
                ToolName.CROP,
                task_id=task_id,
                loop_index=loop_index,
                args=CropArgs(
                    image_ref=runtime_ctx["initial_base_image_ref"],
                    mask_ref=mask_ref,
                    grounding_ref=grounding_ref,
                ),
            )

        if tool_name == ToolName.UNDERSTAND:
            image_ref = runtime_ctx["crop_ref"] or runtime_ctx["base_image_ref"]
            return self._tool_runner.run(
                state,
                ToolName.UNDERSTAND,
                task_id=task_id,
                loop_index=loop_index,
                args=UnderstandArgs(
                    image_ref=image_ref,
                    question=EXECUTE_UNDERSTAND_QUESTION,
                ),
            )

        if tool_name == ToolName.COLLAGE:
            block_artifact_ids = self._collect_collage_blocks(
                state,
                task_id,
                runtime_ctx["reference_refs"],
            )
            if len(block_artifact_ids) < 2:
                return self._tool_runner.failure_result(
                    state,
                    ToolName.COLLAGE,
                    task_id=task_id,
                    loop_index=loop_index,
                    args={},
                    error=ValueError("collage requires at least two image references"),
                )
            return self._tool_runner.run(
                state,
                ToolName.COLLAGE,
                task_id=task_id,
                loop_index=loop_index,
                args=CollageArgs(
                    block_artifact_ids=block_artifact_ids,
                    layout_goal=EXECUTE_COLLAGE_LAYOUT_GOAL,
                ),
            )

        if tool_name == ToolName.EDIT:
            try:
                edit_args = EditArgs(
                    instruction=self._resolve_active_instruction_text_from_artifacts(
                        state, task_id
                    ),
                    image_refs=self._build_edit_image_refs(
                        state=state,
                        task_id=task_id,
                        runtime_ctx=runtime_ctx,
                    ),
                )
            except Exception as exc:
                return self._tool_runner.failure_result(
                    state,
                    ToolName.EDIT,
                    task_id=task_id,
                    loop_index=loop_index,
                    args={},
                    error=exc,
                )
            return self._tool_runner.run(
                state,
                ToolName.EDIT,
                task_id=task_id,
                loop_index=loop_index,
                args=edit_args,
            )

        return self._tool_runner.failure_result(
            state,
            tool_name,
            task_id=task_id,
            loop_index=loop_index,
            args={},
            error=ValueError(f"Unsupported tool in execute loop: {tool_name}"),
        )

    def _find_latest_mask_ref(self, state: RuntimeState, task_id: str) -> str | None:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if artifact is not None and artifact.kind == "mask":
                return artifact_id
        return None

    def _find_latest_grounding_ref(self, state: RuntimeState, task_id: str) -> str | None:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if artifact is not None and artifact.kind == "geometry":
                return artifact_id
        return None

    def _find_latest_crop_ref(self, state: RuntimeState, task_id: str) -> str | None:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if (
                artifact is not None
                and artifact.kind == "image"
                and artifact.role == "cropped_preview"
            ):
                return artifact_id
        return None

    def _find_latest_collage_ref(self, state: RuntimeState, task_id: str) -> str | None:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if (
                artifact is not None
                and artifact.kind == "image"
                and artifact.role == "collage_reference"
            ):
                return artifact_id
        return None

    def _build_edit_image_refs(
        self,
        *,
        state: RuntimeState,
        task_id: str,
        runtime_ctx: dict[str, Any],
    ) -> list[str]:
        image_refs: list[str] = []

        def append_image_ref(artifact_id: str | None) -> None:
            if artifact_id is None:
                return
            artifact = self._get_artifact(state, artifact_id)
            if artifact is None or artifact.kind != "image":
                return
            if artifact_id in image_refs:
                return
            image_refs.append(artifact_id)

        append_image_ref(runtime_ctx["base_image_ref"])
        append_image_ref(self._find_latest_collage_ref(state, task_id))
        append_image_ref(runtime_ctx["crop_ref"])
        for artifact_id in runtime_ctx["reference_refs"]:
            append_image_ref(artifact_id)

        if not image_refs:
            raise ValueError("edit requires at least one image input")
        if len(image_refs) > EDIT_IMAGE_INPUT_LIMIT:
            raise EditInputBudgetExceeded(image_refs)
        return image_refs

    def _has_understanding_for_image(
        self,
        state: RuntimeState,
        task_id: str,
        image_ref: str,
    ) -> bool:
        for artifact_id in state["session"].task_states[task_id].task_artifact_ids:
            artifact = self._get_artifact(state, artifact_id)
            if (
                artifact is not None
                and artifact.kind == "understanding"
                and artifact.payload.get("image_ref") == image_ref
            ):
                return True
        return False

    def _use_llm(self, state: RuntimeState) -> bool:
        if not state["input"].get("use_llm", False):
            return False
        try:
            config = load_llm_config()
        except Exception:
            return False
        return bool(config.api_key and config.base_url and config.model_name)

    def _build_execution_strategy_with_llm(
        self,
        state: RuntimeState,
        task: Any,
    ) -> ExecuteLLMOutput:
        resolved_ids = state["session"].task_states[task.id].resolved_input_artifact_ids
        return invoke_structured_llm(
            system_prompt=EXECUTE_STRATEGY_SYSTEM_PROMPT,
            user_prompt=build_execute_strategy_user_prompt(
                task=task,
                user_instruction=state["input"]["instruction_text"],
                resolved_ids=resolved_ids,
                resolved_image_summaries=self._build_resolved_image_summaries(state, task.id, resolved_ids),
                retry_context=state["session"].task_states[task.id].retry_context_text,
                active_instruction=self._resolve_active_instruction_text_from_artifacts(state, task.id),
                task_working_set_summary=self._build_task_working_set_context(state, task.id),
                task_artifact_context=self._build_task_artifact_context(state, task.id),
                latest_candidate_refs=state["session"].task_states[task.id].latest_artifact_ids,
            ),
            output_schema=ExecuteLLMOutput,
        )

    def _build_task_working_set_context(self, state: RuntimeState, task_id: str) -> str:
        return build_task_working_set_summary(state, task_id)

    def _build_task_artifact_context(self, state: RuntimeState, task_id: str) -> str:
        lines: list[str] = []
        for artifact_id in state["session"].task_states[task_id].task_artifact_ids[-12:]:
            artifact = self._get_artifact(state, artifact_id)
            if artifact is None:
                continue
            if artifact.kind == "image":
                role = artifact.role or artifact.payload.get("role", "image")
                if role == "collage_reference":
                    block_ids = artifact.payload.get("block_artifact_ids", [])
                    lines.append(f"[{artifact_id}] image role={role} summary={artifact.summary or ''} blocks={block_ids}")
                else:
                    lines.append(f"[{artifact_id}] image role={role} summary={artifact.summary or ''}")
            elif artifact.kind == "geometry":
                target = artifact.payload.get("grounding_query")
                candidates = artifact.payload.get("candidates", [])
                lines.append(f"[{artifact_id}] geometry summary={artifact.summary or ''} target={target} candidates={candidates}")
            elif artifact.kind == "mask":
                score = artifact.payload.get("mask_score")
                target = artifact.payload.get("target")
                lines.append(f"[{artifact_id}] mask summary={artifact.summary or ''} target={target} score={score}")
            elif artifact.kind == "understanding":
                summary = artifact.payload.get("summary", "")
                image_ref = artifact.payload.get("image_ref", "")
                lines.append(f"[{artifact_id}] understanding image_ref={image_ref} summary={summary}")
            elif artifact.kind == "evaluation":
                verdict = artifact.payload.get("verdict", "")
                summary = artifact.summary or artifact.payload.get("summary", "")
                lines.append(f"[{artifact_id}] evaluation verdict={verdict} summary={summary}")
            elif artifact.kind == "instruction":
                text = artifact.summary or artifact.payload.get("instruction_text", "")
                lines.append(f"[{artifact_id}] instruction text={text}")
        return "\n".join(lines) or "(no task-local artifacts yet)"

    def _build_resolved_image_summaries(
        self,
        state: RuntimeState,
        task_id: str,
        resolved_ids: list[str],
    ) -> str:
        lines: list[str] = []
        for artifact_id in resolved_ids:
            artifact = self._get_artifact(state, artifact_id)
            if artifact is None or artifact.kind != "image":
                continue
            summary = artifact.summary or self._find_latest_understanding_summary_for_image(state, task_id, artifact_id)
            lines.append(f"- {artifact_id}: {summary}")
        return "\n".join(lines) or "(no image candidates)"

    def _find_latest_understanding_summary_for_image(
        self,
        state: RuntimeState,
        task_id: str,
        image_ref: str,
    ) -> str:
        task_artifact_ids = state["session"].task_states[task_id].task_artifact_ids
        for artifact_id in reversed(task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if (
                artifact is not None
                and artifact.kind == "understanding"
                and artifact.payload.get("image_ref") == image_ref
            ):
                return artifact.payload.get("summary", "")
        for artifact in reversed(list(state["artifacts"].values())):
            if (
                artifact.kind == "understanding"
                and artifact.payload.get("image_ref") == image_ref
            ):
                return artifact.payload.get("summary", "")
        return "(no understanding summary)"

    def _collect_collage_blocks(
        self,
        state: RuntimeState,
        task_id: str,
        reference_refs: list[str],
    ) -> list[str]:
        block_ids: list[str] = []
        for artifact_id in reference_refs:
            if self._is_image_artifact(state, artifact_id) and artifact_id not in block_ids:
                block_ids.append(artifact_id)
        for artifact_id in state["session"].task_states[task_id].resolved_input_artifact_ids:
            if self._is_image_artifact(state, artifact_id) and artifact_id not in block_ids:
                block_ids.append(artifact_id)
        return block_ids

    def _clear_retry_context(self, task_state) -> None:
        task_state.retry_input_artifact_ids = []
        task_state.retry_context_text = None

    def _apply_execute_retry_context(
        self,
        *,
        state: RuntimeState,
        task_id: str,
        task_retry: TaskRetryAdvice | None,
    ) -> None:
        if task_retry is None:
            return
        task_state = state["session"].task_states[task_id]
        retry_ids = list(task_state.resolved_input_artifact_ids)
        if task_retry.base_candidate_artifact_id and task_retry.base_candidate_artifact_id in state["artifacts"]:
            if task_retry.base_candidate_artifact_id not in retry_ids:
                retry_ids.append(task_retry.base_candidate_artifact_id)
        latest_instruction_id = self._find_latest_instruction_artifact_id(state, task_id)
        if latest_instruction_id and latest_instruction_id not in retry_ids:
            retry_ids.append(latest_instruction_id)
        for artifact_id in task_retry.reuse_artifact_ids:
            if artifact_id in state["artifacts"] and artifact_id not in retry_ids:
                retry_ids.append(artifact_id)
        task_state.retry_input_artifact_ids = retry_ids
        task_state.retry_context_text = self._build_retry_context_text(task_retry)
        task_state.task_working_set = []
        task_state.resolved_input_artifact_ids = []
        task_state.input_selection_reasoning = None
        task_state.input_validation_summary = None

    def _find_latest_instruction_artifact_id(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> str | None:
        artifact = resolve_active_instruction_artifact(state, task_id)
        return artifact.id if artifact is not None else None

    def _build_retry_context_text(self, task_retry: TaskRetryAdvice) -> str:
        lines = [
            "Retry feedback:",
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

    def _has_reconstructed_instruction(self, state: RuntimeState, task_id: str) -> bool:
        for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
            artifact = self._get_artifact(state, artifact_id)
            if (
                isinstance(artifact, InstructionArtifact)
                and artifact.created_by == ToolName.PROMPT_RECONSTRUCT.value
            ):
                return True
        return False

    def _resolve_active_instruction_text_from_artifacts(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> str:
        return resolve_active_instruction_text(state, task_id)

    def _build_prompt_reconstruct_context(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> str:
        task = state["tasks"][task_id]
        baseline = (
            f"Task type: {task.type}\n"
            f"Original task instruction: {task.instruction}\n"
            f"Current active instruction: {self._resolve_active_instruction_text_from_artifacts(state, task_id)}\n"
            f"Acceptance criteria: {task.acceptance_criteria}\n"
            f"Selected input artifacts: {state['session'].task_states[task_id].resolved_input_artifact_ids}\n"
            f"Current task artifacts:\n{self._build_task_artifact_context(state, task_id)}\n"
        )
        recent_raw = self._build_recent_raw_tao_text(state, task_id)
        earlier_summary = self._summarize_earlier_tao_rounds(state, task_id)
        return (
            baseline
            + "\nRecent raw rounds:\n"
            + recent_raw
            + "\n\nEarlier rounds summary:\n"
            + earlier_summary
            + "\n\nRewrite the active instruction so it is clearer, more detailed, and replaces ambiguous image references with explicit artifact ids."
        )

    def _build_recent_raw_tao_text(self, state: RuntimeState, task_id: str) -> str:
        records = [
            record for record in state["task_act_records"] if record.task_id == task_id
        ]
        recent = records[-5:]
        if not recent:
            return "(no previous rounds)"
        blocks = []
        for record in recent:
            blocks.append(
                f"Loop {record.loop_index} Act {record.act_index}\n"
                f"Thinking:\n{record.thinking_text}\n"
                f"Act:\ntool={record.tool_name}\nargs={record.tool_args}\n"
                f"Observe:\n{record.observation_text}\n"
                f"Outputs: {record.output_artifact_ids}"
            )
        return "\n\n".join(blocks)

    def _summarize_earlier_tao_rounds(self, state: RuntimeState, task_id: str) -> str:
        records = [
            record for record in state["task_act_records"] if record.task_id == task_id
        ]
        if len(records) <= 5:
            return "(no earlier rounds)"
        earlier = records[-15:-5]
        if not earlier:
            return "(no earlier rounds)"
        raw = "\n\n".join(
            (
                f"Loop {record.loop_index} Act {record.act_index}\n"
                f"Thinking:\n{record.thinking_text}\n"
                f"Act:\ntool={record.tool_name}\nargs={record.tool_args}\n"
                f"Observe:\n{record.observation_text}\n"
                f"Outputs: {record.output_artifact_ids}"
            )
            for record in earlier
        )
        response = invoke_llm(
            system_prompt=EXECUTION_HISTORY_SUMMARY_SYSTEM_PROMPT,
            user_prompt=build_execution_history_summary_user_prompt(
                task_instruction=state["tasks"][task_id].instruction,
                raw_records=raw,
            ),
        )
        return str(response.content).strip()
