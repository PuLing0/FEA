"""Minimal planning agent."""

from __future__ import annotations

from typing import Any

from llm import invoke_structured_llm, load_llm_config
from runtime.artifact_context import add_to_session_working_set, register_artifact_in_session_pool
from runtime.input_selector import initialize_task_working_set_from_session
from runtime.message_store import append_artifact_ref_message, append_message
from runtime.prompts import PLAN_SYSTEM_PROMPT, build_plan_user_prompt
from runtime.scheduler import select_next_runnable_task
from runtime.state import RuntimeState
from schema import (
    DecisionRoute,
    InstructionArtifact,
    Plan,
    PlanBlock,
    PlanLLMOutput,
    PlanTaskSpec,
    ReplanMode,
    SessionPhase,
    Task,
    TaskBlock,
    TaskState,
    TaskStatus,
)


class PlanAgent:
    """Creates a plan and the first runnable task."""

    def run(self, state: RuntimeState) -> RuntimeState:
        session = state["session"]
        session.phase = SessionPhase.EXECUTING
        if self._is_replan(state):
            self._apply_replan_plan(state)
        else:
            self._apply_initial_plan(state)
        state["session"] = session
        return state

    def _apply_initial_plan(self, state: RuntimeState) -> None:
        session = state["session"]
        llm_plan = self._build_plan_with_llm(state) if self._use_llm(state) else None
        if llm_plan and self._validate_llm_plan_output(state, llm_plan, preserved_task_ids=[]):
            task_specs = llm_plan.tasks
            plan_instruction = llm_plan.plan_instruction
        else:
            task_specs = self._build_fallback_tasks(state)
            plan_instruction = state["input"]["instruction_text"]

        plan_id = self._next_plan_id(state)
        tasks = self._materialize_tasks_from_specs(plan_id, task_specs)
        plan_obj = Plan(
            id=plan_id,
            instruction=plan_instruction,
            task_ids=[task.id for task in tasks],
            input_artifact_ids=[entry.artifact_id for entry in state["session"].session_working_set],
            understanding_artifact_ids=[],
        )

        state["plans"][plan_obj.id] = plan_obj
        for task in tasks:
            state["tasks"][task.id] = task
            session.task_states[task.id] = TaskState(
                task_id=task.id,
                status=TaskStatus.PENDING,
            )
            self._attach_instruction_artifact(state, task)
        self._append_plan_messages(state, plan_obj, tasks)

        session.current_plan_id = plan_obj.id
        session.current_task_id = None
        first_runnable = select_next_runnable_task(state)
        session.current_task_id = first_runnable
        if first_runnable is not None:
            session.task_states[first_runnable].status = TaskStatus.RUNNING
            initialize_task_working_set_from_session(state, first_runnable)

    def _apply_replan_plan(self, state: RuntimeState) -> None:
        session = state["session"]
        decision = state["decision"]
        replan = decision.replan
        if replan is None:
            raise ValueError("replan decision requires decision.replan")

        old_plan_id = decision.plan_id or session.current_plan_id
        if old_plan_id is None:
            raise ValueError("replan requires an existing current or decision plan id")

        retained_prefix_task_ids = self._build_retained_prefix_task_ids(state, old_plan_id)
        llm_plan = self._build_plan_with_llm(state) if self._use_llm(state) else None
        if llm_plan and self._validate_llm_plan_output(state, llm_plan, preserved_task_ids=retained_prefix_task_ids):
            task_specs = llm_plan.tasks
            plan_instruction = llm_plan.plan_instruction
        else:
            task_specs = self._build_replan_fallback_tasks(state)
            plan_instruction = self._build_replan_instruction(state)

        new_plan_id = self._next_plan_id(state)
        new_tasks = self._materialize_tasks_from_specs(new_plan_id, task_specs)
        merged_task_ids = retained_prefix_task_ids + [task.id for task in new_tasks]
        new_plan = Plan(
            id=new_plan_id,
            instruction=plan_instruction,
            task_ids=merged_task_ids,
            input_artifact_ids=self._build_replan_input_artifact_ids(state),
            understanding_artifact_ids=[],
        )

        self._mark_old_plan_tasks_for_replan(state, old_plan_id)
        for artifact_id in replan.preserve_artifact_ids:
            add_to_session_working_set(
                state,
                artifact_id,
                usage="replan preserved artifact",
                selection_reason="carried forward from the failed task working set for replanning",
            )

        state["plans"][new_plan.id] = new_plan
        for task in new_tasks:
            state["tasks"][task.id] = task
            session.task_states[task.id] = TaskState(
                task_id=task.id,
                status=TaskStatus.PENDING,
            )
            self._attach_instruction_artifact(state, task)
        self._append_plan_messages(state, new_plan, new_tasks)

        session.current_plan_id = new_plan.id
        session.current_task_id = None
        first_runnable = select_next_runnable_task(state)
        session.current_task_id = first_runnable
        if first_runnable is not None:
            session.task_states[first_runnable].status = TaskStatus.RUNNING
            initialize_task_working_set_from_session(state, first_runnable)

    def _is_replan(self, state: RuntimeState) -> bool:
        decision = state.get("decision")
        return (
            decision is not None
            and decision.route == DecisionRoute.REPLAN
            and decision.replan is not None
        )

    def _build_retained_prefix_task_ids(self, state: RuntimeState, old_plan_id: str) -> list[str]:
        old_plan = state["plans"][old_plan_id]
        return [
            task_id
            for task_id in old_plan.task_ids
            if state["session"].task_states[task_id].status == TaskStatus.PASSED
        ]

    def _build_replan_input_artifact_ids(self, state: RuntimeState) -> list[str]:
        decision = state["decision"]
        replan = decision.replan
        assert replan is not None
        artifact_ids: list[str] = [
            entry.artifact_id
            for entry in state["session"].session_working_set
            if entry.artifact_id in state["artifacts"]
        ]
        for artifact_id in replan.preserve_artifact_ids:
            if artifact_id in state["artifacts"] and artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)
        return artifact_ids

    def _build_replan_instruction(self, state: RuntimeState) -> str:
        decision = state["decision"]
        replan = decision.replan
        assert replan is not None
        return (
            f"Replan mode: {replan.mode}. "
            f"Reason: {replan.reason}. "
            f"Keep the passed tasks as the retained prefix and generate only the new suffix tasks."
        )

    def _mark_old_plan_tasks_for_replan(self, state: RuntimeState, old_plan_id: str) -> None:
        decision = state["decision"]
        replan = decision.replan
        assert replan is not None
        old_plan = state["plans"][old_plan_id]
        target_task_id = decision.task_id
        preserved = set(self._build_retained_prefix_task_ids(state, old_plan_id))

        if target_task_id and target_task_id in state["session"].task_states:
            state["session"].task_states[target_task_id].status = TaskStatus.REPLANNED

        for task_id in old_plan.task_ids:
            if task_id == target_task_id:
                continue
            task_state = state["session"].task_states[task_id]
            if task_state.status == TaskStatus.PASSED:
                continue
            if task_id in preserved:
                continue
            if task_state.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_EVALUATION}:
                task_state.status = TaskStatus.ABANDONED

    def _materialize_tasks_from_specs(
        self,
        plan_id: str,
        task_specs: list[PlanTaskSpec | dict[str, Any]],
    ) -> list[Task]:
        tasks: list[Task] = []
        for spec in task_specs:
            if isinstance(spec, dict):
                spec = PlanTaskSpec.model_validate(spec)
            tasks.append(
                Task(
                    id=spec.id,
                    plan_id=plan_id,
                    type=spec.type,
                    instruction=spec.instruction,
                    input_artifact_ids=list(spec.input_artifact_ids),
                    depends_on=list(spec.depends_on),
                    acceptance_criteria=list(spec.acceptance_criteria),
                )
            )
        return tasks

    def _attach_instruction_artifact(self, state: RuntimeState, task: Task) -> None:
        instruction_artifact = InstructionArtifact(
            id=f"art_instruction_{task.id}_001",
            summary=task.instruction,
            payload={"instruction_text": task.instruction},
            source_ids=list(task.input_artifact_ids),
            created_by="plan_agent",
            role="task_instruction",
            scope="task",
        )
        register_artifact_in_session_pool(state, instruction_artifact)
        append_artifact_ref_message(state, instruction_artifact, task_id=task.id)
        state["session"].task_states[task.id].task_artifact_ids.append(instruction_artifact.id)

    def _append_plan_messages(self, state: RuntimeState, plan_obj: Plan, tasks: list[Task]) -> None:
        append_message(
            state,
            role="assistant",
            content=[
                PlanBlock(
                    plan_id=plan_obj.id,
                    instruction=plan_obj.instruction,
                    task_ids=list(plan_obj.task_ids),
                    input_artifact_ids=list(plan_obj.input_artifact_ids),
                    understanding_artifact_ids=list(plan_obj.understanding_artifact_ids),
                )
            ],
        )
        for task in tasks:
            append_message(
                state,
                role="assistant",
                task_id=task.id,
                content=[
                    TaskBlock(
                        task_id=task.id,
                        plan_id=task.plan_id,
                        task_type=task.type,
                        instruction=task.instruction,
                        input_artifact_ids=list(task.input_artifact_ids),
                        depends_on=list(task.depends_on),
                        acceptance_criteria=list(task.acceptance_criteria),
                    )
                ],
            )

    def _next_plan_id(self, state: RuntimeState) -> str:
        existing_nums = [
            int(plan_id.split("_")[1])
            for plan_id in state.get("plans", {})
            if plan_id.startswith("plan_") and plan_id.split("_")[1].isdigit()
        ]
        next_num = max(existing_nums, default=0) + 1
        return f"plan_{next_num:03d}"

    def _allocate_task_id_block(self, state: RuntimeState, count: int) -> list[str]:
        existing_nums = [
            int(task_id.split("_")[1])
            for task_id in state.get("tasks", {})
            if task_id.startswith("task_") and task_id.split("_")[1].isdigit()
        ]
        start_num = max(existing_nums, default=0) + 1
        return [f"task_{start_num + index:03d}" for index in range(count)]

    def _use_llm(self, state: RuntimeState) -> bool:
        if not state["input"].get("use_llm", False):
            return False
        try:
            config = load_llm_config()
        except Exception:
            return False
        return bool(config.api_key and config.base_url and config.model_name)

    def _build_plan_with_llm(self, state: RuntimeState) -> PlanLLMOutput:
        understanding_summaries = [
            artifact.payload
            for artifact in state["artifacts"].values()
            if artifact.kind == "understanding"
        ]
        decision = state.get("decision")
        retained_prefix_plan = ""
        available_new_task_ids = ""
        available_artifacts = ""
        replan_context = ""
        if decision is not None and decision.replan is not None:
            old_plan_id = decision.plan_id or state["session"].current_plan_id
            retained_prefix_task_ids = self._build_retained_prefix_task_ids(state, old_plan_id)
            retained_tasks = [
                state["tasks"][task_id].model_dump()
                | {"status": state["session"].task_states[task_id].status}
                for task_id in retained_prefix_task_ids
            ]
            available_new_task_ids = str(self._allocate_task_id_block(state, 4))
            replan_context = (
                f"Replan mode: {decision.replan.mode}\n"
                f"Replan reason: {decision.replan.reason}\n"
                f"Directly replaced task id: {decision.task_id}\n"
                f"Decision issues: {decision.issues}\n"
                f"Source execution outcome: {decision.source_execution_outcome}\n"
                f"Preserve artifact ids: {decision.replan.preserve_artifact_ids}\n"
            )
            retained_prefix_plan = (
                f"Retained prefix task ids: {retained_prefix_task_ids}\n"
                f"Retained prefix tasks: {retained_tasks}\n"
                f"Available new task ids: {available_new_task_ids}\n"
            )
            available_artifacts = (
                "Available artifacts for planning:\n"
                f"{self._render_replan_artifact_catalog(state)}\n"
            )
        return invoke_structured_llm(
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_prompt=build_plan_user_prompt(
                instruction_text=state["input"]["instruction_text"],
                replan_context=replan_context,
                retained_prefix_plan=retained_prefix_plan,
                available_artifacts=available_artifacts,
                image_artifact_ids=[
                    entry.artifact_id
                    for entry in state["session"].session_working_set
                    if entry.artifact_id in state["artifacts"]
                    and state["artifacts"][entry.artifact_id].kind == "image"
                ],
                understanding_summaries=understanding_summaries,
            ),
            output_schema=PlanLLMOutput,
        )

    def _validate_llm_plan_output(
        self,
        state: RuntimeState,
        llm_plan: PlanLLMOutput,
        *,
        preserved_task_ids: list[str],
    ) -> bool:
        instruction = llm_plan.plan_instruction.strip()
        if not instruction:
            return False
        if instruction.startswith("[") or instruction.startswith("{"):
            return False
        lower_instruction = instruction.lower()
        if "\"type\"" in lower_instruction or "\"instruction\"" in lower_instruction:
            return False

        if not llm_plan.tasks:
            return False

        task_ids = [task.id for task in llm_plan.tasks]
        if len(task_ids) != len(set(task_ids)):
            return False
        existing_task_ids = set(state.get("tasks", {}).keys())
        for task_id in task_ids:
            if task_id in existing_task_ids:
                return False
            if not task_id.startswith("task_"):
                return False
        for task in llm_plan.tasks:
            if task.id in preserved_task_ids:
                return False
            if not task.type or not task.instruction or not task.acceptance_criteria:
                return False
            if preserved_task_ids:
                available_artifact_ids = set(self._build_replan_input_artifact_ids(state))
                for artifact_id in task.input_artifact_ids:
                    if artifact_id not in available_artifact_ids:
                        return False
            for dep_id in task.depends_on:
                if dep_id not in preserved_task_ids and dep_id not in task_ids:
                    return False
        return True

    def _render_replan_artifact_catalog(self, state: RuntimeState) -> str:
        artifact_ids = self._build_replan_input_artifact_ids(state)
        lines: list[str] = []
        for artifact_id in artifact_ids:
            artifact = state["artifacts"].get(artifact_id)
            if artifact is None:
                continue
            summary = artifact.summary
            if summary is None and artifact.kind == "understanding":
                summary = artifact.payload.get("summary")
            lines.append(
                f"- {artifact_id} | kind={artifact.kind} | summary={summary or '(no summary)'}"
            )
        return "\n".join(lines) or "(no available artifacts)"

    def _build_replan_fallback_tasks(self, state: RuntimeState) -> list[PlanTaskSpec]:
        decision = state["decision"]
        replan = decision.replan
        assert replan is not None
        new_ids = self._allocate_task_id_block(state, 2)
        if replan.mode == ReplanMode.SPLIT_TASK:
            return [
                PlanTaskSpec(
                    id=new_ids[0],
                    type="prepare_refined_inputs",
                    instruction="Prepare a narrower intermediate result for the failed task before attempting the final edit again.",
                    input_artifact_ids=[],
                    depends_on=[],
                    acceptance_criteria=[
                        "intermediate inputs are cleaner and easier to edit",
                    ],
                ),
                PlanTaskSpec(
                    id=new_ids[1],
                    type="retry_split_task_result",
                    instruction="Use the refined intermediate result to complete the original failed task goal.",
                    input_artifact_ids=[],
                    depends_on=[new_ids[0]],
                    acceptance_criteria=[
                        "the split retry path completes the original task goal",
                    ],
                ),
            ]
        return [
            PlanTaskSpec(
                id=new_ids[0],
                type="reroute_stabilize_subject",
                instruction="Stabilize the most important preserved result before continuing the rest of the route.",
                input_artifact_ids=[],
                depends_on=[],
                acceptance_criteria=[
                    "the preserved result remains stable",
                ],
            ),
            PlanTaskSpec(
                id=new_ids[1],
                type="reroute_finish_route",
                instruction="Complete the remaining goal using the preserved outputs and the new safer route.",
                input_artifact_ids=[],
                depends_on=[new_ids[0]],
                acceptance_criteria=[
                    "the rerouted path completes the requested goal",
                ],
            ),
        ]

    def _build_fallback_tasks(self, state: RuntimeState) -> list[PlanTaskSpec]:
        image_ids = list(state["session"].artifact_index.by_type.get("image", []))
        if len(image_ids) >= 4:
            top_id, skirt_id, face_id, background_id = image_ids[:4]
        else:
            top_id, skirt_id, face_id, background_id = (image_ids + [""] * 4)[:4]
        new_ids = self._allocate_task_id_block(state, 2)
        return [
            PlanTaskSpec(
                id=new_ids[0],
                type="compose_subject",
                instruction="Combine the face, top, and skirt into one subject reference.",
                input_artifact_ids=[face_id, top_id, skirt_id],
                depends_on=[],
                acceptance_criteria=[
                    "subject includes the provided face",
                    "subject wears the provided top and skirt",
                ],
            ),
            PlanTaskSpec(
                id=new_ids[1],
                type="place_subject_in_background",
                instruction="Place the composed subject into the provided background for the final photo.",
                input_artifact_ids=[background_id],
                depends_on=[new_ids[0]],
                acceptance_criteria=[
                    "subject appears in the provided background",
                    "composition looks like a photo setup",
                ],
            ),
        ]
