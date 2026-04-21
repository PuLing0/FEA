"""Minimal planning agent."""

from __future__ import annotations

from llm import invoke_structured_llm, load_llm_config
from runtime.scheduler import select_next_runnable_task
from runtime.state import RuntimeState
from schema import (
    InstructionArtifact,
    Plan,
    PlanLLMOutput,
    SessionPhase,
    Task,
    TaskState,
    TaskStatus,
    ToolName,
)



class PlanAgent:
    """Creates a minimal plan and the first runnable task."""

    def run(self, state: RuntimeState) -> RuntimeState:
        session = state["session"]
        session.phase = SessionPhase.EXECUTING
        llm_plan = self._build_plan_with_llm(state) if self._use_llm(state) else None
        task_specs = llm_plan.tasks if llm_plan and llm_plan.tasks else self._build_fallback_tasks(state)

        plan_obj = Plan(
            id="plan_001",
            instruction=(llm_plan.plan_instruction if llm_plan else state["input"]["instruction_text"]),
            task_ids=[f"task_{index + 1:03d}" for index in range(len(task_specs))],
            input_artifact_ids=[
                artifact_id
                for artifact_id in state["session"].artifact_index.by_type.get("image", [])
            ],
            understanding_artifact_ids=[],
        )
        tasks: list[Task] = []
        for index, spec in enumerate(task_specs, start=1):
            task = Task(
                id=f"task_{index:03d}",
                plan_id=plan_obj.id,
                type=spec["type"],
                instruction=spec["instruction"],
                input_artifact_ids=spec["input_artifact_ids"],
                target_artifact_ids=spec["target_artifact_ids"],
                depends_on=spec.get("depends_on", []),
                acceptance_criteria=spec["acceptance_criteria"],
            )
            tasks.append(task)

        state["plans"][plan_obj.id] = plan_obj
        for task in tasks:
            state["tasks"][task.id] = task

        session.current_plan_id = plan_obj.id
        session.current_task_id = None
        for task in tasks:
            session.task_states[task.id] = TaskState(
                task_id=task.id,
                status=TaskStatus.PENDING,
            )
            instruction_artifact = InstructionArtifact(
                id=f"art_instruction_{task.id}_001",
                payload={"instruction_text": task.instruction},
                source_ids=list(task.input_artifact_ids),
                created_by="plan_agent",
                scope="task",
            )
            state["artifacts"][instruction_artifact.id] = instruction_artifact
            session.task_states[task.id].task_artifact_ids.append(instruction_artifact.id)

        first_runnable = select_next_runnable_task(state)
        session.current_task_id = first_runnable
        if first_runnable is not None:
            session.task_states[first_runnable].status = TaskStatus.RUNNING

        state["session"] = session
        return state

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
        return invoke_structured_llm(
            system_prompt=(
                "You are a planning agent for image editing. "
                "Return a task todo list for composing a final fashion image from multiple inputs."
            ),
            user_prompt=(
                f"Instruction: {state['input']['instruction_text']}\n"
                f"Image artifact ids: {state['session'].artifact_index.by_type.get('image', [])}\n"
                f"Understanding summaries: {understanding_summaries}\n"
                "Return a short task todo list. Each task needs type, instruction, input_artifact_ids, "
                "target_artifact_ids, depends_on, acceptance_criteria. "
                "If a task has no forward dependency, specify its input_artifact_ids at planning time. "
                "If a task depends on an earlier task, keep only its static inputs here and express the dependency with depends_on."
            ),
            output_schema=PlanLLMOutput,
        )

    def _build_fallback_tasks(self, state: RuntimeState) -> list[dict[str, object]]:
        image_ids = list(state["session"].artifact_index.by_type.get("image", []))
        if len(image_ids) >= 4:
            top_id, skirt_id, face_id, background_id = image_ids[:4]
        else:
            top_id, skirt_id, face_id, background_id = (image_ids + [""] * 4)[:4]

        return [
            {
                "type": "compose_subject",
                "instruction": "Combine the face, top, and skirt into one subject reference.",
                "input_artifact_ids": [face_id, top_id, skirt_id],
                "target_artifact_ids": [face_id],
                "depends_on": [],
                "acceptance_criteria": [
                    "subject includes the provided face",
                    "subject wears the provided top and skirt",
                ],
            },
            {
                "type": "place_subject_in_background",
                "instruction": "Place the composed subject into the provided background for the final photo.",
                "input_artifact_ids": [background_id],
                "target_artifact_ids": [background_id],
                "depends_on": ["task_001"],
                "acceptance_criteria": [
                    "subject appears in the provided background",
                    "composition looks like a photo setup",
                ],
            },
        ]
