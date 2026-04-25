"""Stateless pre-execution input preparation for tasks."""

from __future__ import annotations

from typing import Any

from llm import invoke_llm, invoke_structured_llm, load_llm_config
from runtime.prompts import (
    CANDIDATE_UNDERSTAND_QUESTION_TEMPLATE,
    INPUT_SELECTOR_SYSTEM_PROMPT,
    INPUT_THINKING_SYSTEM_PROMPT,
    INPUT_VALIDATION_SYSTEM_PROMPT,
    build_input_selector_user_prompt,
    build_input_thinking_user_prompt,
    build_input_validation_user_prompt,
)
from runtime.state import RuntimeState
from schema import ArtifactKind, StrictModel, ToolName, UnderstandArgs
from tools.registry import build_default_tool_registry


TOOL_REGISTRY = build_default_tool_registry()


class TaskInputSelectionOutput(StrictModel):
    """Structured output for stateless task input selection."""

    selected_artifact_ids: list[str]


def prepare_task_inputs(state: RuntimeState, task_id: str) -> TaskInputSelectionOutput:
    """Run the full pre-execution input preparation pipeline."""

    task_state = state["session"].task_states[task_id]
    if task_state.retry_input_artifact_ids:
        task_state.resolved_input_artifact_ids = list(task_state.retry_input_artifact_ids)
        task_state.input_selection_reasoning = "resolved from evaluator retry context"
        task_state.input_validation_summary = "using compressed retry context from evaluator"
        return TaskInputSelectionOutput(
            selected_artifact_ids=list(task_state.retry_input_artifact_ids)
        )

    if task_state.resolved_input_artifact_ids:
        return TaskInputSelectionOutput(
            selected_artifact_ids=list(task_state.resolved_input_artifact_ids)
        )

    candidate_image_ids = build_candidate_image_pool(state)
    ensure_understanding_for_images(state, candidate_image_ids, task_id=task_id)
    candidates_text = build_candidate_images_text(state, candidate_image_ids)
    thinking = build_input_thinking(state, task_id, candidates_text)
    selection = select_input_artifacts(state, task_id, candidates_text, thinking)
    validation_summary = validate_selected_inputs(state, task_id, selection, candidates_text)
    task_state.input_selection_reasoning = thinking
    task_state.input_validation_summary = validation_summary
    task_state.resolved_input_artifact_ids = list(selection.selected_artifact_ids)
    return selection


def build_candidate_image_pool(state: RuntimeState) -> list[str]:
    """Return the current visible image artifact ids.

    Visible candidates come from:
    - session-level retained image pool
    - current plan-level visible image pool
    - final outputs from dependency tasks
    - current task-local image artifacts (for later loop iterations)
    """

    session = state["session"]
    current_task_id = session.current_task_id
    candidates: list[str] = []

    if session.artifact_index is not None:
        for artifact_id in session.artifact_index.by_type.get("image", []):
            _append_image_candidate(state, candidates, artifact_id)

    plans = state.get("plans", {})
    if session.current_plan_id is not None and session.current_plan_id in plans:
        current_plan = plans[session.current_plan_id]
        for artifact_id in current_plan.input_artifact_ids:
            _append_image_candidate(state, candidates, artifact_id)

    if current_task_id is None:
        return candidates

    task = state["tasks"][current_task_id]
    for dep_id in task.depends_on:
        dep_final = session.task_states[dep_id].final_artifact_id
        if dep_final:
            _append_image_candidate(state, candidates, dep_final)

    for artifact_id in session.task_states[current_task_id].task_artifact_ids:
        _append_image_candidate(state, candidates, artifact_id)

    return candidates


def _append_image_candidate(
    state: RuntimeState,
    candidates: list[str],
    artifact_id: str,
) -> None:
    artifact = state["artifacts"].get(artifact_id)
    if artifact is None:
        return
    if artifact.kind != ArtifactKind.IMAGE:
        return
    if artifact_id in candidates:
        return
    candidates.append(artifact_id)


def ensure_understanding_for_images(state: RuntimeState, image_ids: list[str], *, task_id: str) -> None:
    """Ensure every candidate image has an understanding artifact."""

    understood_ids = {
        artifact.payload["image_ref"]
        for artifact in state["artifacts"].values()
        if artifact.kind == ArtifactKind.UNDERSTANDING and artifact.payload.get("image_ref")
    }

    for image_id in image_ids:
        if image_id in understood_ids:
            continue
        understand_execution = TOOL_REGISTRY.get(ToolName.UNDERSTAND).run(
            state,
            task_id=task_id,
            loop_index=0,
            args=UnderstandArgs(
                image_ref=image_id,
                question=CANDIDATE_UNDERSTAND_QUESTION_TEMPLATE.format(image_id=image_id),
            ),
        )
        state["operations"].append(understand_execution.invocation)
        for artifact in understand_execution.artifacts:
            state["artifacts"][artifact.id] = artifact


def build_candidate_images_text(state: RuntimeState, image_ids: list[str]) -> str:
    """Convert candidate image artifacts into compact text lines.

    Each line keeps only:
    - artifact_id
    - artifact summary
    - source
    """

    lines: list[str] = []
    for image_id in image_ids:
        artifact = state["artifacts"][image_id]
        source_label = _resolve_source_label(image_id, artifact.payload)
        summary = artifact.summary or ""
        if source_label is None:
            lines.append(f"[{image_id}] {summary} | 原始输入图片")
        else:
            lines.append(f"[{image_id}] {summary} | source={source_label}")
    return "\n".join(lines)


def _resolve_source_label(image_id: str, payload: dict[str, Any]) -> str | None:
    """Resolve a compact source label for candidate image text."""

    if payload.get("source"):
        return str(payload["source"])
    if image_id.startswith("art_img_input_"):
        return None
    if image_id.startswith("art_image_"):
        return "generated_result"
    return "unknown"


def build_input_thinking(state: RuntimeState, task_id: str, candidates_text: str) -> str:
    """Produce pre-execution thinking for image input selection."""

    task = state["tasks"][task_id]
    if _use_llm(state):
        response = invoke_llm(
            system_prompt=INPUT_THINKING_SYSTEM_PROMPT,
            user_prompt=build_input_thinking_user_prompt(
                task=task,
                candidates_text=candidates_text,
            ),
        )
        return str(response.content)

    if task.depends_on:
        return (
            "This task depends on a previous task result. "
            "Use the static background-style image plus the dependency output image if available."
        )
    return "This task has no dependency. Use only the static input images needed for this task."


def select_input_artifacts(
    state: RuntimeState,
    task_id: str,
    candidates_text: str,
    thinking: str,
) -> TaskInputSelectionOutput:
    """Select task input images using a stateless selector call."""

    task = state["tasks"][task_id]

    if _use_llm(state):
        return invoke_structured_llm(
            system_prompt=INPUT_SELECTOR_SYSTEM_PROMPT,
            user_prompt=build_input_selector_user_prompt(
                task=task,
                thinking=thinking,
                candidates_text=candidates_text,
            ),
            output_schema=TaskInputSelectionOutput,
        )

    selected = [
        artifact_id
        for artifact_id in task.input_artifact_ids
        if artifact_id in state["artifacts"]
        and state["artifacts"][artifact_id].kind == ArtifactKind.IMAGE
    ]
    if task.depends_on:
        for dep_id in task.depends_on:
            dep_outputs = state["session"].task_states[dep_id].latest_artifact_ids
            for artifact_id in dep_outputs:
                if artifact_id not in selected:
                    selected.append(artifact_id)
    if not selected:
        selected = list(build_candidate_image_pool(state))
    return TaskInputSelectionOutput(
        selected_artifact_ids=selected,
    )


def validate_selected_inputs(
    state: RuntimeState,
    task_id: str,
    selection: TaskInputSelectionOutput,
    candidates_text: str,
) -> str:
    """Validate the selected images before entering the execute loop."""

    task = state["tasks"][task_id]

    if _use_llm(state):
        response = invoke_llm(
            system_prompt=INPUT_VALIDATION_SYSTEM_PROMPT,
            user_prompt=build_input_validation_user_prompt(
                task=task,
                selected_artifact_ids=selection.selected_artifact_ids,
                input_thinking=state["session"].task_states[task_id].input_selection_reasoning,
                candidates_text=candidates_text,
            ),
        )
        return str(response.content)

    return "selected inputs look acceptable for this task."


def _use_llm(state: RuntimeState) -> bool:
    if not state["input"].get("use_llm", False):
        return False
    try:
        config = load_llm_config()
    except Exception:
        return False
    return bool(config.api_key and config.base_url and config.model_name)
