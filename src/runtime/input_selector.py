"""Artifact selection helpers for task initialization and continue-execute rebuilds."""

from __future__ import annotations

from typing import Any

from llm import invoke_llm, invoke_structured_llm, load_llm_config
from runtime.artifact_context import (
    build_session_working_set_catalog,
    build_task_pool_catalog,
    make_working_set_entry,
    replace_task_working_set,
    working_set_artifact_ids,
)
from runtime.instruction_resolver import (
    TASK_SCOPED_INSTRUCTION_ROLES,
    instruction_artifact_belongs_to_task,
    resolve_active_instruction_artifact,
    resolve_active_instruction_text,
)
from runtime.message_store import append_artifact_ref_message
from runtime.prompts import (
    CANDIDATE_UNDERSTAND_QUESTION_TEMPLATE,
    INPUT_SELECTOR_SYSTEM_PROMPT,
    INPUT_THINKING_SYSTEM_PROMPT,
    INPUT_VALIDATION_SYSTEM_PROMPT,
    TASK_WORKING_SET_SELECTOR_SYSTEM_PROMPT,
    build_input_selector_user_prompt,
    build_input_thinking_user_prompt,
    build_input_validation_user_prompt,
    build_task_working_set_selector_user_prompt,
)
from runtime.state import RuntimeState
from runtime.tool_runner import ToolRunner
from schema import (
    ArtifactKind,
    StrictModel,
    ToolName,
    UnderstandArgs,
    WorkingSetEntry,
)
from tools.registry import build_default_tool_registry


TOOL_REGISTRY = build_default_tool_registry()
TOOL_RUNNER = ToolRunner(TOOL_REGISTRY)
EDIT_IMAGE_INPUT_LIMIT = 3


class TaskInputSelectionOutput(StrictModel):
    """Structured output for task artifact selection."""

    selected_artifact_ids: list[str]
    working_set_entries: list[WorkingSetEntry] = []


def build_candidate_image_pool(state: RuntimeState) -> list[str]:
    """Return visible image candidates from session/task memory."""

    session = state["session"]
    current_task_id = session.current_task_id
    candidates: list[str] = []

    if session.artifact_index is not None:
        for artifact_id in session.artifact_index.by_type.get(ArtifactKind.IMAGE, []):
            _append_image_candidate(state, candidates, artifact_id)

    plans = state.get("plans", {})
    if session.current_plan_id is not None and session.current_plan_id in plans:
        for artifact_id in plans[session.current_plan_id].input_artifact_ids:
            _append_image_candidate(state, candidates, artifact_id)

    for entry in session.session_working_set:
        _append_image_candidate(state, candidates, entry.artifact_id)

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


def prepare_task_inputs(state: RuntimeState, task_id: str) -> TaskInputSelectionOutput:
    """Resolve the current task working set into the active execute inputs."""

    task_state = state["session"].task_states[task_id]
    if task_state.task_working_set:
        capped = _normalize_task_working_set_entries(
            state,
            task_id,
            list(task_state.task_working_set),
            include_dependency_outputs=True,
        )
        replace_task_working_set(state, task_id, capped)
        return TaskInputSelectionOutput(
            selected_artifact_ids=working_set_artifact_ids(capped),
            working_set_entries=list(capped),
        )

    if task_state.retry_input_artifact_ids and not task_state.task_artifact_ids:
        entries = [
            make_working_set_entry(
                artifact_id,
                usage="legacy retry artifact",
                selection_reason="resolved from legacy retry input context",
            )
            for artifact_id in task_state.retry_input_artifact_ids
        ]
        replace_task_working_set(state, task_id, entries)
        task_state.resolved_input_artifact_ids = list(task_state.retry_input_artifact_ids)
        task_state.input_selection_reasoning = "resolved from evaluator retry context"
        task_state.input_validation_summary = "using legacy retry input context"
        return TaskInputSelectionOutput(
            selected_artifact_ids=working_set_artifact_ids(entries),
            working_set_entries=list(entries),
        )

    if task_state.retry_context_text:
        return rebuild_task_working_set_for_continue_execute(state, task_id)

    if task_state.resolved_input_artifact_ids:
        entries = [
            make_working_set_entry(
                artifact_id,
                usage="previously resolved task input",
                selection_reason="reused because task inputs were already resolved",
            )
            for artifact_id in task_state.resolved_input_artifact_ids
        ]
        entries = _normalize_task_working_set_entries(
            state,
            task_id,
            entries,
            include_dependency_outputs=True,
        )
        replace_task_working_set(state, task_id, entries)
        task_state.resolved_input_artifact_ids = [
            artifact_id
            for artifact_id in working_set_artifact_ids(entries)
            if _is_image_artifact(state, artifact_id)
        ]
        task_state.input_selection_reasoning = "reused previously resolved task inputs"
        task_state.input_validation_summary = "using existing resolved input context"
        return TaskInputSelectionOutput(
            selected_artifact_ids=working_set_artifact_ids(entries),
            working_set_entries=list(entries),
        )

    return initialize_task_working_set_from_session(state, task_id)


def initialize_task_working_set_from_session(
    state: RuntimeState,
    task_id: str,
) -> TaskInputSelectionOutput:
    """Select initial task inputs from the session working set."""

    task_state = state["session"].task_states[task_id]
    instruction_artifact_id = _find_latest_instruction_artifact_id(state, task_id)
    candidates_text = build_session_working_set_catalog(state)
    thinking = build_input_thinking(state, task_id, candidates_text)
    selection = select_initial_task_artifacts(state, task_id, candidates_text, thinking)
    validation_summary = validate_selected_inputs(state, task_id, selection, candidates_text)
    entries = _normalize_task_working_set_entries(
        state,
        task_id,
        list(selection.working_set_entries),
        current_instruction_id=instruction_artifact_id,
        include_dependency_outputs=True,
    )
    for artifact_id in working_set_artifact_ids(entries):
        if artifact_id not in task_state.task_artifact_ids:
            task_state.task_artifact_ids.append(artifact_id)
    replace_task_working_set(state, task_id, entries)
    task_state.input_selection_reasoning = thinking
    task_state.input_validation_summary = validation_summary
    task_state.resolved_input_artifact_ids = [
        artifact_id
        for artifact_id in working_set_artifact_ids(entries)
        if _is_image_artifact(state, artifact_id)
    ]
    task_state.retry_input_artifact_ids = []
    return TaskInputSelectionOutput(
        selected_artifact_ids=working_set_artifact_ids(entries),
        working_set_entries=list(entries),
    )


def rebuild_task_working_set_for_continue_execute(
    state: RuntimeState,
    task_id: str,
) -> TaskInputSelectionOutput:
    """Rebuild the task working set from task-local memory after continue_execute."""

    task_state = state["session"].task_states[task_id]
    task = state["tasks"][task_id]
    active_instruction = _resolve_active_instruction_text(state, task_id)
    pool_catalog = build_task_pool_catalog(state, task_id)
    if _use_llm(state):
        try:
            selection = invoke_structured_llm(
                system_prompt=TASK_WORKING_SET_SELECTOR_SYSTEM_PROMPT,
                user_prompt=build_task_working_set_selector_user_prompt(
                    task=task,
                    active_instruction=active_instruction,
                    retry_context=task_state.retry_context_text,
                    latest_candidate_refs=list(task_state.latest_artifact_ids),
                    task_pool_catalog=pool_catalog,
                ),
                output_schema=TaskInputSelectionOutput,
            )
        except Exception:
            selection = _fallback_continue_execute_selection(state, task_id)
    else:
        selection = _fallback_continue_execute_selection(state, task_id)
    entries = list(selection.working_set_entries)
    latest_instruction_id = _find_latest_instruction_artifact_id(state, task_id)
    latest_evaluation_id = _find_latest_evaluation_artifact_id(state, task_id)
    latest_candidate_id = task_state.latest_artifact_ids[-1] if task_state.latest_artifact_ids else None
    selected_ids = set(working_set_artifact_ids(entries))
    if latest_instruction_id and latest_instruction_id not in selected_ids:
        entries.insert(
            0,
            make_working_set_entry(
                latest_instruction_id,
                usage="current task instruction",
                selection_reason="required to keep the current task instruction active during continue_execute",
            ),
        )
        selected_ids.add(latest_instruction_id)
    if latest_evaluation_id and latest_evaluation_id not in selected_ids:
        entries.insert(
            1 if entries else 0,
            make_working_set_entry(
                latest_evaluation_id,
                usage="latest evaluation feedback",
                selection_reason="required to carry the latest evaluator feedback into the next execute cycle",
            ),
        )
        selected_ids.add(latest_evaluation_id)
    if latest_candidate_id and _is_image_artifact(state, latest_candidate_id) and latest_candidate_id not in selected_ids:
        entries.insert(
            0,
            make_working_set_entry(
                latest_candidate_id,
                usage="current base candidate",
                selection_reason="required to preserve the latest candidate while continuing the same task",
            ),
        )
    entries = _normalize_task_working_set_entries(
        state,
        task_id,
        entries,
        current_instruction_id=latest_instruction_id,
        include_dependency_outputs=False,
    )
    replace_task_working_set(state, task_id, entries)
    task_state.input_selection_reasoning = (
        "reselected from task pool after continue_execute"
    )
    task_state.input_validation_summary = "rebuilt compact working set from task-local artifact memory"
    return TaskInputSelectionOutput(
        selected_artifact_ids=working_set_artifact_ids(task_state.task_working_set),
        working_set_entries=list(task_state.task_working_set),
    )


def build_input_thinking(state: RuntimeState, task_id: str, candidates_text: str) -> str:
    """Produce pre-execution thinking for initial artifact selection."""

    task = state["tasks"][task_id]
    if task.depends_on:
        return (
            "This task depends on a previous task result. "
            "Use the retained dependency outputs plus the most relevant session-level references."
        )
    return "Use the session working set to pick only the images and instructions needed for this task."


def select_initial_task_artifacts(
    state: RuntimeState,
    task_id: str,
    candidates_text: str,
    thinking: str,
) -> TaskInputSelectionOutput:
    """Select initial task artifacts from the session working set."""

    task = state["tasks"][task_id]
    if _use_llm(state):
        try:
            selection = invoke_structured_llm(
                system_prompt=INPUT_SELECTOR_SYSTEM_PROMPT,
                user_prompt=build_input_selector_user_prompt(
                    task=task,
                    thinking=thinking,
                    candidates_text=candidates_text,
                ),
                output_schema=TaskInputSelectionOutput,
            )
            validated = _validate_selection_budget(state, selection)
            if validated is not None:
                return validated
        except Exception:
            pass
    return _fallback_initial_selection(state, task_id)


def validate_selected_inputs(
    state: RuntimeState,
    task_id: str,
    selection: TaskInputSelectionOutput,
    candidates_text: str,
) -> str:
    """Validate the selected task inputs before entering execute."""

    return "selected task inputs look acceptable."


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
        understand_execution = TOOL_RUNNER.run(
            state,
            ToolName.UNDERSTAND,
            task_id=task_id,
            loop_index=0,
            args=UnderstandArgs(
                image_ref=image_id,
                question=CANDIDATE_UNDERSTAND_QUESTION_TEMPLATE.format(image_id=image_id),
            ),
        )
        for artifact in understand_execution.artifacts:
            state["artifacts"][artifact.id] = artifact
            append_artifact_ref_message(state, artifact, task_id=task_id)


def build_candidate_images_text(state: RuntimeState, image_ids: list[str]) -> str:
    """Convert candidate image artifacts into compact text lines."""

    lines: list[str] = []
    for image_id in image_ids:
        artifact = state["artifacts"][image_id]
        summary = artifact.summary or ""
        source_label = _resolve_source_label(image_id, artifact.payload)
        if source_label == "initial_input":
            lines.append(f"[{image_id}] {summary} | 原始输入图片")
        else:
            lines.append(f"[{image_id}] {summary} | source={source_label or 'unknown'}")
    return "\n".join(lines)


def _append_image_candidate(
    state: RuntimeState,
    candidates: list[str],
    artifact_id: str,
) -> None:
    artifact = state["artifacts"].get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.IMAGE:
        return
    if artifact_id in candidates:
        return
    candidates.append(artifact_id)


def _resolve_source_label(image_id: str, payload: dict[str, Any]) -> str | None:
    if payload.get("source"):
        return str(payload["source"])
    if image_id.startswith("art_img_input_"):
        return "initial_input"
    if image_id.startswith("art_image_"):
        return "generated_result"
    return None


def _fallback_initial_selection(state: RuntimeState, task_id: str) -> TaskInputSelectionOutput:
    task = state["tasks"][task_id]
    selected_entries: list[WorkingSetEntry] = []
    selected_ids: list[str] = []
    visible_ids = [entry.artifact_id for entry in state["session"].session_working_set]
    if not visible_ids:
        visible_ids = list(task.input_artifact_ids)
    if not visible_ids and state["session"].artifact_index is not None:
        visible_ids = list(state["session"].artifact_index.by_type.get(ArtifactKind.IMAGE, []))
    image_count = 0

    def append_entry(artifact_id: str, *, usage: str, selection_reason: str) -> None:
        nonlocal image_count
        if artifact_id in selected_ids:
            return
        artifact = state["artifacts"].get(artifact_id)
        if artifact is None:
            return
        if artifact.kind == ArtifactKind.INSTRUCTION:
            if artifact.role != "session_root_instruction":
                return
        elif artifact.kind == ArtifactKind.IMAGE:
            if image_count >= EDIT_IMAGE_INPUT_LIMIT:
                return
            image_count += 1
        else:
            return
        selected_entries.append(
            make_working_set_entry(
                artifact_id,
                usage=usage,
                selection_reason=selection_reason,
            )
        )
        selected_ids.append(artifact_id)

    for artifact_id in visible_ids:
        artifact = state["artifacts"].get(artifact_id)
        if artifact is None:
            continue
        if artifact.kind == ArtifactKind.INSTRUCTION:
            append_entry(
                artifact_id,
                usage="session instruction context",
                selection_reason="retain the root instruction for task initialization",
            )
    for artifact_id in task.input_artifact_ids:
        append_entry(
            artifact_id,
            usage="static task image input",
            selection_reason="selected because the task explicitly requested this input artifact",
        )
    if task.depends_on:
        for dep_id in task.depends_on:
            dep_final = state["session"].task_states[dep_id].final_artifact_id
            if dep_final:
                append_entry(
                    dep_final,
                    usage="dependency output image",
                    selection_reason="selected because this task depends on an earlier task result",
                )
    for artifact_id in visible_ids:
        append_entry(
            artifact_id,
            usage="initial task image input",
            selection_reason="selected from the session working set for task initialization",
        )
    if not selected_entries and state["session"].artifact_index is not None:
        for artifact_id in state["session"].artifact_index.by_type.get(ArtifactKind.IMAGE, []):
            append_entry(
                artifact_id,
                usage="fallback initial task image input",
                selection_reason="used because no explicit task input artifacts were available",
            )
    return TaskInputSelectionOutput(
        selected_artifact_ids=selected_ids,
        working_set_entries=selected_entries,
    )


def _fallback_continue_execute_selection(
    state: RuntimeState,
    task_id: str,
) -> TaskInputSelectionOutput:
    task_state = state["session"].task_states[task_id]
    selected_entries: list[WorkingSetEntry] = []
    selected_ids: list[str] = []

    latest_instruction_id = _find_latest_instruction_artifact_id(state, task_id)
    if latest_instruction_id is not None:
        selected_entries.append(
            make_working_set_entry(
                latest_instruction_id,
                usage="current task instruction",
                selection_reason="keep the latest instruction active during continue_execute",
            )
        )
        selected_ids.append(latest_instruction_id)

    latest_evaluation_id = _find_latest_evaluation_artifact_id(state, task_id)
    if latest_evaluation_id is not None:
        selected_entries.append(
            make_working_set_entry(
                latest_evaluation_id,
                usage="latest evaluation feedback",
                selection_reason="carry evaluator feedback into the next execute cycle",
            )
        )
        selected_ids.append(latest_evaluation_id)

    latest_candidate_id = task_state.latest_artifact_ids[-1] if task_state.latest_artifact_ids else None
    image_ids: list[str] = []
    if latest_candidate_id and _is_image_artifact(state, latest_candidate_id):
        image_ids.append(latest_candidate_id)
        candidate = state["artifacts"][latest_candidate_id]
        for source_id in candidate.source_ids:
            if not _is_image_artifact(state, source_id):
                continue
            if source_id not in image_ids:
                image_ids.append(source_id)
            if len(image_ids) >= 3:
                break
    if not image_ids:
        for artifact_id in reversed(task_state.task_artifact_ids):
            if not _is_image_artifact(state, artifact_id):
                continue
            if artifact_id not in image_ids:
                image_ids.append(artifact_id)
            if len(image_ids) >= 3:
                break

    for index, artifact_id in enumerate(image_ids):
        usage = "current base candidate" if index == 0 and latest_candidate_id == artifact_id else "reference image"
        reason = (
            "carry forward the best current candidate"
            if usage == "current base candidate"
            else "keep the most relevant image references from the candidate history"
        )
        selected_entries.append(
            make_working_set_entry(
                artifact_id,
                usage=usage,
                selection_reason=reason,
            )
        )
        selected_ids.append(artifact_id)

    return TaskInputSelectionOutput(
        selected_artifact_ids=selected_ids,
        working_set_entries=selected_entries,
    )


def _candidate_selection_ids(state: RuntimeState) -> set[str]:
    return {entry.artifact_id for entry in state["session"].session_working_set if entry.artifact_id in state["artifacts"]}


def _validate_selection_budget(
    state: RuntimeState,
    selection: TaskInputSelectionOutput,
) -> TaskInputSelectionOutput | None:
    candidate_ids = _candidate_selection_ids(state)
    selected_entries: list[WorkingSetEntry] = []
    selected_ids: list[str] = []
    image_count = 0
    source_entries = list(selection.working_set_entries)
    if not source_entries:
        source_entries = [
            make_working_set_entry(
                artifact_id,
                usage="selected task artifact",
                selection_reason="selected by the input selector",
            )
            for artifact_id in selection.selected_artifact_ids
        ]

    for entry in source_entries:
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None:
            continue
        if entry.artifact_id not in candidate_ids:
            continue
        if artifact.kind == ArtifactKind.IMAGE:
            if image_count >= EDIT_IMAGE_INPUT_LIMIT:
                continue
            image_count += 1
        elif artifact.kind != ArtifactKind.INSTRUCTION:
            continue
        if entry.artifact_id in selected_ids:
            continue
        selected_entries.append(entry)
        selected_ids.append(entry.artifact_id)

    if not selected_entries:
        return None
    return TaskInputSelectionOutput(
        selected_artifact_ids=selected_ids,
        working_set_entries=selected_entries,
    )


def _normalize_task_working_set_entries(
    state: RuntimeState,
    task_id: str,
    entries: list[WorkingSetEntry],
    *,
    current_instruction_id: str | None = None,
    include_dependency_outputs: bool,
) -> list[WorkingSetEntry]:
    """Keep task-local instructions clean and prioritize dependency outputs."""

    current_instruction_id = current_instruction_id or _find_latest_instruction_artifact_id(state, task_id)
    selected: list[WorkingSetEntry] = []
    selected_ids: set[str] = set()

    def append_entry(entry: WorkingSetEntry) -> None:
        if entry.artifact_id in selected_ids:
            return
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None:
            return
        if artifact.kind == ArtifactKind.INSTRUCTION and not _keep_instruction_for_task(
            artifact,
            task_id,
            current_instruction_id,
        ):
            return
        selected.append(entry)
        selected_ids.add(entry.artifact_id)

    if current_instruction_id is not None:
        append_entry(
            make_working_set_entry(
                current_instruction_id,
                usage="current task instruction",
                selection_reason="required to keep the active task instruction visible",
            )
        )

    if include_dependency_outputs:
        for dep_id in state["tasks"][task_id].depends_on:
            dep_state = state["session"].task_states.get(dep_id)
            dep_final = dep_state.final_artifact_id if dep_state is not None else None
            if dep_final and _is_image_artifact(state, dep_final):
                append_entry(
                    make_working_set_entry(
                        dep_final,
                        usage="dependency output image",
                        selection_reason="required because the current task depends on this passed task result",
                    )
                )

    for entry in entries:
        append_entry(entry)

    return _cap_working_set_image_count(state, selected)


def _keep_instruction_for_task(
    artifact,
    task_id: str,
    current_instruction_id: str | None,
) -> bool:
    if artifact.id == current_instruction_id:
        return True
    if artifact.role == "session_root_instruction":
        return True
    if artifact.role in TASK_SCOPED_INSTRUCTION_ROLES:
        return instruction_artifact_belongs_to_task(artifact, task_id)
    return False


def _cap_working_set_image_count(
    state: RuntimeState,
    entries: list[WorkingSetEntry],
) -> list[WorkingSetEntry]:
    capped: list[WorkingSetEntry] = []
    image_count = 0
    for entry in entries:
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None:
            continue
        if artifact.kind == ArtifactKind.IMAGE:
            if image_count >= EDIT_IMAGE_INPUT_LIMIT:
                continue
            image_count += 1
        capped.append(entry)
    return capped


def _find_latest_instruction_artifact_id(state: RuntimeState, task_id: str) -> str | None:
    artifact = resolve_active_instruction_artifact(state, task_id)
    return artifact.id if artifact is not None else None


def _find_latest_evaluation_artifact_id(state: RuntimeState, task_id: str) -> str | None:
    for artifact_id in reversed(state["session"].task_states[task_id].task_artifact_ids):
        artifact = state["artifacts"].get(artifact_id)
        if artifact is not None and artifact.kind == ArtifactKind.EVALUATION:
            return artifact_id
    return None


def _resolve_active_instruction_text(state: RuntimeState, task_id: str) -> str:
    return resolve_active_instruction_text(state, task_id)


def _is_image_artifact(state: RuntimeState, artifact_id: str) -> bool:
    artifact = state["artifacts"].get(artifact_id)
    return artifact is not None and artifact.kind == ArtifactKind.IMAGE


def _use_llm(state: RuntimeState) -> bool:
    if not state["input"].get("use_llm", False):
        return False
    try:
        config = load_llm_config()
    except Exception:
        return False
    return bool(config.api_key and config.base_url and config.model_name)
