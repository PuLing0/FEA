"""Helpers for resolving active instruction text for a task."""

from __future__ import annotations

from runtime.state import RuntimeState
from schema import InstructionArtifact

INSTRUCTION_ROLE_PRIORITY = (
    "rewritten_instruction",
    "task_instruction",
    "session_root_instruction",
)
TASK_SCOPED_INSTRUCTION_ROLES = {"rewritten_instruction", "task_instruction"}


def instruction_artifact_belongs_to_task(
    artifact: InstructionArtifact,
    task_id: str,
) -> bool:
    """Return whether a task-scoped instruction artifact belongs to task_id."""

    payload_task_id = artifact.payload.get("task_id") if isinstance(artifact.payload, dict) else None
    if isinstance(payload_task_id, str) and payload_task_id.strip():
        return payload_task_id.strip() == task_id
    return artifact.id.startswith(f"art_instruction_{task_id}_")


def resolve_active_instruction_artifact(
    state: RuntimeState,
    task_id: str,
) -> InstructionArtifact | None:
    """Return the active instruction artifact visible to the task.

    Preference order:
    1. latest rewritten_instruction
    2. latest task_instruction
    3. latest session_root_instruction
    """

    task_state = state["session"].task_states[task_id]
    latest_by_role: dict[str, InstructionArtifact] = {}
    latest_unclassified: InstructionArtifact | None = None
    for artifact_id in task_state.task_artifact_ids:
        artifact = state["artifacts"].get(artifact_id)
        if not isinstance(artifact, InstructionArtifact):
            continue
        if not artifact.get_instruction_text():
            continue
        if (
            artifact.role in TASK_SCOPED_INSTRUCTION_ROLES
            and not instruction_artifact_belongs_to_task(artifact, task_id)
        ):
            continue
        if artifact.role not in INSTRUCTION_ROLE_PRIORITY:
            latest_unclassified = artifact
            continue
        latest_by_role[artifact.role] = artifact
    for role in INSTRUCTION_ROLE_PRIORITY:
        artifact = latest_by_role.get(role)
        if artifact is not None:
            return artifact
    tasks = state.get("tasks", {})
    if latest_unclassified is not None and task_id not in tasks:
        return latest_unclassified
    return None


def resolve_active_instruction_text(state: RuntimeState, task_id: str) -> str:
    """Return the active instruction text visible to the task."""

    artifact = resolve_active_instruction_artifact(state, task_id)
    if artifact is not None:
        return artifact.get_instruction_text().strip()
    tasks = state.get("tasks", {})
    task = tasks.get(task_id)
    return task.instruction if task is not None else ""
