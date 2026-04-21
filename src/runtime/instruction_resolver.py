"""Helpers for resolving active instruction text for a task."""

from __future__ import annotations

from runtime.state import RuntimeState
from schema import InstructionArtifact


def resolve_active_instruction_text(state: RuntimeState, task_id: str) -> str:
    """Return the latest task instruction text.

    Preference order:
    1. latest task-level instruction artifact
    2. fallback to Task.instruction
    """

    task_state = state["session"].task_states[task_id]
    for artifact_id in reversed(task_state.task_artifact_ids):
        artifact = state["artifacts"].get(artifact_id)
        if isinstance(artifact, InstructionArtifact):
            text = artifact.get_instruction_text().strip()
            if text:
                return text
    return state["tasks"][task_id].instruction
