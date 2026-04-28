"""Helpers for task/session artifact memory and working-set management."""

from __future__ import annotations

from typing import Iterable

from runtime.state import RuntimeState
from schema import Artifact, ArtifactKind, WorkingSetEntry


SESSION_WORKING_SET_VISIBLE_KINDS = {
    ArtifactKind.IMAGE,
    ArtifactKind.INSTRUCTION,
}


def make_working_set_entry(
    artifact_id: str,
    *,
    usage: str,
    selection_reason: str = "",
) -> WorkingSetEntry:
    return WorkingSetEntry(
        artifact_id=artifact_id,
        usage=usage,
        selection_reason=selection_reason,
    )


def upsert_working_set_entry(
    entries: list[WorkingSetEntry],
    entry: WorkingSetEntry,
) -> None:
    for index, existing in enumerate(entries):
        if existing.artifact_id == entry.artifact_id:
            entries[index] = entry
            return
    entries.append(entry)


def working_set_artifact_ids(entries: Iterable[WorkingSetEntry]) -> list[str]:
    return [entry.artifact_id for entry in entries]


def working_set_image_ids(state: RuntimeState, entries: Iterable[WorkingSetEntry]) -> list[str]:
    image_ids: list[str] = []
    for artifact_id in working_set_artifact_ids(entries):
        artifact = state["artifacts"].get(artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.IMAGE:
            continue
        if artifact_id not in image_ids:
            image_ids.append(artifact_id)
    return image_ids


def register_artifact_in_session_pool(state: RuntimeState, artifact: Artifact) -> None:
    state["artifacts"][artifact.id] = artifact
    session = state["session"]
    if session.artifact_index is None:
        return
    if artifact.kind != ArtifactKind.IMAGE:
        return
    bucket = session.artifact_index.by_type.setdefault(artifact.kind, [])
    if artifact.id not in bucket:
        bucket.append(artifact.id)


def add_to_session_working_set(
    state: RuntimeState,
    artifact_id: str,
    *,
    usage: str,
    selection_reason: str,
) -> None:
    artifact = state["artifacts"].get(artifact_id)
    if artifact is None or artifact.kind not in SESSION_WORKING_SET_VISIBLE_KINDS:
        return
    upsert_working_set_entry(
        state["session"].session_working_set,
        make_working_set_entry(
            artifact_id,
            usage=usage,
            selection_reason=selection_reason,
        ),
    )


def replace_task_working_set(
    state: RuntimeState,
    task_id: str,
    entries: list[WorkingSetEntry],
) -> None:
    task_state = state["session"].task_states[task_id]
    deduped: list[WorkingSetEntry] = []
    for entry in entries:
        upsert_working_set_entry(deduped, entry)
    task_state.task_working_set = deduped
    task_state.resolved_input_artifact_ids = working_set_image_ids(state, deduped)


def add_to_task_working_set(
    state: RuntimeState,
    task_id: str,
    artifact_id: str,
    *,
    usage: str,
    selection_reason: str,
) -> None:
    task_state = state["session"].task_states[task_id]
    upsert_working_set_entry(
        task_state.task_working_set,
        make_working_set_entry(
            artifact_id,
            usage=usage,
            selection_reason=selection_reason,
        ),
    )


def register_task_artifact(
    state: RuntimeState,
    task_id: str,
    artifact: Artifact,
    *,
    usage: str,
    selection_reason: str,
    expose_to_session_working_set: bool = False,
    session_usage: str | None = None,
    session_selection_reason: str | None = None,
) -> None:
    register_artifact_in_session_pool(state, artifact)
    task_state = state["session"].task_states[task_id]
    if artifact.id not in task_state.task_artifact_ids:
        task_state.task_artifact_ids.append(artifact.id)
    add_to_task_working_set(
        state,
        task_id,
        artifact.id,
        usage=usage,
        selection_reason=selection_reason,
    )
    if expose_to_session_working_set:
        add_to_session_working_set(
            state,
            artifact.id,
            usage=session_usage or usage,
            selection_reason=session_selection_reason or selection_reason,
        )


def preserve_from_task_working_set(state: RuntimeState, task_id: str) -> list[str]:
    task_state = state["session"].task_states[task_id]
    preserved: list[str] = []
    for entry in task_state.task_working_set:
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None or artifact.kind not in SESSION_WORKING_SET_VISIBLE_KINDS:
            continue
        if entry.artifact_id not in preserved:
            preserved.append(entry.artifact_id)
    return preserved


def build_task_working_set_summary(state: RuntimeState, task_id: str) -> str:
    task_state = state["session"].task_states[task_id]
    lines: list[str] = []
    for entry in task_state.task_working_set:
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None:
            continue
        lines.append(
            f"[{artifact.id}] kind={artifact.kind} role={artifact.role or '(none)'} "
            f"usage={entry.usage} summary={artifact.summary or '(no summary)'} "
            f"reason={entry.selection_reason or '(no reason)'}"
        )
    return "\n".join(lines) or "(no current task working set)"


def build_session_working_set_catalog(state: RuntimeState) -> str:
    lines: list[str] = []
    for entry in state["session"].session_working_set:
        artifact = state["artifacts"].get(entry.artifact_id)
        if artifact is None:
            continue
        lines.append(
            f"[{artifact.id}] kind={artifact.kind} role={artifact.role or '(none)'} "
            f"usage={entry.usage} summary={artifact.summary or '(no summary)'}"
        )
    return "\n".join(lines) or "(no session working set artifacts)"


def build_task_pool_catalog(state: RuntimeState, task_id: str) -> str:
    lines: list[str] = []
    for artifact_id in state["session"].task_states[task_id].task_artifact_ids:
        artifact = state["artifacts"].get(artifact_id)
        if artifact is None:
            continue
        lines.append(
            f"[{artifact.id}] kind={artifact.kind} role={artifact.role or '(none)'} "
            f"summary={artifact.summary or '(no summary)'} source_ids={artifact.source_ids}"
        )
    return "\n".join(lines) or "(no task pool artifacts)"
