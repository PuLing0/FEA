"""Helpers for runtime tool implementations."""

from __future__ import annotations

from typing import Iterable

from runtime.state import RuntimeState
from schema import Artifact, ArtifactKind, ToolInvocationRecord, ToolName


def next_artifact_id(state: RuntimeState, kind: ArtifactKind) -> str:
    existing = [
        artifact_id
        for artifact_id, artifact in state.get("artifacts", {}).items()
        if artifact.kind == kind
    ]
    return f"art_{kind.value}_{len(existing) + 1:03d}"


def next_operation_id(state: RuntimeState, tool_name: ToolName) -> str:
    return f"op_{tool_name.value}_{len(state.get('operations', [])) + 1:03d}"


def register_artifacts(state: RuntimeState, artifacts: Iterable[Artifact]) -> None:
    for artifact in artifacts:
        state["artifacts"][artifact.id] = artifact
        state["session"].artifact_index.by_type.setdefault(artifact.kind, []).append(
            artifact.id
        )
