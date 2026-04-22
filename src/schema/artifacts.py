"""Artifact schemas for the fig edit agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .base import ArtifactKind, ArtifactStage, StrictModel


class ArtifactBase(StrictModel):
    """Shared artifact fields."""

    id: str
    kind: ArtifactKind
    uri: str | None = None
    summary: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    created_by: str | None = None
    scope: Literal["session", "task"] = "task"
    stage: ArtifactStage = ArtifactStage.WORKING


class ImageArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.IMAGE] = ArtifactKind.IMAGE
    uri: str


class MaskArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.MASK] = ArtifactKind.MASK
    uri: str


class GeometryArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.GEOMETRY] = ArtifactKind.GEOMETRY
    uri: None = None


class InstructionArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.INSTRUCTION] = ArtifactKind.INSTRUCTION

    def get_instruction_text(self) -> str:
        if isinstance(self.payload, dict):
            text = self.payload.get("instruction_text") or self.payload.get(
                "task_instruction"
            )
            if isinstance(text, str):
                return text.strip()
        return ""


class UnderstandingArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.UNDERSTANDING] = ArtifactKind.UNDERSTANDING
    uri: None = None


class EvaluationArtifact(ArtifactBase):
    kind: Literal[ArtifactKind.EVALUATION] = ArtifactKind.EVALUATION
    uri: None = None


Artifact = (
    ImageArtifact
    | MaskArtifact
    | GeometryArtifact
    | InstructionArtifact
    | UnderstandingArtifact
    | EvaluationArtifact
)


class ArtifactIndex(StrictModel):
    """Lightweight type index for visible artifacts."""

    by_type: dict[ArtifactKind, list[str]] = Field(default_factory=dict)
