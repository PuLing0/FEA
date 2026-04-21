"""Shared tool runtime primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from schema import Artifact, StrictModel, ToolInvocationRecord

if TYPE_CHECKING:
    from runtime.state import RuntimeState


class ToolExecutionResult(StrictModel):
    """Result of a single runtime tool invocation."""

    invocation: ToolInvocationRecord
    artifacts: list[Artifact] = Field(default_factory=list)
