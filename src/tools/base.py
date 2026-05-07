"""Shared tool runtime primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from schema import Artifact, StrictModel, ToolInvocationRecord, ToolName

if TYPE_CHECKING:
    from runtime.state import RuntimeState


class ToolExecutionResult(StrictModel):
    """Result of a single runtime tool invocation."""

    invocation: ToolInvocationRecord
    artifacts: list[Artifact] = Field(default_factory=list)


class BaseTool(ABC):
    """Base contract for all runtime tools.

    Tool execution must go through runtime.tool_runner.ToolRunner so validation,
    error conversion, and future governance hooks stay centralized.
    """

    name: ClassVar[ToolName]
    args_schema: ClassVar[type[StrictModel]]
    is_concurrency_safe: ClassVar[bool] = False
    is_read_only: ClassVar[bool] = False
    is_expensive: ClassVar[bool] = False
    requires_backend: ClassVar[bool] = False

    def run(self, *args, **kwargs) -> ToolExecutionResult:
        """Reject direct tool execution outside ToolRunner."""

        raise RuntimeError(
            f"{type(self).__name__}.run() is no longer supported; "
            "use runtime.tool_runner.ToolRunner.run()."
        )

    def validate_input(
        self,
        state: "RuntimeState",
        *,
        task_id: str,
        loop_index: int,
        args: StrictModel,
    ) -> None:
        """Validate an already parsed tool argument object before execution."""

    @abstractmethod
    def execute(
        self,
        state: "RuntimeState",
        *,
        task_id: str,
        loop_index: int,
        args: StrictModel,
    ) -> ToolExecutionResult:
        """Execute the tool implementation."""
