"""Shared tool runtime primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from schema import Artifact, StrictModel, ToolName

if TYPE_CHECKING:
    from runtime.state import RuntimeState


class ToolExecutionResult(StrictModel):
    """Result of a single runtime tool invocation."""

    status: Literal["succeeded", "failed"] = "succeeded"
    tool_call_id: str | None = None
    tool_name: ToolName | None = None
    task_id: str | None = None
    loop_index: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    result_payload: dict[str, Any] | None = None
    raw_output_uri: str | None = None
    error: dict[str, Any] | None = None
    invocation: Any | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_invocation(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "invocation" not in data:
            return data
        migrated = dict(data)
        invocation = migrated.pop("invocation")
        migrated.setdefault("invocation", invocation)
        status = cls._explicit_invocation_attr(invocation, "status", "succeeded")
        migrated.setdefault("status", status if status in {"succeeded", "failed"} else "succeeded")
        migrated.setdefault(
            "tool_call_id",
            cls._optional_str(cls._explicit_invocation_attr(invocation, "id", None)),
        )
        migrated.setdefault(
            "tool_name",
            cls._explicit_invocation_attr(invocation, "tool_name", None),
        )
        migrated.setdefault(
            "task_id",
            cls._optional_str(cls._explicit_invocation_attr(invocation, "task_id", None)),
        )
        migrated.setdefault(
            "loop_index",
            cls._optional_int(cls._explicit_invocation_attr(invocation, "loop_index", None)),
        )
        migrated.setdefault(
            "args",
            cls._coerce_args(cls._explicit_invocation_attr(invocation, "args", {})),
        )
        migrated.setdefault(
            "result_payload",
            cls._optional_dict(
                cls._explicit_invocation_attr(invocation, "result_payload", None)
            ),
        )
        migrated.setdefault(
            "raw_output_uri",
            cls._optional_str(
                cls._explicit_invocation_attr(invocation, "raw_output_uri", None)
            ),
        )
        migrated.setdefault(
            "error",
            cls._optional_dict(cls._explicit_invocation_attr(invocation, "error", None)),
        )
        return migrated

    @staticmethod
    def _explicit_invocation_attr(invocation: Any, name: str, default: Any) -> Any:
        if invocation is None:
            return default
        if isinstance(invocation, dict):
            return invocation.get(name, default)
        try:
            attrs = vars(invocation)
        except TypeError:
            attrs = {}
        if name in attrs:
            return attrs[name]
        if type(invocation).__module__.startswith("unittest.mock"):
            return default
        return getattr(invocation, name, default)

    @staticmethod
    def _coerce_args(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _optional_dict(value: Any) -> dict[str, Any] | None:
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) else None

    @property
    def output_refs(self) -> list[str]:
        refs = [artifact.id for artifact in self.artifacts]
        if refs:
            return refs
        legacy_refs = getattr(self.invocation, "output_refs", None)
        if isinstance(legacy_refs, list):
            return list(legacy_refs)
        return []


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
