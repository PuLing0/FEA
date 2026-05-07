"""Centralized tool execution wrapper."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from runtime.state import RuntimeState
from schema import StrictModel, ToolInvocationRecord, ToolName
from tools.base import BaseTool, ToolExecutionResult
from tools.registry import ToolRegistry
from tools.utils import next_operation_id


class ToolRunner:
    """Runs tools through a common validation and error-conversion path."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(
        self,
        state: RuntimeState,
        tool_name: ToolName,
        *,
        task_id: str,
        loop_index: int,
        args: StrictModel | dict[str, Any],
    ) -> ToolExecutionResult:
        raw_args = self._dump_args(args)
        try:
            tool = self._registry.get(tool_name)
            parsed_args = self._parse_args(tool, args)
            tool.validate_input(
                state,
                task_id=task_id,
                loop_index=loop_index,
                args=parsed_args,
            )
            return tool.execute(
                state,
                task_id=task_id,
                loop_index=loop_index,
                args=parsed_args,
            )
        except Exception as exc:
            return ToolExecutionResult(
                invocation=self._build_failed_invocation(
                    state=state,
                    tool_name=tool_name,
                    task_id=task_id,
                    loop_index=loop_index,
                    args=raw_args,
                    error=exc,
                ),
                artifacts=[],
            )

    def failure_result(
        self,
        state: RuntimeState,
        tool_name: ToolName,
        *,
        task_id: str,
        loop_index: int,
        args: StrictModel | dict[str, Any] | None,
        error: Exception,
    ) -> ToolExecutionResult:
        """Build a failed result for errors raised before a tool can be called."""

        return ToolExecutionResult(
            invocation=self._build_failed_invocation(
                state=state,
                tool_name=tool_name,
                task_id=task_id,
                loop_index=loop_index,
                args=self._dump_args(args if args is not None else {}),
                error=error,
            ),
            artifacts=[],
        )

    def _parse_args(
        self,
        tool: BaseTool,
        args: StrictModel | dict[str, Any],
    ) -> StrictModel:
        if isinstance(args, tool.args_schema):
            return args
        if isinstance(args, StrictModel):
            args = args.model_dump()
        return tool.args_schema.model_validate(args)

    def _dump_args(self, args: StrictModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(args, StrictModel):
            return args.model_dump()
        return dict(args)

    def _build_failed_invocation(
        self,
        *,
        state: RuntimeState,
        tool_name: ToolName,
        task_id: str,
        loop_index: int,
        args: dict[str, Any],
        error: Exception,
    ) -> ToolInvocationRecord:
        return ToolInvocationRecord(
            id=next_operation_id(state, tool_name),
            task_id=task_id,
            loop_index=loop_index,
            tool_name=tool_name,
            args=args,
            status="failed",
            output_refs=[],
            error=self._build_error_payload(error),
        )

    def _build_error_payload(self, error: Exception) -> dict[str, Any]:
        if self._is_edit_input_budget_error(error):
            attempted_refs = list(getattr(error, "attempted_refs"))
            return {
                "type": type(error).__name__,
                "message": str(error),
                "attempted_refs": attempted_refs,
                "attempted_count": int(getattr(error, "attempted_count")),
                "limit": 3,
            }
        if isinstance(error, ValidationError):
            return {
                "type": "InputValidationError",
                "message": str(error),
            }
        return {
            "type": type(error).__name__,
            "message": str(error),
        }

    def _is_edit_input_budget_error(self, error: Exception) -> bool:
        return (
            type(error).__name__ == "EditInputBudgetExceeded"
            and hasattr(error, "attempted_refs")
            and hasattr(error, "attempted_count")
        )
