"""Centralized tool execution wrapper."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from runtime.state import RuntimeState
from schema import StrictModel, ToolInvocationRecord, ToolName
from tools.base import BaseTool, ToolExecutionResult
from tools.registry import ToolRegistry

from .message_store import (
    append_tool_call_message,
    append_tool_result_message,
    next_tool_call_id,
)


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
        tool_call_id = next_tool_call_id(state, tool_name)
        append_tool_call_message(
            state,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=raw_args,
            task_id=task_id,
            loop_index=loop_index,
        )
        try:
            tool = self._registry.get(tool_name)
            parsed_args = self._parse_args(tool, args)
            tool.validate_input(
                state,
                task_id=task_id,
                loop_index=loop_index,
                args=parsed_args,
            )
            result = self._coerce_result(
                tool.execute(
                    state,
                    task_id=task_id,
                    loop_index=loop_index,
                    args=parsed_args,
                )
            )
            result = result.model_copy(
                update={
                    "status": "succeeded",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "task_id": task_id,
                    "loop_index": loop_index,
                    "args": self._dump_args(parsed_args),
                    "invocation": self._legacy_invocation(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        task_id=task_id,
                        loop_index=loop_index,
                        args=self._dump_args(parsed_args),
                        status="succeeded",
                        output_refs=result.output_refs,
                        result_payload=result.result_payload,
                        raw_output_uri=result.raw_output_uri,
                        error=None,
                    ),
                }
            )
            append_tool_result_message(
                state,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="succeeded",
                args=result.args,
                artifact_ids=result.output_refs,
                result_payload=result.result_payload,
                raw_output_uri=result.raw_output_uri,
                summary=self._summarize_artifacts(result.artifacts),
                task_id=task_id,
                loop_index=loop_index,
            )
            return result
        except Exception as exc:
            error = self._build_error_payload(exc)
            append_tool_result_message(
                state,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="failed",
                args=raw_args,
                artifact_ids=[],
                error=error,
                task_id=task_id,
                loop_index=loop_index,
            )
            return self._failed_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                task_id=task_id,
                loop_index=loop_index,
                args=raw_args,
                error=error,
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

        raw_args = self._dump_args(args if args is not None else {})
        tool_call_id = next_tool_call_id(state, tool_name)
        append_tool_call_message(
            state,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=raw_args,
            task_id=task_id,
            loop_index=loop_index,
        )
        payload = self._build_error_payload(error)
        append_tool_result_message(
            state,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status="failed",
            args=raw_args,
            artifact_ids=[],
            error=payload,
            task_id=task_id,
            loop_index=loop_index,
        )
        return self._failed_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            loop_index=loop_index,
            args=raw_args,
            error=payload,
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

    def _failed_result(
        self,
        *,
        tool_call_id: str,
        tool_name: ToolName,
        task_id: str,
        loop_index: int,
        args: dict[str, Any],
        error: dict[str, Any],
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            status="failed",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            loop_index=loop_index,
            args=args,
            error=error,
            artifacts=[],
            invocation=self._legacy_invocation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                task_id=task_id,
                loop_index=loop_index,
                args=args,
                status="failed",
                output_refs=[],
                result_payload=None,
                raw_output_uri=None,
                error=error,
            ),
        )

    def _coerce_result(self, result: Any) -> ToolExecutionResult:
        if isinstance(result, ToolExecutionResult):
            return result
        data = {
            "artifacts": self._coerce_artifacts(
                self._explicit_attr(result, "artifacts", [])
            ),
        }
        invocation = self._explicit_attr(result, "invocation", None)
        if invocation is not None:
            data["invocation"] = invocation
        result_payload = self._explicit_attr(result, "result_payload", None)
        if isinstance(result_payload, dict):
            data["result_payload"] = result_payload
        raw_output_uri = self._explicit_attr(result, "raw_output_uri", None)
        if isinstance(raw_output_uri, str):
            data["raw_output_uri"] = raw_output_uri
        error = self._explicit_attr(result, "error", None)
        if isinstance(error, dict):
            data["error"] = error
        return ToolExecutionResult.model_validate(data)

    def _explicit_attr(self, value: Any, name: str, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(name, default)
        try:
            attrs = vars(value)
        except TypeError:
            attrs = {}
        if name in attrs:
            return attrs[name]
        if type(value).__module__.startswith("unittest.mock"):
            return default
        return getattr(value, name, default)

    def _coerce_artifacts(self, value: Any) -> list[Any]:
        if value is None or type(value).__module__.startswith("unittest.mock"):
            return []
        try:
            return list(value)
        except TypeError:
            return []

    def _legacy_invocation(
        self,
        *,
        tool_call_id: str,
        tool_name: ToolName,
        task_id: str,
        loop_index: int,
        args: dict[str, Any],
        status: str,
        output_refs: list[str],
        result_payload: dict[str, Any] | None,
        raw_output_uri: str | None,
        error: dict[str, Any] | None,
    ) -> ToolInvocationRecord:
        return ToolInvocationRecord(
            id=tool_call_id,
            task_id=task_id,
            loop_index=loop_index,
            tool_name=tool_name,
            args=args,
            status=status,  # type: ignore[arg-type]
            output_refs=output_refs,
            result_payload=result_payload,
            raw_output_uri=raw_output_uri,
            error=error,
        )

    def _summarize_artifacts(self, artifacts: list[Any]) -> str | None:
        if not artifacts:
            return None
        parts = []
        for artifact in artifacts:
            summary = getattr(artifact, "summary", None) or ""
            kind = getattr(getattr(artifact, "kind", None), "value", getattr(artifact, "kind", None))
            parts.append(f"{artifact.id} ({kind}){': ' + summary if summary else ''}")
        return "; ".join(parts)

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
