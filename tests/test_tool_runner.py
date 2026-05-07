from __future__ import annotations

import pytest
from pydantic import Field

from runtime.tool_runner import ToolRunner
from schema import (
    SessionPhase,
    SessionState,
    StrictModel,
    ToolInvocationRecord,
    ToolName,
)
from tools.base import BaseTool, ToolExecutionResult
from tools.registry import ToolRegistry


class RunnerArgs(StrictModel):
    value: int = Field(ge=1)


class FakeTool(BaseTool):
    name = ToolName.UNDERSTAND
    args_schema = RunnerArgs

    def __init__(self, *, validate_error: Exception | None = None, execute_error: Exception | None = None) -> None:
        self.validate_error = validate_error
        self.execute_error = execute_error
        self.seen_args: RunnerArgs | None = None

    def validate_input(self, state, *, task_id: str, loop_index: int, args: RunnerArgs) -> None:
        if self.validate_error is not None:
            raise self.validate_error

    def execute(self, state, *, task_id: str, loop_index: int, args: RunnerArgs) -> ToolExecutionResult:
        self.seen_args = args
        if self.execute_error is not None:
            raise self.execute_error
        invocation = ToolInvocationRecord(
            id="op_fake_001",
            task_id=task_id,
            loop_index=loop_index,
            tool_name=self.name,
            args=args.model_dump(),
            status="succeeded",
            output_refs=[],
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[])


def _state() -> dict:
    return {
        "session": SessionState(session_id="sess_tool_runner", phase=SessionPhase.EXECUTING),
        "artifacts": {},
        "operations": [],
    }


def _runner(tool: BaseTool) -> ToolRunner:
    return ToolRunner(ToolRegistry({tool.name: tool}))


def test_tool_runner_parses_dict_args_and_does_not_append_operations() -> None:
    state = _state()
    tool = FakeTool()

    result = _runner(tool).run(
        state,
        tool.name,
        task_id="task_001",
        loop_index=2,
        args={"value": 3},
    )

    assert result.invocation.status == "succeeded"
    assert result.invocation.args == {"value": 3}
    assert isinstance(tool.seen_args, RunnerArgs)
    assert tool.seen_args.value == 3
    assert state["operations"] == []


def test_tool_runner_accepts_parsed_args() -> None:
    tool = FakeTool()

    result = _runner(tool).run(
        _state(),
        tool.name,
        task_id="task_001",
        loop_index=1,
        args=RunnerArgs(value=5),
    )

    assert result.invocation.status == "succeeded"
    assert result.invocation.args == {"value": 5}


def test_tool_runner_converts_pydantic_validation_error_to_failed_invocation() -> None:
    tool = FakeTool()

    result = _runner(tool).run(
        _state(),
        tool.name,
        task_id="task_001",
        loop_index=1,
        args={"value": 0},
    )

    assert result.invocation.status == "failed"
    assert result.invocation.args == {"value": 0}
    assert result.invocation.error is not None
    assert result.invocation.error["type"] == "InputValidationError"
    assert "greater than or equal to 1" in result.invocation.error["message"]


def test_tool_runner_converts_validate_input_error_to_failed_invocation() -> None:
    tool = FakeTool(validate_error=ValueError("blocked by policy"))

    result = _runner(tool).run(
        _state(),
        tool.name,
        task_id="task_001",
        loop_index=1,
        args={"value": 1},
    )

    assert result.invocation.status == "failed"
    assert result.invocation.error == {
        "type": "ValueError",
        "message": "blocked by policy",
    }


def test_tool_runner_converts_execute_error_to_failed_invocation() -> None:
    tool = FakeTool(execute_error=RuntimeError("backend unavailable"))

    result = _runner(tool).run(
        _state(),
        tool.name,
        task_id="task_001",
        loop_index=1,
        args={"value": 1},
    )

    assert result.invocation.status == "failed"
    assert result.invocation.error == {
        "type": "RuntimeError",
        "message": "backend unavailable",
    }


def test_tool_runner_converts_missing_tool_to_failed_invocation() -> None:
    result = ToolRunner(ToolRegistry({})).run(
        _state(),
        ToolName.EDIT,
        task_id="task_001",
        loop_index=1,
        args={"value": 1},
    )

    assert result.invocation.status == "failed"
    assert result.invocation.tool_name == ToolName.EDIT
    assert result.invocation.args == {"value": 1}
    assert result.invocation.error is not None
    assert result.invocation.error["type"] == "KeyError"
    assert "tool is not registered" in result.invocation.error["message"]


def test_tool_runner_preserves_edit_input_budget_error_payload() -> None:
    class EditInputBudgetExceeded(RuntimeError):
        def __init__(self) -> None:
            self.attempted_refs = ["art_1", "art_2", "art_3", "art_4"]
            self.attempted_count = 4
            super().__init__("too many edit inputs")

    tool = FakeTool(execute_error=EditInputBudgetExceeded())

    result = _runner(tool).run(
        _state(),
        tool.name,
        task_id="task_001",
        loop_index=1,
        args={"value": 1},
    )

    assert result.invocation.status == "failed"
    assert result.invocation.error == {
        "type": "EditInputBudgetExceeded",
        "message": "too many edit inputs",
        "attempted_refs": ["art_1", "art_2", "art_3", "art_4"],
        "attempted_count": 4,
        "limit": 3,
    }


def test_base_tool_rejects_direct_run() -> None:
    with pytest.raises(RuntimeError, match="ToolRunner.run"):
        FakeTool().run(_state(), task_id="task_001", loop_index=1, args=RunnerArgs(value=1))
