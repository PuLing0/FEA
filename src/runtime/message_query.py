"""Helpers for querying message-first runtime history."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from schema import (
    DecisionBlock,
    EvaluationBlock,
    ObservationBlock,
    ThinkingSummaryBlock,
    ToolCallBlock,
    ToolName,
    ToolResultBlock,
)


BlockT = TypeVar("BlockT")


def iter_blocks(state: dict[str, Any], block_type: type[BlockT] | None = None) -> Iterable[BlockT | Any]:
    for envelope in state.get("messages", []):
        for block in envelope.message.content:
            if block_type is None or isinstance(block, block_type):
                yield block


def iter_task_blocks(
    state: dict[str, Any],
    block_type: type[BlockT],
    *,
    task_id: str | None = None,
) -> Iterable[BlockT]:
    for envelope in state.get("messages", []):
        if task_id is not None and envelope.task_id != task_id:
            continue
        for block in envelope.message.content:
            if isinstance(block, block_type):
                yield block


def find_latest_tool_call(
    state: dict[str, Any],
    *,
    tool_name: ToolName | None = None,
    task_id: str | None = None,
) -> ToolCallBlock | None:
    calls = list(iter_task_blocks(state, ToolCallBlock, task_id=task_id))
    for block in reversed(calls):
        if tool_name is None or block.tool_name == tool_name:
            return block
    return None


def find_latest_tool_result(
    state: dict[str, Any],
    *,
    tool_name: ToolName | None = None,
    task_id: str | None = None,
    status: str | None = None,
) -> ToolResultBlock | None:
    results = list(iter_task_blocks(state, ToolResultBlock, task_id=task_id))
    for block in reversed(results):
        if tool_name is not None and block.tool_name != tool_name:
            continue
        if status is not None and block.status != status:
            continue
        return block
    return None


def find_tool_results(
    state: dict[str, Any],
    *,
    tool_name: ToolName | None = None,
    task_id: str | None = None,
    status: str | None = None,
) -> list[ToolResultBlock]:
    results: list[ToolResultBlock] = []
    for block in iter_task_blocks(state, ToolResultBlock, task_id=task_id):
        if tool_name is not None and block.tool_name != tool_name:
            continue
        if status is not None and block.status != status:
            continue
        results.append(block)
    return results


def count_failed_tool_results(
    state: dict[str, Any],
    *,
    tool_name: ToolName | None = None,
    task_id: str | None = None,
    exclude_error_type: str | None = None,
) -> int:
    count = 0
    for result in find_tool_results(state, tool_name=tool_name, task_id=task_id, status="failed"):
        if exclude_error_type and (result.error or {}).get("type") == exclude_error_type:
            continue
        count += 1
    return count


def find_latest_artifact_ids_by_tool(
    state: dict[str, Any],
    *,
    tool_name: ToolName,
    task_id: str | None = None,
) -> list[str]:
    result = find_latest_tool_result(
        state,
        tool_name=tool_name,
        task_id=task_id,
        status="succeeded",
    )
    return list(result.artifact_ids) if result is not None else []


def find_latest_decision(
    state: dict[str, Any],
    *,
    task_id: str | None = None,
) -> DecisionBlock | None:
    decisions = list(iter_task_blocks(state, DecisionBlock, task_id=task_id))
    return decisions[-1] if decisions else None


def find_latest_evaluation(
    state: dict[str, Any],
    *,
    task_id: str | None = None,
) -> EvaluationBlock | None:
    evaluations = list(iter_task_blocks(state, EvaluationBlock, task_id=task_id))
    return evaluations[-1] if evaluations else None


def find_recent_observations(
    state: dict[str, Any],
    *,
    task_id: str,
    limit: int,
) -> list[ObservationBlock]:
    observations = list(iter_task_blocks(state, ObservationBlock, task_id=task_id))
    return observations[-limit:]


def find_recent_thinking(
    state: dict[str, Any],
    *,
    task_id: str,
    limit: int,
) -> list[ThinkingSummaryBlock]:
    thinking = list(iter_task_blocks(state, ThinkingSummaryBlock, task_id=task_id))
    return thinking[-limit:]


def summarize_tool_result(result: ToolResultBlock) -> dict[str, Any]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "task_id": result.task_id,
        "loop_index": result.loop_index,
        "act_index": result.act_index,
        "status": result.status,
        "args": dict(result.args),
        "artifact_ids": list(result.artifact_ids),
        "error": result.error,
        "raw_output_uri": result.raw_output_uri,
    }
