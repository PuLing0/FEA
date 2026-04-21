"""Helpers for DAG task scheduling."""

from __future__ import annotations

from runtime.state import RuntimeState
from schema import TaskStatus


def select_next_runnable_task(state: RuntimeState) -> str | None:
    """Return the next pending task whose dependencies are all satisfied."""

    current_plan_id = state["session"].current_plan_id
    if current_plan_id is None:
        return None

    plan = state["plans"][current_plan_id]
    for task_id in plan.task_ids:
        task_state = state["session"].task_states[task_id]
        if task_state.status != TaskStatus.PENDING:
            continue

        task = state["tasks"][task_id]
        if all(
            state["session"].task_states[dep_id].status == TaskStatus.PASSED
            for dep_id in task.depends_on
        ):
            return task_id

    return None
