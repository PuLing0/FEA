"""Minimal agent interfaces for the fig edit runtime."""

from .evaluator_agent import EvaluatorAgent
from .execute_agent import ExecuteAgent
from .plan_agent import PlanAgent

__all__ = [
    "EvaluatorAgent",
    "ExecuteAgent",
    "PlanAgent",
]
