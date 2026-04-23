"""Entry point for the minimal fig edit agent.

This module provides a stable top-level runtime entry:

- `create_agent()` returns a compiled LangGraph runtime
- `agent` exposes a default compiled graph instance

This keeps higher-level integrations from importing deep runtime internals
directly.
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from runtime.graph import build_runtime_graph


def create_agent() -> CompiledStateGraph:
    """Create and return the minimal fig edit runtime graph."""

    return build_runtime_graph()


agent = create_agent()
"""Default compiled runtime graph for the fig edit agent."""
