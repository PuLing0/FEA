"""Environment-driven runtime defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_EXECUTE_ACTS = 4

load_dotenv(REPO_ROOT / ".env")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return value


def default_max_execute_acts() -> int:
    """Return the default execute-loop act budget from environment."""

    return _env_positive_int("AGENT_MAX_EXECUTE_ACTS", DEFAULT_MAX_EXECUTE_ACTS)
