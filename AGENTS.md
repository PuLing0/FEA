# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runtime code. Keep orchestration logic in `src/agents/`, graph assembly and scheduling in `src/runtime/`, shared Pydantic models in `src/schema/`, tool adapters in `src/tools/`, and LLM client code in `src/llm/`. The public entrypoint is [src/agent.py](/Users/sijuzheng/project/deepagents_260420/fig_edit_agent/src/agent.py).  
`tests/` currently holds schema and runtime regression tests. `examples/` contains sample images for local experiments. `dev_docs/` stores design notes; update it when replan, input selection, or artifact behavior changes.

## Build, Test, and Development Commands
Use Python 3.11+ and `uv`.

```bash
uv sync
uv run pytest
uv run python -c "from agent import create_agent; print(create_agent())"
uv run langgraph dev
```

`uv sync` installs runtime and dev dependencies from `pyproject.toml` and `uv.lock`. `uv run pytest` runs the test suite. The Python one-liner is a quick import smoke test for the compiled graph. `uv run langgraph dev` starts the local LangGraph dev server using `langgraph.json`.

## Coding Style & Naming Conventions
Follow the existing style: 4-space indentation, type hints on public functions, and concise docstrings. Use `snake_case` for functions, variables, and modules; `PascalCase` for Pydantic models and agent/tool classes; enum values stay lowercase. Prefer strict schemas over loose dictionaries. When adding files, keep names descriptive, for example `runtime/input_selector.py` or `tools/prompt_reconstruct_tool.py`.

## Testing Guidelines
Tests use `pytest`. Add or extend tests in `tests/test_schema.py` unless a new module clearly deserves its own file such as `tests/test_execute_agent.py`. Name tests `test_<behavior>()` and cover both success paths and guardrails, especially around plan validation, replan state transitions, and artifact selection. Run `uv run pytest` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Refactor planner and execute runtime schemas` and `Cover replan and observe runtime behavior`. Keep that pattern: one line, present tense, focused on the main change. PRs should include a brief summary, affected modules, test evidence, and any `.env` or model-config assumptions. Attach screenshots only if a LangGraph UI or visual artifact output changed.

## Security & Configuration Tips
LLM access is configured through `.env` with `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME`, and optional `LLM_TEMPERATURE`. Do not commit secrets or generated runtime artifacts. If `use_llm` is disabled or config is missing, preserve the rule-based fallback path.
