from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_agent_entrypoint_exports_compiled_graph() -> None:
    compiled = create_agent()
    assert type(compiled).__name__ == "CompiledStateGraph"
    assert type(agent).__name__ == "CompiledStateGraph"


def test_agent_cli_help_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        agent_main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Run the fig edit agent runtime" in captured.out
    assert "--images" in captured.out


def test_agent_cli_defaults_allow_multi_act_execute_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import _build_parser

    monkeypatch.setenv("AGENT_MAX_EXECUTE_ACTS", "5")
    monkeypatch.setenv("AGENT_MAX_TOOL_FAILURES", "4")
    args = _build_parser().parse_args(
        [
            "--images",
            "examples/fig1.jpg",
            "--instruction",
            "Keep the image unchanged.",
        ]
    )

    assert args.max_execute_acts == 5
    assert args.max_tool_failures == 4


def test_agent_cli_runs_rule_based_fallback(mocker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AGENT_LOG_ENABLED", "false")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "false")
    unload_mock = mocker.patch("agent.unload_pipeline")

    exit_code = agent_main(
        [
            "--no-use-llm",
            "--stop-after-first-edit",
            "--session-id",
            "agent-cli-test",
            "--images",
            "examples/fig1.jpg",
            "--instruction",
            "Keep the image unchanged.",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "运行摘要" in captured.out
    assert "已生成第一张编辑候选图后停止" in captured.out
    assert "最终图片" in captured.out
    assert "操作流水" in captured.out
    assert "图像编辑 [succeeded]" in captured.out
    assert "art_image_fake_edit" in captured.out
    assert "store://generated/" in captured.out
    unload_mock.assert_called_once_with()
