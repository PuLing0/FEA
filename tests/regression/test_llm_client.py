from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


def test_load_llm_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    config = load_llm_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.invalid/v1"
    assert config.model_name == "test-model"
    assert config.temperature == 0.2


def test_invoke_llm_uses_langchain_model(mocker) -> None:
    fake_response = mocker.Mock()
    fake_response.__class__.__name__ = "AIMessage"

    fake_model = mocker.Mock()
    fake_model.invoke.return_value = fake_response

    result = invoke_llm(
        user_prompt="hello",
        system_prompt="you are helpful",
        model=fake_model,
    )

    assert result is fake_response
    fake_model.invoke.assert_called_once()


def test_invoke_structured_llm_uses_structured_output(mocker) -> None:
    class OutputSchema(BaseModel):
        answer: str

    parsed = OutputSchema(answer="done")
    fake_result = mocker.Mock()
    fake_result.output = parsed
    fake_agent = mocker.Mock()
    fake_agent.run_sync.return_value = fake_result
    agent_cls = mocker.patch("llm.client.Agent", return_value=fake_agent)

    result = invoke_structured_llm(
        user_prompt="return structured",
        system_prompt="system",
        output_schema=OutputSchema,
        model="openai:gpt-5.4",
    )

    assert result == parsed
    agent_cls.assert_called_once()
    fake_agent.run_sync.assert_called_once_with("return structured")


def test_encode_image_path_to_data_url_reads_local_file(tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    data_url = encode_image_path_to_data_url(str(image_path))

    assert data_url.startswith("data:image/png;base64,")


def test_invoke_multimodal_llm_uses_langchain_model_with_image_content(mocker, tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    fake_response = mocker.Mock()
    fake_response.__class__.__name__ = "AIMessage"
    fake_model = mocker.Mock()
    fake_model.invoke.return_value = fake_response

    result = invoke_multimodal_llm(
        user_prompt="describe image",
        system_prompt="system",
        image_paths=[str(image_path)],
        model=fake_model,
    )

    assert result is fake_response
    fake_model.invoke.assert_called_once()
    messages = fake_model.invoke.call_args.args[0]
    assert messages[-1].content[0]["type"] == "text"
    assert messages[-1].content[1]["type"] == "image_url"
    assert messages[-1].content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_invoke_structured_multimodal_llm_uses_structured_output(mocker, tmp_path) -> None:
    class OutputSchema(BaseModel):
        answer: str

    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    parsed = OutputSchema(answer="done")
    fake_structured_model = mocker.Mock()
    fake_structured_model.invoke.return_value = parsed
    fake_model = mocker.Mock()
    fake_model.with_structured_output.return_value = fake_structured_model

    result = invoke_structured_multimodal_llm(
        user_prompt="return structured multimodal",
        system_prompt="system",
        image_paths=[str(image_path)],
        output_schema=OutputSchema,
        model=fake_model,
    )

    assert result == parsed
    fake_model.with_structured_output.assert_called_once_with(OutputSchema)
    fake_structured_model.invoke.assert_called_once()
