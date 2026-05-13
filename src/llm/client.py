"""Thin LLM wrapper for text and multimodal model access."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import Field

from schema import StrictModel


class LLMRequestError(RuntimeError):
    """Raised when an outbound LLM request fails before a usable response is returned."""


class LLMConfig(StrictModel):
    """Environment-driven LLM configuration."""

    api_key: str = Field(alias="LLM_API_KEY")
    base_url: str = Field(alias="LLM_BASE_URL")
    model_name: str = Field(alias="LLM_MODEL_NAME")
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")


def load_llm_config() -> LLMConfig:
    """Load LLM configuration from `.env` and the current environment."""

    load_dotenv()
    raw = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", ""),
        "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", ""),
        "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE", "0.0"),
    }
    return LLMConfig.model_validate(raw)


def make_chat_model(config: LLMConfig | None = None) -> ChatOpenAI:
    """Create a LangChain chat model from config."""

    resolved = config or load_llm_config()
    return ChatOpenAI(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        model=resolved.model_name,
        temperature=resolved.temperature,
    )


def make_pydantic_ai_model(config: LLMConfig | None = None) -> OpenAIChatModel:
    """Create a PydanticAI OpenAI-compatible model from config."""

    resolved = config or load_llm_config()
    return OpenAIChatModel(
        resolved.model_name,
        provider=OpenAIProvider(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
        ),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
        ),
    )


def _describe_llm_target(
    *,
    model: Any | None = None,
    config: LLMConfig | None = None,
) -> tuple[str, str]:
    resolved_config = config
    if resolved_config is None and model is None:
        try:
            resolved_config = load_llm_config()
        except Exception:
            resolved_config = None

    model_name = (
        getattr(resolved_config, "model_name", None)
        or getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or (model if isinstance(model, str) else None)
        or "unknown"
    )
    base_url = getattr(resolved_config, "base_url", None) or "unknown"
    return str(model_name), str(base_url)


def _raise_request_error(
    *,
    request_kind: str,
    model: Any | None,
    config: LLMConfig | None,
    error: Exception,
) -> None:
    model_name, base_url = _describe_llm_target(model=model, config=config)
    detail = str(error).strip() or error.__class__.__name__
    raise LLMRequestError(
        f"LLM request failed while sending {request_kind} request "
        f"(model={model_name}, base_url={base_url}): {detail}"
    ) from error


def _build_messages(system_prompt: str | None, user_prompt: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    return messages


def encode_image_path_to_data_url(image_path: str) -> str:
    """Read a local image file and return a data URL."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image file type: {image_path}")

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _build_multimodal_user_content(
    user_prompt: str,
    image_paths: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_prompt,
        }
    ]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_path_to_data_url(image_path),
                },
            }
        )
    return content


def _build_multimodal_messages(
    system_prompt: str | None,
    user_prompt: str,
    image_paths: list[str],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content=_build_multimodal_user_content(
                user_prompt=user_prompt,
                image_paths=image_paths,
            )
        )
    )
    return messages


def invoke_llm(
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    model: ChatOpenAI | None = None,
    config: LLMConfig | None = None,
) -> AIMessage:
    """Send a standard chat request and return the raw AI message."""

    chat_model = model or make_chat_model(config)
    try:
        response = chat_model.invoke(_build_messages(system_prompt, user_prompt))
    except Exception as exc:
        _raise_request_error(
            request_kind="chat",
            model=model,
            config=config,
            error=exc,
        )
    if not isinstance(response, AIMessage) and not hasattr(response, "content"):
        raise TypeError("Expected AIMessage response from chat model")
    return response


def invoke_multimodal_llm(
    *,
    user_prompt: str,
    image_paths: list[str],
    system_prompt: str | None = None,
    model: ChatOpenAI | None = None,
    config: LLMConfig | None = None,
) -> AIMessage:
    """Send a multimodal chat request with local image paths."""

    if not image_paths:
        raise ValueError("image_paths must be non-empty for multimodal calls")

    chat_model = model or make_chat_model(config)
    try:
        response = chat_model.invoke(
            _build_multimodal_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
            )
        )
    except Exception as exc:
        _raise_request_error(
            request_kind="multimodal chat",
            model=model,
            config=config,
            error=exc,
        )
    if not isinstance(response, AIMessage) and not hasattr(response, "content"):
        raise TypeError("Expected AIMessage response from chat model")
    return response


def invoke_structured_llm(
    *,
    user_prompt: str,
    output_schema: type,
    system_prompt: str | None = None,
    model: Any | None = None,
    config: LLMConfig | None = None,
) -> Any:
    """Send a chat request and parse the response as structured output.

    Uses PydanticAI because the current OpenAI-compatible provider does not
    reliably support LangChain's native structured output adapters.
    """

    pydantic_ai_model = model or make_pydantic_ai_model(config)
    agent = Agent(
        pydantic_ai_model,
        instructions=system_prompt or "",
        output_type=output_schema,
    )
    try:
        return agent.run_sync(user_prompt).output
    except Exception as exc:
        _raise_request_error(
            request_kind="structured",
            model=model,
            config=config,
            error=exc,
        )


def invoke_structured_multimodal_llm(
    *,
    user_prompt: str,
    image_paths: list[str],
    output_schema: type,
    system_prompt: str | None = None,
    model: ChatOpenAI | None = None,
    config: LLMConfig | None = None,
) -> Any:
    """Send a multimodal request and parse JSON output with a Pydantic schema."""

    if not image_paths:
        raise ValueError("image_paths must be non-empty for multimodal calls")

    chat_model = model or make_chat_model(config)
    structured_model = chat_model.with_structured_output(output_schema)
    try:
        return structured_model.invoke(
            _build_multimodal_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
            )
        )
    except Exception as exc:
        _raise_request_error(
            request_kind="structured multimodal",
            model=model,
            config=config,
            error=exc,
        )
