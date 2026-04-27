"""HTTP client helpers for remote vision model backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class RemoteBackendError(RuntimeError):
    """Raised when a remote vision backend request fails."""


def _extract_error_message(response_body: str) -> str:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body
    if not isinstance(parsed, dict):
        return response_body
    error_payload = parsed.get("error")
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if isinstance(error_payload, str) and error_payload.strip():
        return error_payload.strip()
    return response_body


def _get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _get_float(name: str, default: float) -> float:
    raw = _get_setting(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RemoteBackendError(f"{name} must be a float") from exc


def resolve_edit_backend() -> str:
    return (_get_setting("EDIT_BACKEND", "remote") or "remote").lower()


def resolve_segment_backend() -> str:
    return (_get_setting("SEGMENT_BACKEND", "remote") or "remote").lower()


def remote_base_url(name: str | None = None) -> str:
    if name == "edit":
        base_url = _get_setting("FIRERED_EDIT_BACKEND_BASE_URL") or _get_setting("VISION_BACKEND_BASE_URL")
        default_url = "http://127.0.0.1:8765"
    elif name == "segment":
        base_url = _get_setting("SAM31_SEGMENT_BACKEND_BASE_URL") or _get_setting("VISION_BACKEND_BASE_URL")
        default_url = "http://127.0.0.1:8766"
    else:
        base_url = _get_setting("VISION_BACKEND_BASE_URL")
        default_url = "http://127.0.0.1:8765"
    if base_url is None:
        base_url = default_url
    if base_url is None:
        raise RemoteBackendError("VISION_BACKEND_BASE_URL is required for remote vision backends")
    return base_url.rstrip("/")


def remote_timeout_seconds() -> float:
    return _get_float("VISION_BACKEND_TIMEOUT_SECONDS", 600.0)


@dataclass(frozen=True)
class RemoteEditResult:
    output_path: str
    backend_config: dict[str, Any]


@dataclass(frozen=True)
class RemoteSegmentResult:
    mask_path: str
    mask_score: float
    selection_metrics: dict[str, float]
    source_stage: str
    text_prompt: str


def _post_json(path: str, payload: dict[str, Any], *, backend_name: str | None = None) -> dict[str, Any]:
    url = f"{remote_base_url(backend_name)}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=remote_timeout_seconds()) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RemoteBackendError(
            f"remote backend HTTP {exc.code}: {_extract_error_message(error_body)}"
        ) from exc
    except error.URLError as exc:
        raise RemoteBackendError(f"remote backend request failed: {exc}") from exc
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RemoteBackendError(f"remote backend returned invalid JSON: {response_body[:200]}") from exc
    if not isinstance(parsed, dict):
        raise RemoteBackendError("remote backend returned a non-object JSON response")
    if parsed.get("error") is not None:
        raise RemoteBackendError(str(parsed["error"]))
    return parsed


def request_firered_edit(*, image_paths: list[str], instruction: str, output_path: str | None = None) -> RemoteEditResult:
    payload: dict[str, Any] = {
        "image_paths": image_paths,
        "instruction": instruction,
    }
    if output_path is not None:
        payload["output_path"] = output_path
    response = _post_json("/v1/edit/firered", payload, backend_name="edit")
    raw_output_path = response.get("output_path")
    if not isinstance(raw_output_path, str) or not raw_output_path:
        raise RemoteBackendError("remote edit response missing output_path")
    backend_config = response.get("backend_config")
    if not isinstance(backend_config, dict):
        backend_config = {}
    return RemoteEditResult(output_path=raw_output_path, backend_config=backend_config)


def request_sam31_segment(*, image_path: str, prompt: str, output_path: str | None = None) -> RemoteSegmentResult:
    payload: dict[str, Any] = {
        "image_path": image_path,
        "prompt": prompt,
    }
    if output_path is not None:
        payload["output_path"] = output_path
    response = _post_json("/v1/segment/sam31", payload, backend_name="segment")
    raw_mask_path = response.get("mask_path")
    if not isinstance(raw_mask_path, str) or not raw_mask_path:
        raise RemoteBackendError("remote segment response missing mask_path")
    metrics = response.get("selection_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return RemoteSegmentResult(
        mask_path=raw_mask_path,
        mask_score=float(response.get("mask_score", 0.0)),
        selection_metrics={str(key): float(value) for key, value in metrics.items()},
        source_stage=str(response.get("source_stage", "remote_sam31")),
        text_prompt=str(response.get("text_prompt", prompt)),
    )


__all__ = [
    "RemoteBackendError",
    "RemoteEditResult",
    "RemoteSegmentResult",
    "remote_base_url",
    "remote_timeout_seconds",
    "request_firered_edit",
    "request_sam31_segment",
    "resolve_edit_backend",
    "resolve_segment_backend",
]
