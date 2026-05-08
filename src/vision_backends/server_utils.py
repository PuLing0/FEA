"""Shared helpers for lightweight vision backend HTTP servers."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
from typing import Any, Callable

from PIL import Image


class VisionBackendRequestError(ValueError):
    """Raised for client-side request validation errors."""


JsonHandler = Callable[[dict[str, Any]], dict[str, Any]]
HealthHandler = Callable[[], dict[str, Any]]
ShutdownHandler = Callable[[], None]


def read_image(path_value: Any) -> Image.Image:
    if not isinstance(path_value, str) or not path_value.strip():
        raise VisionBackendRequestError("image path must be a non-empty string")
    image_path = Path(path_value)
    if not image_path.is_file():
        raise VisionBackendRequestError(f"image path not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def default_output_path(kind: str, suffix: str) -> str:
    output_dir = Path("generated") / "vision_backend" / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_count = len(list(output_dir.glob(f"*.{suffix}")))
    return str(output_dir / f"remote_{existing_count + 1:06d}.{suffix}")


def resolve_output_path(raw_output_path: Any, *, kind: str, suffix: str) -> str:
    if raw_output_path is None:
        output_path = Path(default_output_path(kind, suffix))
    elif isinstance(raw_output_path, str) and raw_output_path.strip():
        output_path = Path(raw_output_path)
    else:
        raise VisionBackendRequestError("output_path must be a non-empty string when provided")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def log_service_event(service_name: str, message: str, **fields: Any) -> None:
    field_text = " ".join(f"{key}={value!r}" for key, value in fields.items() if value is not None)
    if field_text:
        print(f"[{service_name}] {message} {field_text}", flush=True)
    else:
        print(f"[{service_name}] {message}", flush=True)


def build_single_endpoint_handler(
    *,
    service_name: str,
    post_path: str,
    post_handler: JsonHandler,
    health_handler: HealthHandler,
) -> type[BaseHTTPRequestHandler]:
    class SingleEndpointHandler(BaseHTTPRequestHandler):
        server_version = f"FEA{service_name}/1.0"

        def do_GET(self) -> None:
            if self.path == "/health":
                started_at = time.perf_counter()
                json_response(self, HTTPStatus.OK, health_handler())
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                log_service_event(
                    service_name,
                    "health check completed",
                    status=HTTPStatus.OK.value,
                    elapsed_ms=round(elapsed_ms, 2),
                )
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            log_service_event(service_name, "health check rejected", path=self.path, status=HTTPStatus.NOT_FOUND.value)

        def do_POST(self) -> None:
            started_at = time.perf_counter()
            status = HTTPStatus.OK
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body or "{}")
                if not isinstance(payload, dict):
                    raise VisionBackendRequestError("request body must be a JSON object")
                if self.path != post_path:
                    status = HTTPStatus.NOT_FOUND
                    json_response(self, status, {"error": "not found"})
                    log_service_event(
                        service_name,
                        "request rejected",
                        path=self.path,
                        status=status.value,
                        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                    )
                    return
                log_service_event(
                    service_name,
                    "request received",
                    path=self.path,
                    content_length=content_length,
                )
                response_payload = post_handler(payload)
                json_response(self, status, response_payload)
                log_service_event(
                    service_name,
                    "request completed",
                    path=self.path,
                    status=status.value,
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )
            except VisionBackendRequestError as exc:
                status = HTTPStatus.BAD_REQUEST
                json_response(
                    self,
                    status,
                    {"error": {"error_type": type(exc).__name__, "message": str(exc), "retryable": False}},
                )
                log_service_event(
                    service_name,
                    "request failed",
                    path=self.path,
                    status=status.value,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )
            except Exception as exc:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                json_response(
                    self,
                    status,
                    {"error": {"error_type": type(exc).__name__, "message": str(exc), "retryable": False}},
                )
                log_service_event(
                    service_name,
                    "request failed",
                    path=self.path,
                    status=status.value,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{service_name}] {self.address_string()} - {format % args}", flush=True)

    return SingleEndpointHandler


def build_server(host: str, port: int, handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_cls)


__all__ = [
    "HealthHandler",
    "JsonHandler",
    "ShutdownHandler",
    "VisionBackendRequestError",
    "build_server",
    "build_single_endpoint_handler",
    "default_output_path",
    "json_response",
    "log_service_event",
    "read_image",
    "resolve_output_path",
]
