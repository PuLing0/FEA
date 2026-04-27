"""Shared helpers for lightweight vision backend HTTP servers."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
                json_response(self, HTTPStatus.OK, health_handler())
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body or "{}")
                if not isinstance(payload, dict):
                    raise VisionBackendRequestError("request body must be a JSON object")
                if self.path != post_path:
                    json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                json_response(self, HTTPStatus.OK, post_handler(payload))
            except VisionBackendRequestError as exc:
                json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"error_type": type(exc).__name__, "message": str(exc), "retryable": False}},
                )
            except Exception as exc:
                json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"error_type": type(exc).__name__, "message": str(exc), "retryable": False}},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{service_name}] {self.address_string()} - {format % args}")

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
    "read_image",
    "resolve_output_path",
]
