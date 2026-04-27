"""Standalone HTTP service for the FireRed edit backend."""

from __future__ import annotations

import argparse
import os
from typing import Any

from .firered_edit_backend import (
    backend_config_snapshot,
    edit_images,
    is_pipeline_loaded,
    load_pipeline,
    unload_pipeline,
)
from .server_utils import (
    VisionBackendRequestError,
    build_server,
    build_single_endpoint_handler,
    read_image,
    resolve_output_path,
)

POST_PATH = "/v1/edit/firered"


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def preload_backend() -> None:
    """Load FireRed once during service startup so VRAM is occupied immediately."""

    load_pipeline()


def handle_edit(payload: dict[str, Any]) -> dict[str, Any]:
    image_paths = payload.get("image_paths")
    if not isinstance(image_paths, list) or not image_paths:
        raise VisionBackendRequestError("image_paths must be a non-empty list")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise VisionBackendRequestError("instruction must be a non-empty string")
    images = [read_image(path) for path in image_paths]
    output = edit_images(images=images, instruction=instruction)
    output_path = resolve_output_path(payload.get("output_path"), kind="edit", suffix="png")
    output.save(output_path)
    return {
        "output_path": output_path,
        "backend_config": backend_config_snapshot(),
    }


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "firered_edit",
        "pid": os.getpid(),
        "firered_cached": is_pipeline_loaded(),
        "firered_preload_on_start": _get_bool("FIRERED_PRELOAD_ON_START", True),
    }


Handler = build_single_endpoint_handler(
    service_name="FireRedEditBackend",
    post_path=POST_PATH,
    post_handler=handle_edit,
    health_handler=health,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone FireRed edit backend service.")
    parser.add_argument("--host", default=os.getenv("FIRERED_EDIT_BACKEND_HOST", os.getenv("VISION_BACKEND_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(os.getenv("FIRERED_EDIT_BACKEND_PORT", "8765")))
    args = parser.parse_args()
    server = build_server(args.host, args.port, Handler)
    preload_on_start = _get_bool("FIRERED_PRELOAD_ON_START", True)
    try:
        if preload_on_start:
            print("Preloading FireRed pipeline on startup...")
            preload_backend()
            print("FireRed pipeline preloaded")
        else:
            print("FireRed startup preload disabled; first request will load the pipeline")
    except Exception:
        server.server_close()
        raise
    print(f"FireRed edit backend listening on http://{args.host}:{args.port}")
    print(f"endpoint: POST {POST_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("FireRed edit backend shutting down")
    finally:
        server.server_close()
        unload_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
