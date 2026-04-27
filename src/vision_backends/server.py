"""Compatibility launcher for vision backend services.

Prefer running dedicated services directly:

- `python -m vision_backends.firered_edit_server`
- `python -m vision_backends.sam31_segment_server`

This module keeps the old `python -m vision_backends.server` entrypoint as a
FireRed edit service alias for backward compatibility.
"""

from __future__ import annotations

from .firered_edit_server import Handler, POST_PATH, handle_edit, health, main
from .server_utils import VisionBackendRequestError, build_server, read_image, resolve_output_path

__all__ = [
    "Handler",
    "POST_PATH",
    "VisionBackendRequestError",
    "build_server",
    "handle_edit",
    "health",
    "main",
    "read_image",
    "resolve_output_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
