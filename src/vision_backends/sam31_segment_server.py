"""Standalone HTTP service for the SAM3.1 text-prompt segment backend."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from .sam3_point_backend import preload_text_prompt_runtime, sam3_runtime_status, unload_runtime
from .server_utils import (
    VisionBackendRequestError,
    build_server,
    build_single_endpoint_handler,
    log_service_event,
    read_image,
    resolve_output_path,
)

POST_PATH = "/v1/segment/sam31"
SERVICE_NAME = "SAM31SegmentBackend"


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def preload_backend() -> None:
    """Load SAM 3.1 once during service startup so VRAM is occupied immediately."""

    log_service_event(SERVICE_NAME, "preload started")
    preload_text_prompt_runtime()
    log_service_event(SERVICE_NAME, "preload completed")


def _preview_text(value: str, *, limit: int = 200) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def handle_segment(payload: dict[str, Any]) -> dict[str, Any]:
    from tools.segment_tool import SegmentTool

    started_at = time.perf_counter()
    image_path = payload.get("image_path")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VisionBackendRequestError("prompt must be a non-empty string")
    log_service_event(
        SERVICE_NAME,
        "segment request payload",
        image_path=image_path,
        prompt=_preview_text(prompt),
        output_path=payload.get("output_path"),
    )
    log_service_event(SERVICE_NAME, "reading segment input image", image_path=image_path)
    image = read_image(image_path)
    image_array = np.asarray(image, dtype=np.uint8)
    tool = SegmentTool()
    if not tool._is_sam31_text_prompt_backend_available(backend_name="sam31"):
        raise VisionBackendRequestError(
            "SAM 3.1 backend is unavailable; set SAM3_CHECKPOINT_PATH on the SAM segment backend server"
        )
    log_service_event(
        SERVICE_NAME,
        "running SAM3.1 text segmentation",
        image_size=image.size,
        runtime_status=sam3_runtime_status(),
    )
    candidates = tool._predict_text_prompt_candidates(
        image_array=image_array,
        text_prompt=prompt,
        source_stage="remote_sam31",
    )
    log_service_event(SERVICE_NAME, "SAM3.1 candidates predicted", candidate_count=len(candidates))
    final_candidate = tool._select_best_candidate(candidates=candidates)
    if final_candidate is None:
        raise VisionBackendRequestError("SAM 3.1 found no acceptable mask candidate")
    final_candidate.mask = tool._postprocess_mask(mask=final_candidate.mask)
    output_path = resolve_output_path(payload.get("output_path"), kind="segment", suffix="png")
    mask_uint8 = (np.asarray(final_candidate.mask, dtype=bool).astype(np.uint8)) * 255
    Image.fromarray(mask_uint8).save(output_path)
    log_service_event(
        SERVICE_NAME,
        "segment mask saved",
        output_path=output_path,
        mask_score=float(final_candidate.score),
        selection_metrics=final_candidate.metrics,
        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return {
        "mask_path": output_path,
        "mask_score": float(final_candidate.score),
        "selection_metrics": final_candidate.metrics,
        "source_stage": final_candidate.source_stage,
        "text_prompt": prompt,
    }


def health() -> dict[str, Any]:
    checkpoint_path = os.getenv("SAM3_CHECKPOINT_PATH")
    runtime_status = sam3_runtime_status()
    return {
        "status": "ok",
        "service": "sam31_segment",
        "pid": os.getpid(),
        "sam3_checkpoint_configured": bool(checkpoint_path),
        "sam31_preload_on_start": _get_bool("SAM31_PRELOAD_ON_START", True),
        "sam3_ready": runtime_status["sam3_image_model_loaded"] and runtime_status["sam3_text_processor_loaded"],
        **runtime_status,
    }


Handler = build_single_endpoint_handler(
    service_name=SERVICE_NAME,
    post_path=POST_PATH,
    post_handler=handle_segment,
    health_handler=health,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone SAM3.1 segment backend service.")
    parser.add_argument("--host", default=os.getenv("SAM31_SEGMENT_BACKEND_HOST", os.getenv("VISION_BACKEND_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(os.getenv("SAM31_SEGMENT_BACKEND_PORT", "8766")))
    args = parser.parse_args()
    server = build_server(args.host, args.port, Handler)
    preload_on_start = _get_bool("SAM31_PRELOAD_ON_START", True)
    try:
        if preload_on_start:
            print("Preloading SAM3.1 text-prompt runtime on startup...", flush=True)
            preload_backend()
            print("SAM3.1 runtime preloaded", flush=True)
        else:
            print("SAM3.1 startup preload disabled; first request will load the runtime", flush=True)
    except Exception:
        server.server_close()
        raise
    print(f"SAM3.1 segment backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"endpoint: POST {POST_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("SAM3.1 segment backend shutting down", flush=True)
    finally:
        server.server_close()
        unload_runtime()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
