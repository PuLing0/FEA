"""Official SAM 3.1 backend adapter for prompt-based image segmentation."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import importlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from schema import GroundingPoint

from .config import settings


class Sam3BackendError(RuntimeError):
    """Raised when the SAM 3.1 backend cannot be initialized or executed."""


def _maybe_configure_cuda_visible_devices() -> None:
    visible_devices = settings.sam3_cuda_visible_devices
    if visible_devices is None:
        return
    visible_devices = visible_devices.strip()
    if not visible_devices:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _ensure_official_sam3_on_syspath() -> None:
    repo_root = Path(
        os.getenv("SAM3_REPO_ROOT", "/mnt/sda/sijuzheng/project/sam3")
    ).resolve()
    package_root = repo_root / "sam3"
    init_file = package_root / "__init__.py"
    if not init_file.exists():
        raise Sam3BackendError(
            "Official SAM3 repo is missing or invalid. "
            f"Expected to find {init_file}."
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


@lru_cache(maxsize=1)
def _load_runtime_modules():
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise Sam3BackendError(
            "SAM3 runtime requires torch to be installed."
        ) from exc
    return torch


def _sam3_autocast_context(torch_module):
    device = settings.sam3_device.strip().lower()
    if not device.startswith("cuda") or not torch_module.cuda.is_available():
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)


@lru_cache(maxsize=1)
def _load_image_model():
    _maybe_configure_cuda_visible_devices()
    _ensure_official_sam3_on_syspath()
    try:
        model_builder = importlib.import_module("sam3.model_builder")
    except Exception as exc:
        raise Sam3BackendError(
            "Failed to import the official SAM3 package. "
            "Make sure the SAM3 repo dependencies are installed in the active environment."
        ) from exc

    build_sam3_image_model = getattr(model_builder, "build_sam3_image_model")
    download_ckpt_from_hf = getattr(model_builder, "download_ckpt_from_hf")

    checkpoint_path = settings.sam3_checkpoint_path or None
    load_from_hf = settings.sam3_load_from_hf
    if checkpoint_path is None and load_from_hf:
        checkpoint_path = download_ckpt_from_hf(version=settings.sam3_model_version)
        load_from_hf = False

    try:
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_hf,
            device=settings.sam3_device,
            eval_mode=True,
            enable_inst_interactivity=False,
            compile=settings.sam3_compile,
        )
    except Exception as exc:
        raise Sam3BackendError(
            "Failed to initialize the official SAM3 image model. "
            "Check checkpoint settings, repo dependencies, and device visibility."
        ) from exc

    return model


@lru_cache(maxsize=1)
def _load_text_processor():
    _ensure_official_sam3_on_syspath()
    try:
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
    except Exception as exc:
        raise Sam3BackendError(
            "Failed to import the official SAM3 image processor."
        ) from exc
    Sam3Processor = getattr(processor_module, "Sam3Processor")
    model = _load_image_model()
    return Sam3Processor(model, device=settings.sam3_device)


def _prepare_point_arrays(
    *,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    all_points = [*positive_points, *negative_points]
    if not all_points:
        return None, None
    point_coords = np.array(
        [[point.x, point.y] for point in all_points],
        dtype=np.float32,
    )
    point_labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points),
        dtype=np.int32,
    )
    return point_coords, point_labels


def _prepare_mask_input(mask_input: np.ndarray | None) -> np.ndarray | None:
    if mask_input is None:
        return None
    normalized = np.asarray(mask_input, dtype=np.float32)
    if normalized.ndim == 2:
        normalized = normalized[None, ...]
    if normalized.ndim != 3:
        raise Sam3BackendError("mask_input must be a 2D or 3D array.")
    return normalized


def predict_candidates(
    *,
    image_array: np.ndarray,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
    bbox: list[int] | None = None,
    mask_input: np.ndarray | None = None,
    multimask_output: bool = True,
) -> list[dict[str, object]]:
    """Run official SAM 3.1 prompt-based segmentation and return raw candidate masks."""

    model = _load_image_model()
    torch = _load_runtime_modules()
    point_coords, point_labels = _prepare_point_arrays(
        positive_points=positive_points,
        negative_points=negative_points,
    )
    prepared_mask_input = _prepare_mask_input(mask_input)
    pil_image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))

    try:
        with torch.inference_mode(), _sam3_autocast_context(torch):
            if getattr(model, "inst_interactive_predictor", None) is None:
                raise Sam3BackendError(
                    "Official SAM3 image model does not expose inst_interactive_predictor."
                )
            model.inst_interactive_predictor.set_image(pil_image)
            masks, iou_predictions, low_res_masks = model.inst_interactive_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray(bbox, dtype=np.float32) if bbox is not None else None,
                mask_input=prepared_mask_input,
                multimask_output=multimask_output,
                return_logits=True,
                normalize_coords=False,
            )
    except Exception as exc:
        raise Sam3BackendError(
            "Official SAM3 prediction failed while processing the input image."
        ) from exc

    masks = np.asarray(masks)
    iou_predictions = np.asarray(iou_predictions)
    low_res_masks = np.asarray(low_res_masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if iou_predictions.ndim == 0:
        iou_predictions = iou_predictions[None]
    if low_res_masks.ndim == 2:
        low_res_masks = low_res_masks[None, ...]

    candidates: list[dict[str, object]] = []
    for index in range(min(len(masks), len(iou_predictions), len(low_res_masks))):
        candidates.append(
            {
                "name": f"sam31_mask_{index}",
                "mask": np.asarray(masks[index] > 0, dtype=bool),
                "mask_logits": np.asarray(low_res_masks[index], dtype=np.float32),
                "score": float(iou_predictions[index]),
            }
        )
    return candidates


def predict_text_prompt_candidates(
    *,
    image_array: np.ndarray,
    text_prompt: str,
) -> list[dict[str, object]]:
    """Run official SAM 3.1 text-only image segmentation and return raw candidates."""

    if not text_prompt.strip():
        raise Sam3BackendError("text_prompt must be non-empty for text-based segmentation.")

    processor = _load_text_processor()
    torch = _load_runtime_modules()
    pil_image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
    try:
        with torch.inference_mode(), _sam3_autocast_context(torch):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(prompt=text_prompt, state=state)
    except Exception as exc:
        raise Sam3BackendError(
            "Official SAM3 text-prompt prediction failed while processing the input image."
        ) from exc

    masks = state["masks"].detach().float().cpu().numpy()
    scores = state["scores"].detach().float().cpu().numpy()
    boxes = state["boxes"].detach().float().cpu().numpy()
    if masks.ndim == 3:
        masks = masks[:, None, ...]

    candidates: list[dict[str, object]] = []
    for index in range(min(len(masks), len(scores), len(boxes))):
        mask = np.asarray(masks[index, 0], dtype=bool)
        box = boxes[index].tolist()
        candidates.append(
            {
                "name": f"sam31_text_{index}",
                "mask": mask,
                "mask_logits": mask.astype(np.float32)[None, ...],
                "score": float(scores[index]),
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            }
        )
    return candidates


__all__ = ["Sam3BackendError", "predict_candidates", "predict_text_prompt_candidates"]
