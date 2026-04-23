"""Lazy SAM 3.1 point-prompt backend for segmentation."""

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

from .config import PACKAGE_ROOT, settings


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


def _ensure_vendored_sam3_on_syspath() -> None:
    vendored_root = PACKAGE_ROOT / "vision_backends" / "_vendor"
    vendored_package_dir = vendored_root / "sam3"
    vendored_model_builder = vendored_package_dir / "model_builder.py"
    if not vendored_model_builder.exists():
        raise Sam3BackendError(
            "Vendored SAM3 package is missing from this project. "
            "Expected to find src/vision_backends/_vendor/sam3/model_builder.py."
        )
    vendored_root_str = str(vendored_root)
    if vendored_root_str not in sys.path:
        sys.path.insert(0, vendored_root_str)


@lru_cache(maxsize=1)
def _load_model():
    _maybe_configure_cuda_visible_devices()
    _ensure_vendored_sam3_on_syspath()
    try:
        model_builder = importlib.import_module("sam3.model_builder")
    except Exception as exc:
        raise Sam3BackendError(
            "Failed to import the vendored SAM3 package. "
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
            enable_inst_interactivity=True,
            compile=settings.sam3_compile,
        )
    except Exception as exc:
        raise Sam3BackendError(
            "Failed to initialize the SAM3 image model. "
            "Check checkpoint settings, device selection, and dependency installation."
        ) from exc

    if getattr(model, "inst_interactive_predictor", None) is None:
        raise Sam3BackendError(
            "SAM3 image model was created without inst_interactive_predictor."
        )
    return model


@lru_cache(maxsize=1)
def _load_runtime_modules():
    try:
        torch = importlib.import_module("torch")
        transforms_v2 = importlib.import_module("torchvision.transforms.v2")
    except Exception as exc:
        raise Sam3BackendError(
            "SAM3 runtime requires torch and torchvision to be installed."
        ) from exc
    return torch, transforms_v2


def _sam3_autocast_context(torch_module):
    device = settings.sam3_device.strip().lower()
    if not device.startswith("cuda") or not torch_module.cuda.is_available():
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)


def _build_inference_state(*, model, pil_image, torch_module, transforms_v2):
    width, height = pil_image.size
    image_tensor = transforms_v2.functional.to_image(pil_image).to(settings.sam3_device)
    preprocess = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch_module.uint8, scale=True),
            transforms_v2.Resize(size=(1008, 1008)),
            transforms_v2.ToDtype(torch_module.float32, scale=True),
            transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = preprocess(image_tensor).unsqueeze(0)

    state = {
        "original_height": height,
        "original_width": width,
        "backbone_out": model.backbone.forward_image(image_tensor),
    }

    inst_predictor = model.inst_interactive_predictor
    if inst_predictor is None:
        return state

    sam2_backbone_out = state["backbone_out"].get("sam2_backbone_out")
    if sam2_backbone_out is None:
        return state

    backbone_fpn = sam2_backbone_out.get("backbone_fpn")
    if backbone_fpn is None or len(backbone_fpn) < 2:
        return state

    sam2_backbone_out["backbone_fpn"][0] = (
        inst_predictor.model.sam_mask_decoder.conv_s0(backbone_fpn[0])
    )
    sam2_backbone_out["backbone_fpn"][1] = (
        inst_predictor.model.sam_mask_decoder.conv_s1(backbone_fpn[1])
    )
    return state


def predict_candidates(
    *,
    image_array: np.ndarray,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
) -> list[dict[str, object]]:
    """Run real SAM 3.1 point-prompt segmentation and return raw candidate masks."""

    model = _load_model()
    torch, transforms_v2 = _load_runtime_modules()
    pil_image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
    try:
        point_coords = np.array(
            [[point.x, point.y] for point in [*positive_points, *negative_points]],
            dtype=np.float32,
        )
        point_labels = np.array(
            [1] * len(positive_points) + [0] * len(negative_points),
            dtype=np.int32,
        )
        with torch.inference_mode(), _sam3_autocast_context(torch):
            inference_state = _build_inference_state(
                model=model,
                pil_image=pil_image,
                torch_module=torch,
                transforms_v2=transforms_v2,
            )
            masks, iou_predictions, _ = model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                return_logits=False,
                normalize_coords=False,
            )
    except Exception as exc:
        raise Sam3BackendError(
            "SAM3 point-prompt prediction failed while processing the input image."
        ) from exc

    masks = np.asarray(masks)
    iou_predictions = np.asarray(iou_predictions)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if iou_predictions.ndim == 0:
        iou_predictions = iou_predictions[None]

    candidates: list[dict[str, object]] = []
    for index in range(min(len(masks), len(iou_predictions))):
        candidates.append(
            {
                "name": f"sam31_mask_{index}",
                "mask": np.asarray(masks[index] > 0, dtype=bool),
                "score": float(iou_predictions[index]),
            }
        )
    return candidates


__all__ = ["Sam3BackendError", "predict_candidates"]
