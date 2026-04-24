"""Configuration for optional vision backends."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class VisionSettings:
    """Environment-backed settings for segmentation backends."""

    def __init__(self) -> None:
        self.sam3_checkpoint_path = os.getenv("SAM3_CHECKPOINT_PATH") or None
        self.sam3_model_version = os.getenv("SAM3_MODEL_VERSION", "sam3.1")
        self.sam3_load_from_hf = os.getenv("SAM3_LOAD_FROM_HF", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.sam3_device = os.getenv("SAM3_DEVICE", "cpu")
        self.sam3_cuda_visible_devices = os.getenv("SAM3_CUDA_VISIBLE_DEVICES") or None
        if self.sam3_cuda_visible_devices and os.getenv("SAM3_DEVICE") is None:
            self.sam3_device = "cuda"
        self.sam3_compile = os.getenv("SAM3_COMPILE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


settings = VisionSettings()


__all__ = ["PACKAGE_ROOT", "VisionSettings", "settings"]
