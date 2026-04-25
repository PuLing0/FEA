"""FireRed image editing backend with optional multi-GPU sharding."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
from typing import Any

from PIL import Image
from dotenv import load_dotenv


load_dotenv()


DEFAULT_MODEL_PATH = "/mnt/sda/sijuzheng/models/FireRedTeam/FireRed-Image-Edit-1.1"
DEFAULT_LORA_PATH = "/mnt/sda/sijuzheng/models/FireRedTeam/FireRed-Image-Edit-1.0-Lightning"
DEFAULT_LORA_WEIGHT_NAME = "FireRed-Image-Edit-1.0-Lightning-8steps-v1.1.safetensors"


class FireRedBackendError(RuntimeError):
    """Raised when the FireRed backend cannot be loaded or executed."""


def _get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise FireRedBackendError(f"{name} must be an integer") from exc


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise FireRedBackendError(f"{name} must be an integer") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise FireRedBackendError(f"{name} must be a float") from exc


def _build_backend_settings() -> dict[str, Any]:
    disable_lora = _get_bool("FIRERED_DISABLE_LORA", False)
    return {
        "model_path": _get_setting("FIRERED_MODEL_PATH", DEFAULT_MODEL_PATH),
        "inference_mode": _get_setting("FIRERED_INFERENCE_MODE", "normal"),
        "local_files_only": _get_bool("FIRERED_LOCAL_FILES_ONLY", True),
        "cuda_visible_devices": _get_setting("FIRERED_CUDA_VISIBLE_DEVICES", "3,4,5,6"),
        "device_map": _get_setting("FIRERED_DEVICE_MAP", "manual"),
        "per_gpu_max_memory": _get_setting("FIRERED_PER_GPU_MAX_MEMORY"),
        "cpu_max_memory": _get_setting("FIRERED_CPU_MAX_MEMORY", "128GiB"),
        "generator_device": _get_setting("FIRERED_GENERATOR_DEVICE", "auto"),
        "enable_attention_slicing": _get_bool("FIRERED_ENABLE_ATTENTION_SLICING", False),
        "lora_path": None if disable_lora else _get_setting("FIRERED_LORA_PATH", DEFAULT_LORA_PATH),
        "lora_weight_name": None
        if disable_lora
        else _get_setting(
            "FIRERED_LORA_WEIGHT_NAME",
            DEFAULT_LORA_WEIGHT_NAME,
        ),
        "lora_adapter_name": _get_setting("FIRERED_LORA_ADAPTER_NAME", "demo"),
        "fuse_lora": _get_bool("FIRERED_FUSE_LORA", False),
        "num_inference_steps": _get_int("FIRERED_NUM_INFERENCE_STEPS", 8),
        "true_cfg_scale": _get_float("FIRERED_TRUE_CFG_SCALE", 1.0),
        "guidance_scale": _get_float("FIRERED_GUIDANCE_SCALE", 1.0),
        "negative_prompt": _get_setting("FIRERED_NEGATIVE_PROMPT", " ") or " ",
        "seed": _get_int("FIRERED_SEED", 42),
        "height": _get_optional_int("FIRERED_HEIGHT"),
        "width": _get_optional_int("FIRERED_WIDTH"),
    }


def _maybe_configure_cuda_visible_devices(settings: dict[str, Any]) -> None:
    visible_devices = settings["cuda_visible_devices"]
    if visible_devices is None:
        return
    visible_devices = str(visible_devices).strip()
    if not visible_devices:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _build_max_memory(
    torch_module: Any,
    per_gpu_max_memory: str | None,
    cpu_max_memory: str | None,
) -> dict[Any, str] | None:
    if per_gpu_max_memory is None or not str(per_gpu_max_memory).strip():
        return None
    if not torch_module.cuda.is_available():
        raise FireRedBackendError("FIRERED_PER_GPU_MAX_MEMORY requires CUDA to be available.")
    max_memory: dict[Any, str] = {
        idx: per_gpu_max_memory for idx in range(torch_module.cuda.device_count())
    }
    if cpu_max_memory:
        max_memory["cpu"] = cpu_max_memory
    return max_memory


def _resolve_generator_device(torch_module: Any, settings: dict[str, Any]) -> str:
    generator_device = settings["generator_device"]
    if generator_device == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    return str(generator_device)


@lru_cache(maxsize=1)
def load_pipeline():
    """Load and cache the FireRed image edit pipeline."""

    settings = _build_backend_settings()
    _maybe_configure_cuda_visible_devices(settings)

    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except Exception as exc:
        raise FireRedBackendError(
            "FireRed backend requires torch and diffusers to be installed."
        ) from exc

    try:
        pipeline_cls = getattr(diffusers, "QwenImageEditPlusPipeline")
    except AttributeError as exc:
        raise FireRedBackendError(
            "Installed diffusers package does not expose QwenImageEditPlusPipeline."
        ) from exc

    inference_mode = settings["inference_mode"]
    if inference_mode != "normal":
        raise FireRedBackendError(
            f"Unsupported FireRed inference mode: {inference_mode!r}. Use 'normal'."
        )

    device_map = settings["device_map"] or None
    per_gpu_max_memory = settings["per_gpu_max_memory"] or None
    lora_path = settings["lora_path"] or None
    lora_weight_name = settings["lora_weight_name"] or None

    if device_map == "manual":
        if not torch.cuda.is_available():
            raise FireRedBackendError("FIRERED_DEVICE_MAP='manual' requires CUDA to be available.")
        if torch.cuda.device_count() < 2:
            raise FireRedBackendError(
                "FIRERED_DEVICE_MAP='manual' requires at least 2 visible CUDA devices."
            )
        try:
            manual_module = importlib.import_module("vision_backends._vendor.firered_manual_pipeline")
            load_manual_sharded_pipeline = getattr(
                manual_module,
                "load_manual_sharded_pipeline",
            )
            pipe = load_manual_sharded_pipeline(
                settings["model_path"],
                local_files_only=settings["local_files_only"],
            )
        except Exception as exc:
            raise FireRedBackendError(
                "Failed to load FireRed with manual multi-GPU sharding."
            ) from exc

        if lora_path:
            lora_kwargs: dict[str, Any] = {"adapter_name": settings["lora_adapter_name"]}
            if lora_weight_name:
                lora_kwargs["weight_name"] = lora_weight_name
            if settings["local_files_only"]:
                lora_kwargs["local_files_only"] = True
            pipe.load_lora_weights(lora_path, **lora_kwargs)
            if settings["fuse_lora"]:
                pipe.fuse_lora()

        if settings["enable_attention_slicing"]:
            pipe.enable_attention_slicing()

        pipe.set_progress_bar_config(disable=None)
        return pipe

    if device_map is None and per_gpu_max_memory:
        device_map = "balanced"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
    }
    if settings["local_files_only"]:
        load_kwargs["local_files_only"] = True
    if device_map:
        load_kwargs["device_map"] = device_map

    max_memory = _build_max_memory(
        torch,
        per_gpu_max_memory,
        settings["cpu_max_memory"],
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    try:
        pipe = pipeline_cls.from_pretrained(
            settings["model_path"],
            **load_kwargs,
        )
    except Exception as exc:
        raise FireRedBackendError(
            "Failed to load FireRed model from the configured local path."
        ) from exc

    if not device_map:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(target_device)

    if lora_path:
        lora_kwargs: dict[str, Any] = {"adapter_name": settings["lora_adapter_name"]}
        if lora_weight_name:
            lora_kwargs["weight_name"] = lora_weight_name
        if settings["local_files_only"]:
            lora_kwargs["local_files_only"] = True
        pipe.load_lora_weights(lora_path, **lora_kwargs)
        if settings["fuse_lora"]:
            pipe.fuse_lora()

    if settings["enable_attention_slicing"] or device_map:
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=None)
    return pipe


def edit_images(*, images: list[Image.Image], instruction: str) -> Image.Image:
    """Run one FireRed edit call and return a single edited image."""

    if not images:
        raise FireRedBackendError("FireRed backend requires at least one input image.")
    if not instruction.strip():
        raise FireRedBackendError("FireRed backend requires a non-empty instruction.")

    settings = _build_backend_settings()
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise FireRedBackendError("FireRed backend requires torch to be installed.") from exc

    pipeline = load_pipeline()
    generator_device = _resolve_generator_device(torch, settings)
    try:
        inputs = {
            "image": images,
            "prompt": instruction,
            "generator": torch.Generator(device=generator_device).manual_seed(settings["seed"]),
            "true_cfg_scale": settings["true_cfg_scale"],
            "guidance_scale": settings["guidance_scale"],
            "negative_prompt": settings["negative_prompt"],
            "num_inference_steps": settings["num_inference_steps"],
            "num_images_per_prompt": 1,
        }
        if settings["height"] is not None:
            inputs["height"] = settings["height"]
        if settings["width"] is not None:
            inputs["width"] = settings["width"]
        with torch.inference_mode():
            result = pipeline(**inputs)
    except Exception as exc:
        raise FireRedBackendError(
            "FireRed inference failed while editing the provided images."
        ) from exc

    if not getattr(result, "images", None):
        raise FireRedBackendError("FireRed pipeline returned no output images.")
    output = result.images[0]
    if not isinstance(output, Image.Image):
        raise FireRedBackendError("FireRed pipeline returned a non-PIL output image.")
    return output


def backend_config_snapshot() -> dict[str, Any]:
    """Return a compact non-sensitive snapshot of FireRed runtime settings."""

    settings = _build_backend_settings()
    return {
        "model_path": settings["model_path"],
        "inference_mode": settings["inference_mode"],
        "local_files_only": settings["local_files_only"],
        "cuda_visible_devices": settings["cuda_visible_devices"],
        "device_map": settings["device_map"],
        "per_gpu_max_memory": settings["per_gpu_max_memory"],
        "cpu_max_memory": settings["cpu_max_memory"],
        "generator_device": settings["generator_device"],
        "enable_attention_slicing": settings["enable_attention_slicing"],
        "lora_path": settings["lora_path"],
        "lora_weight_name": settings["lora_weight_name"],
        "lora_adapter_name": settings["lora_adapter_name"],
        "fuse_lora": settings["fuse_lora"],
        "num_inference_steps": settings["num_inference_steps"],
        "true_cfg_scale": settings["true_cfg_scale"],
        "guidance_scale": settings["guidance_scale"],
        "negative_prompt": settings["negative_prompt"],
        "seed": settings["seed"],
        "height": settings["height"],
        "width": settings["width"],
    }


__all__ = [
    "FireRedBackendError",
    "backend_config_snapshot",
    "edit_images",
    "load_pipeline",
]
