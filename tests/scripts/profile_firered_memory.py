"""Profile FireRed edit backend CUDA memory usage by load/inference stage.

Run on a CUDA machine, for example:

    FIRERED_CUDA_VISIBLE_DEVICES=3,4,5,6 \
    ./.venv/bin/python tests/scripts/profile_firered_memory.py

Optional:
    --image examples/fig1.jpg
    --height 768 --width 348 --steps 1
    --skip-lora
    --load-only
    --output generated/profile_firered_memory.json
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

from PIL import Image

from vision_backends.firered_edit_backend import (
    _build_backend_settings,
    _maybe_configure_cuda_visible_devices,
)
from vision_backends._vendor.firered_manual_pipeline import (
    _move_tree_to_device,
    build_manual_shard_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.getenv("FIRERED_PROFILE_IMAGE", "examples/fig1.jpg"))
    parser.add_argument("--prompt", default="Keep the image content unchanged.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--output", default="generated/profile_firered_memory.json")
    return parser.parse_args()


def _bytes_to_gib(value: int) -> float:
    return round(value / 1024**3, 3)


def _snapshot(torch: Any, label: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "logical_device": index,
                "name": torch.cuda.get_device_name(index),
                "total_gib": _bytes_to_gib(total),
                "free_gib": _bytes_to_gib(free),
                "used_gib": _bytes_to_gib(total - free),
                "torch_allocated_gib": _bytes_to_gib(torch.cuda.memory_allocated(index)),
                "torch_reserved_gib": _bytes_to_gib(torch.cuda.memory_reserved(index)),
                "torch_max_allocated_gib": _bytes_to_gib(torch.cuda.max_memory_allocated(index)),
                "torch_max_reserved_gib": _bytes_to_gib(torch.cuda.max_memory_reserved(index)),
            }
        )
    return {"label": label, "devices": devices}


def _reset_peak_memory_stats(torch: Any) -> None:
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats()
        except RuntimeError as exc:
            print(f"warning: could not reset peak memory stats for cuda:{index}: {exc}")


def _record(history: list[dict[str, Any]], torch: Any, label: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    history.append(_snapshot(torch, label))
    print(f"\n[{label}]")
    for device in history[-1]["devices"]:
        print(
            "  cuda:{logical_device} used={used_gib:.3f}GiB "
            "free={free_gib:.3f}GiB torch_alloc={torch_allocated_gib:.3f}GiB "
            "torch_reserved={torch_reserved_gib:.3f}GiB max_alloc={torch_max_allocated_gib:.3f}GiB".format(
                **device
            )
        )


def _patch_manual_pipeline_execution(
    *,
    pipeline: Any,
    torch: Any,
    visible_gpu_ids: list[int],
    shard_plan: dict[str, Any],
) -> None:
    latent_device = torch.device(f"cuda:{visible_gpu_ids[0]}")
    text_encoder_input_device = torch.device(f"cuda:{shard_plan['text_encoder_devices'][0]}")
    vae_device = torch.device(f"cuda:{shard_plan['vae_device']}")
    original_encode_vae_image = pipeline._encode_vae_image
    original_vae_decode = pipeline.vae.decode
    original_get_qwen_prompt_embeds = pipeline._get_qwen_prompt_embeds

    forced_cls = type(
        f"{pipeline.__class__.__name__}ManualExecutionDevice",
        (pipeline.__class__,),
        {"_execution_device": property(lambda self: latent_device)},
    )
    pipeline.__class__ = forced_cls

    def _manual_encode_vae_image(self, image, generator):
        image = image.to(device=vae_device, dtype=self.vae.dtype)
        image_latents = original_encode_vae_image(image=image, generator=generator)
        return image_latents.to(device=latent_device)

    def _manual_vae_decode(latents, *args, **kwargs):
        latents = latents.to(device=vae_device, dtype=pipeline.vae.dtype)
        return original_vae_decode(latents, *args, **kwargs)

    def _manual_get_qwen_prompt_embeds(prompt=None, image=None, device=None, dtype=None):
        requested_device = device or latent_device
        prompt_embeds, encoder_attention_mask = original_get_qwen_prompt_embeds(
            prompt=prompt,
            image=image,
            device=text_encoder_input_device,
            dtype=dtype,
        )
        return (
            _move_tree_to_device(prompt_embeds, requested_device),
            _move_tree_to_device(encoder_attention_mask, requested_device),
        )

    pipeline._encode_vae_image = MethodType(_manual_encode_vae_image, pipeline)
    pipeline.vae.decode = _manual_vae_decode
    pipeline._get_qwen_prompt_embeds = _manual_get_qwen_prompt_embeds
    pipeline.hf_device_map = {
        "transformer": "manual",
        "text_encoder": "manual",
        "vae": shard_plan["vae_device"],
    }
    pipeline._manual_devices = {
        "latent_device": str(latent_device),
        "text_encoder_input_device": str(text_encoder_input_device),
        "vae_device": str(vae_device),
    }
    pipeline._manual_shard_summary = shard_plan


def _load_manual_components(settings: dict[str, Any], history: list[dict[str, Any]]) -> Any:
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    transformers = importlib.import_module("transformers")

    QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    AutoencoderKLQwenImage = getattr(diffusers, "AutoencoderKLQwenImage")
    Qwen2_5_VLForConditionalGeneration = getattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    )

    visible_gpu_ids = list(range(torch.cuda.device_count()))
    shard_plan = build_manual_shard_plan(
        model_path=settings["model_path"],
        visible_gpu_ids=visible_gpu_ids,
        local_files_only=settings["local_files_only"],
        include_estimates=True,
    )
    print("\nmanual shard plan:")
    print(json.dumps(shard_plan, ensure_ascii=False, indent=2, default=str))

    weight_dtype = torch.bfloat16
    pretrained_kwargs = {
        "torch_dtype": weight_dtype,
        "local_files_only": settings["local_files_only"],
    }

    _record(history, torch, "before_load")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        settings["model_path"],
        subfolder="transformer",
        device_map=shard_plan["transformer_device_map"],
        **pretrained_kwargs,
    )
    _record(history, torch, "after_transformer_load")

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings["model_path"],
        subfolder="text_encoder",
        device_map=shard_plan["text_encoder_device_map"],
        **pretrained_kwargs,
    )
    _record(history, torch, "after_text_encoder_load")

    vae = AutoencoderKLQwenImage.from_pretrained(
        settings["model_path"],
        subfolder="vae",
        device_map={"": shard_plan["vae_device"]},
        **pretrained_kwargs,
    )
    _record(history, torch, "after_vae_load")

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        settings["model_path"],
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        **pretrained_kwargs,
    )
    _patch_manual_pipeline_execution(
        pipeline=pipeline,
        torch=torch,
        visible_gpu_ids=visible_gpu_ids,
        shard_plan=shard_plan,
    )
    _record(history, torch, "after_pipeline_assemble")
    return pipeline


def main() -> None:
    args = _parse_args()
    if args.height is not None:
        os.environ["FIRERED_HEIGHT"] = str(args.height)
    if args.width is not None:
        os.environ["FIRERED_WIDTH"] = str(args.width)
    if args.steps is not None:
        os.environ["FIRERED_NUM_INFERENCE_STEPS"] = str(args.steps)
    if args.skip_lora:
        os.environ["FIRERED_DISABLE_LORA"] = "true"

    settings = _build_backend_settings()
    _maybe_configure_cuda_visible_devices(settings)

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    _reset_peak_memory_stats(torch)

    history: list[dict[str, Any]] = []
    if settings["device_map"] == "manual":
        pipeline = _load_manual_components(settings, history)
    else:
        from vision_backends.firered_edit_backend import load_pipeline

        _record(history, torch, "before_load")
        pipeline = load_pipeline()
        _record(history, torch, "after_pipeline_load")

    if settings["lora_path"] and not args.skip_lora:
        lora_kwargs: dict[str, Any] = {"adapter_name": settings["lora_adapter_name"]}
        if settings["lora_weight_name"]:
            lora_kwargs["weight_name"] = settings["lora_weight_name"]
        if settings["local_files_only"]:
            lora_kwargs["local_files_only"] = True
        pipeline.load_lora_weights(settings["lora_path"], **lora_kwargs)
        if settings["fuse_lora"]:
            pipeline.fuse_lora()
        _record(history, torch, "after_lora_load")

    if args.load_only:
        output = {"settings": settings, "history": history}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))
        return

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")

    inputs: dict[str, Any] = {
        "image": [image],
        "prompt": args.prompt,
        "generator": torch.Generator(device="cuda:0").manual_seed(settings["seed"]),
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

    _record(history, torch, "before_inference")
    with torch.inference_mode():
        result = pipeline(**inputs)
    _record(history, torch, "after_inference")

    output_image = result.images[0]
    output_image_path = Path(args.output).with_suffix(".png")
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_image_path)

    output = {
        "settings": settings,
        "output_image": str(output_image_path),
        "history": history,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output}")
    print(f"wrote {output_image_path}")


if __name__ == "__main__":
    main()
