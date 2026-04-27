"""Manual multi-GPU sharding helpers for FireRed normal-mode loading."""

from __future__ import annotations

from functools import lru_cache
import importlib
from types import MethodType
from typing import Any


def _chunk_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if parts <= 0:
        raise ValueError("parts must be positive")
    width, remainder = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        stop = start + width + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return [item for item in ranges if item[0] < item[1]]


def _apply_block_ranges(prefix: str, ranges: list[tuple[int, int]], devices: list[int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for device, (start, stop) in zip(devices, ranges, strict=False):
        for index in range(start, stop):
            mapping[f"{prefix}.{index}"] = device
    return mapping


def _match_tensor_device(tensor_name: str, device_map: dict[str, int]) -> int:
    best_prefix = ""
    best_device: int | None = None
    for prefix, device in device_map.items():
        if prefix == "":
            if best_device is None:
                best_device = device
            continue
        if tensor_name == prefix or tensor_name.startswith(f"{prefix}."):
            if len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_device = device
    if best_device is None:
        raise KeyError(f"No device assignment found for tensor '{tensor_name}'")
    return best_device


def _estimate_device_param_bytes(model: Any, device_map: dict[str, int]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for name, param in model.named_parameters():
        device = _match_tensor_device(name, device_map)
        totals[device] = totals.get(device, 0) + param.numel() * param.element_size()
    return totals


def _move_tree_to_device(value: Any, device: Any) -> Any:
    torch = importlib.import_module("torch")
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree_to_device(item, device) for key, item in value.items()}
    return value


@lru_cache(maxsize=8)
def _inspect_transformer_structure(model_path: str, local_files_only: bool) -> dict[str, int]:
    diffusers = importlib.import_module("diffusers")
    accelerate = importlib.import_module("accelerate")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    init_empty_weights = getattr(accelerate, "init_empty_weights")

    config = QwenImageTransformer2DModel.load_config(
        model_path,
        subfolder="transformer",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(config)
    return {"block_count": len(model.transformer_blocks)}


@lru_cache(maxsize=8)
def _inspect_text_encoder_structure(model_path: str, local_files_only: bool) -> dict[str, int]:
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    init_empty_weights = getattr(accelerate, "init_empty_weights")
    Qwen2_5_VLForConditionalGeneration = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")

    config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(
        model_path,
        subfolder="text_encoder",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
    return {
        "visual_block_count": len(model.model.visual.blocks),
        "language_layer_count": len(model.model.language_model.layers),
    }


@lru_cache(maxsize=8)
def _estimate_transformer_device_bytes(
    model_path: str,
    local_files_only: bool,
    device_map_items: tuple[tuple[str, int], ...],
) -> dict[int, int]:
    diffusers = importlib.import_module("diffusers")
    accelerate = importlib.import_module("accelerate")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    init_empty_weights = getattr(accelerate, "init_empty_weights")

    config = QwenImageTransformer2DModel.load_config(
        model_path,
        subfolder="transformer",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(config)
    return _estimate_device_param_bytes(model, dict(device_map_items))


@lru_cache(maxsize=8)
def _estimate_text_encoder_device_bytes(
    model_path: str,
    local_files_only: bool,
    device_map_items: tuple[tuple[str, int], ...],
) -> dict[int, int]:
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    init_empty_weights = getattr(accelerate, "init_empty_weights")
    Qwen2_5_VLForConditionalGeneration = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")

    config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(
        model_path,
        subfolder="text_encoder",
        local_files_only=local_files_only,
    )
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)
    return _estimate_device_param_bytes(model, dict(device_map_items))


def build_manual_shard_plan(
    *,
    model_path: str,
    visible_gpu_ids: list[int],
    local_files_only: bool,
    include_estimates: bool = False,
) -> dict[str, Any]:
    if len(visible_gpu_ids) < 2:
        raise ValueError("manual FireRed sharding requires at least 2 visible CUDA devices")

    transformer_info = _inspect_transformer_structure(model_path, local_files_only)
    text_encoder_info = _inspect_text_encoder_structure(model_path, local_files_only)

    forward_devices = list(visible_gpu_ids)
    if len(forward_devices) >= 4:
        transformer_devices = forward_devices[:2]
        text_devices = forward_devices[2:4]
    elif len(forward_devices) == 3:
        transformer_devices = forward_devices[:2]
        text_devices = forward_devices[2:]
    else:
        transformer_devices = list(forward_devices)
        text_devices = list(forward_devices)

    text_reverse_devices = list(reversed(text_devices))

    transformer_ranges = _chunk_ranges(transformer_info["block_count"], len(transformer_devices))
    visual_ranges = _chunk_ranges(text_encoder_info["visual_block_count"], len(text_devices))
    language_ranges = _chunk_ranges(text_encoder_info["language_layer_count"], len(text_reverse_devices))

    transformer_device_map: dict[str, int] = {
        "pos_embed": transformer_devices[0],
        "time_text_embed": transformer_devices[0],
        "txt_norm": transformer_devices[0],
        "img_in": transformer_devices[0],
        "txt_in": transformer_devices[0],
        "norm_out": transformer_devices[-1],
        "proj_out": transformer_devices[-1],
    }
    transformer_device_map.update(
        _apply_block_ranges("transformer_blocks", transformer_ranges, transformer_devices)
    )

    text_encoder_device_map: dict[str, int] = {
        "model.visual.patch_embed": text_devices[0],
        "model.visual.rotary_pos_emb": text_devices[0],
        "model.visual.merger": text_reverse_devices[0],
        "model.language_model.embed_tokens": text_reverse_devices[0],
        "model.language_model.rotary_emb": text_reverse_devices[-1],
        "model.language_model.norm": text_reverse_devices[-1],
        "lm_head": text_reverse_devices[-1],
    }
    text_encoder_device_map.update(
        _apply_block_ranges("model.visual.blocks", visual_ranges, text_devices)
    )
    text_encoder_device_map.update(
        _apply_block_ranges("model.language_model.layers", language_ranges, text_reverse_devices)
    )

    estimated_param_bytes_by_device: dict[int, int] | None = None
    if include_estimates:
        transformer_device_bytes = _estimate_transformer_device_bytes(
            model_path,
            local_files_only,
            tuple(sorted(transformer_device_map.items())),
        )
        text_encoder_device_bytes = _estimate_text_encoder_device_bytes(
            model_path,
            local_files_only,
            tuple(sorted(text_encoder_device_map.items())),
        )
        estimated_param_bytes_by_device = {
            device: transformer_device_bytes.get(device, 0) + text_encoder_device_bytes.get(device, 0)
            for device in forward_devices
        }
    vae_device = forward_devices[-1]

    return {
        "strategy": "manual_grouped_component_shard" if len(forward_devices) >= 3 else "manual_visible_gpu_shard",
        "visible_gpu_ids": forward_devices,
        "transformer_devices": transformer_devices,
        "text_encoder_devices": text_devices,
        "transformer_device_map": transformer_device_map,
        "text_encoder_device_map": text_encoder_device_map,
        "estimated_param_bytes_by_device": estimated_param_bytes_by_device,
        "vae_device": vae_device,
    }


def load_manual_sharded_pipeline(
    model_path: str,
    *,
    local_files_only: bool = False,
    include_estimates: bool = False,
):
    """Load FireRed with explicit layer-wise sharding across all visible GPUs."""

    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    transformers = importlib.import_module("transformers")

    if not torch.cuda.is_available():
        raise RuntimeError("manual FireRed sharding requires CUDA to be available")

    visible_gpu_ids = list(range(torch.cuda.device_count()))
    shard_plan = build_manual_shard_plan(
        model_path=model_path,
        visible_gpu_ids=visible_gpu_ids,
        local_files_only=local_files_only,
        include_estimates=include_estimates,
    )

    QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline")
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel")
    AutoencoderKLQwenImage = getattr(diffusers, "AutoencoderKLQwenImage")
    Qwen2_5_VLForConditionalGeneration = getattr(
        transformers,
        "Qwen2_5_VLForConditionalGeneration",
    )

    weight_dtype = torch.bfloat16
    pretrained_kwargs = {
        "torch_dtype": weight_dtype,
        "local_files_only": local_files_only,
    }

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        device_map=shard_plan["transformer_device_map"],
        **pretrained_kwargs,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        device_map=shard_plan["text_encoder_device_map"],
        **pretrained_kwargs,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        device_map={"": shard_plan["vae_device"]},
        **pretrained_kwargs,
    )

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        **pretrained_kwargs,
    )

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
    return pipeline


__all__ = [
    "build_manual_shard_plan",
    "load_manual_sharded_pipeline",
]
