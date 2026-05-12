"""Minimal crop tool implementation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, CropArgs, ImageArtifact, ToolName

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id


class CropTool(BaseTool):
    name = ToolName.CROP
    args_schema = CropArgs

    def _resolve_local_image_path(self, state, image_ref: str) -> str:
        artifact = state["artifacts"].get(image_ref)
        if artifact is None:
            raise ValueError(f"Unknown image artifact ref: {image_ref}")
        if artifact.kind != ArtifactKind.IMAGE:
            raise ValueError(f"Artifact is not an image: {image_ref}")
        if not artifact.uri:
            raise ValueError(f"Image artifact has no uri: {image_ref}")
        if "://" in artifact.uri:
            raise ValueError(
                f"Image artifact uri is not a local file path: {artifact.uri}"
            )
        path = Path(artifact.uri)
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {artifact.uri}")
        return str(path)

    def _resolve_mask_artifact(self, state, mask_ref: str, image_ref: str):
        artifact = state["artifacts"].get(mask_ref)
        if artifact is None:
            raise ValueError(f"Unknown mask artifact ref: {mask_ref}")
        if artifact.kind != ArtifactKind.MASK:
            raise ValueError(f"Artifact is not a mask: {mask_ref}")
        if artifact.payload.get("image_ref") != image_ref:
            raise ValueError("mask artifact does not belong to the requested image")
        if not artifact.uri:
            raise ValueError(f"Mask artifact has no uri: {mask_ref}")
        if "://" in artifact.uri:
            raise ValueError(
                f"Mask artifact uri is not a local file path: {artifact.uri}"
            )
        path = Path(artifact.uri)
        if not path.is_file():
            raise FileNotFoundError(f"Mask path does not exist: {artifact.uri}")
        return artifact

    def _resolve_grounding_artifact(self, state, grounding_ref: str, image_ref: str):
        artifact = state["artifacts"].get(grounding_ref)
        if artifact is None:
            raise ValueError(f"Unknown grounding artifact ref: {grounding_ref}")
        if artifact.kind != ArtifactKind.GEOMETRY:
            raise ValueError(f"Artifact is not geometry: {grounding_ref}")
        if artifact.payload.get("image_artifact_id") != image_ref:
            raise ValueError("grounding artifact does not belong to the requested image")
        candidates = artifact.payload.get("candidates", [])
        if not candidates:
            raise ValueError("grounding artifact has no candidates")
        return artifact

    def _apply_padding_and_clamp(
        self,
        bbox: list[int],
        *,
        width: int,
        height: int,
        padding: int,
    ) -> list[int]:
        left, top, right, bottom = bbox
        left = max(0, int(left) - padding)
        top = max(0, int(top) - padding)
        right = min(width, int(right) + padding)
        bottom = min(height, int(bottom) + padding)
        if left >= right or top >= bottom:
            raise ValueError("crop bbox becomes invalid after clamp")
        return [left, top, right, bottom]

    def _resolve_crop_geometry(self, state, args: CropArgs, *, width: int, height: int):
        if args.mask_ref is not None:
            mask_artifact = self._resolve_mask_artifact(state, args.mask_ref, args.image_ref)
            mask = Image.open(Path(mask_artifact.uri)).convert("L")
            bbox_tuple = mask.getbbox()
            if bbox_tuple is None:
                raise ValueError("mask artifact produced an empty bounding box")
            bbox = self._apply_padding_and_clamp(
                [bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], bbox_tuple[3]],
                width=width,
                height=height,
                padding=args.padding,
            )
            return bbox, mask, "mask_cutout"

        assert args.grounding_ref is not None
        geometry_artifact = self._resolve_grounding_artifact(
            state,
            args.grounding_ref,
            args.image_ref,
        )
        first_candidate = geometry_artifact.payload["candidates"][0]
        bbox = self._apply_padding_and_clamp(
            list(first_candidate["bbox"]),
            width=width,
            height=height,
            padding=args.padding,
        )
        return bbox, None, "grounding_preview"

    def _write_crop_image(self, image: Image.Image, *, task_id: str, loop_index: int, artifact_id: str) -> str:
        output_dir = Path("generated") / "crop"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{loop_index:03d}_{artifact_id}.png"
        image.save(output_path)
        return str(output_path)

    def execute(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: CropArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        image_path = self._resolve_local_image_path(state, args.image_ref)
        image = Image.open(Path(image_path)).convert("RGBA")
        bbox, mask, crop_mode = self._resolve_crop_geometry(
            state,
            args,
            width=image.width,
            height=image.height,
        )
        left, top, right, bottom = bbox
        cropped = image.crop((left, top, right, bottom))
        if crop_mode == "mask_cutout":
            assert mask is not None
            local_mask = mask.crop((left, top, right, bottom)).convert("L")
            cropped.putalpha(local_mask)
        artifact_id = next_artifact_id(state, ArtifactKind.IMAGE)
        output_path = self._write_crop_image(
            cropped,
            task_id=task_id,
            loop_index=loop_index,
            artifact_id=artifact_id,
        )
        artifact = ImageArtifact(
            id=artifact_id,
            uri=output_path,
            payload={
                "role": "cropped_preview",
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "grounding_ref": args.grounding_ref,
                "bbox": bbox,
                "padding": args.padding,
                "crop_mode": crop_mode,
                "task_instruction": task_instruction,
                "source": "crop_preview",
            },
            source_ids=[
                args.image_ref,
                *([args.mask_ref] if args.mask_ref else []),
                *([args.grounding_ref] if args.grounding_ref else []),
            ],
            created_by=self.name.value,
            scope="task",
        )
        return ToolExecutionResult(
            artifacts=[artifact],
            result_payload={
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "grounding_ref": args.grounding_ref,
                "crop_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
