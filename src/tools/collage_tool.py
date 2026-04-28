"""Collage tool that renders a real local reference image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm import invoke_structured_multimodal_llm
from PIL import Image
from pydantic import Field, model_validator

from runtime.instruction_resolver import resolve_active_instruction_text
from runtime.prompts import COLLAGE_LAYOUT_SYSTEM_PROMPT, build_collage_layout_user_prompt
from schema import (
    ArtifactKind,
    CollageArgs,
    ImageArtifact,
    StrictModel,
    ToolInvocationRecord,
    ToolName,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


MAX_CANVAS_WIDTH = 4096
MAX_CANVAS_HEIGHT = 4096
MAX_CANVAS_PIXELS = 16_777_216


class CollageLayoutItem(StrictModel):
    """One source image placement in the final collage."""

    artifact_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation_degrees: float = 0.0
    z_index: int = 0
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class CollageLayoutResult(StrictModel):
    """Structured collage layout returned by the multimodal LLM."""

    canvas_width: int = Field(gt=0, le=MAX_CANVAS_WIDTH)
    canvas_height: int = Field(gt=0, le=MAX_CANVAS_HEIGHT)
    background: str = "transparent"
    items: list[CollageLayoutItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_canvas_pixels(self) -> "CollageLayoutResult":
        if self.canvas_width * self.canvas_height > MAX_CANVAS_PIXELS:
            raise ValueError(f"canvas pixel count must be <= {MAX_CANVAS_PIXELS}")
        return self


@dataclass(slots=True)
class SourceImage:
    artifact_id: str
    path: Path
    width: int
    height: int
    summary: str | None
    role: str | None
    description: str | None


class CollageTool:
    name = ToolName.COLLAGE

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: CollageArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        source_images = [
            self._resolve_source_image(state, artifact_id)
            for artifact_id in args.block_artifact_ids
        ]
        planned_layout = self._plan_layout(
            task_instruction=task_instruction,
            args=args,
            source_images=source_images,
        )
        layout = self._expand_canvas_to_fit(
            layout=planned_layout,
            source_images=source_images,
        )
        self._validate_layout(layout=layout, input_ids=args.block_artifact_ids)
        canvas = self._render_collage(layout=layout, source_images=source_images)
        output_path = self._write_collage(
            canvas=canvas,
            task_id=task_id,
            loop_index=loop_index,
        )

        artifact = ImageArtifact(
            id=next_artifact_id(state, ArtifactKind.IMAGE),
            uri=str(output_path),
            payload={
                "role": "collage_reference",
                "task_instruction": task_instruction,
                "layout_goal": args.layout_goal,
                "block_artifact_ids": list(args.block_artifact_ids),
                "canvas": {
                    "width": layout.canvas_width,
                    "height": layout.canvas_height,
                    "background": layout.background,
                },
                "layers": [item.model_dump() for item in layout.items],
                "source": "collage_output",
            },
            source_ids=list(args.block_artifact_ids),
            created_by=self.name.value,
            scope="task",
        )
        if (
            layout.canvas_width != planned_layout.canvas_width
            or layout.canvas_height != planned_layout.canvas_height
        ):
            artifact.payload["canvas_auto_expanded"] = True
            artifact.payload["planned_canvas"] = {
                "width": planned_layout.canvas_width,
                "height": planned_layout.canvas_height,
            }
        invocation = ToolInvocationRecord(
            id=next_operation_id(state, self.name),
            task_id=task_id,
            loop_index=loop_index,
            tool_name=self.name,
            args=args.model_dump(),
            status="succeeded",
            output_refs=[artifact.id],
            result_payload={
                "layout_goal": args.layout_goal,
                "block_artifact_ids": list(args.block_artifact_ids),
                "collage_artifact_id": artifact.id,
                "output_path": str(output_path),
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])

    def _resolve_source_image(self, state, artifact_id: str) -> SourceImage:
        artifact = state["artifacts"].get(artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown image artifact ref: {artifact_id}")
        if artifact.kind != ArtifactKind.IMAGE:
            raise ValueError(f"Artifact is not an image: {artifact_id}")
        if not artifact.uri:
            raise ValueError(f"Image artifact has no uri: {artifact_id}")
        if "://" in artifact.uri:
            raise ValueError(
                f"Image artifact uri is not a local file path: {artifact.uri}"
            )

        path = Path(artifact.uri)
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {artifact.uri}")
        with Image.open(path) as image:
            width, height = image.size

        return SourceImage(
            artifact_id=artifact_id,
            path=path,
            width=width,
            height=height,
            summary=artifact.summary or artifact.payload.get("summary"),
            role=artifact.role,
            description=artifact.payload.get("description"),
        )

    def _plan_layout(
        self,
        *,
        task_instruction: str,
        args: CollageArgs,
        source_images: list[SourceImage],
    ) -> CollageLayoutResult:
        source_context = "\n".join(
            self._format_source_context(source_image, index=index)
            for index, source_image in enumerate(source_images, start=1)
        )
        return invoke_structured_multimodal_llm(
            system_prompt=COLLAGE_LAYOUT_SYSTEM_PROMPT,
            user_prompt=build_collage_layout_user_prompt(
                task_instruction=task_instruction,
                layout_goal=args.layout_goal,
                source_context=source_context,
                max_canvas_width=MAX_CANVAS_WIDTH,
                max_canvas_height=MAX_CANVAS_HEIGHT,
                max_canvas_pixels=MAX_CANVAS_PIXELS,
            ),
            image_paths=[str(source_image.path) for source_image in source_images],
            output_schema=CollageLayoutResult,
        )

    @staticmethod
    def _format_source_context(source_image: SourceImage, *, index: int) -> str:
        pieces = [
            f"{index}. artifact_id={source_image.artifact_id}",
            f"size={source_image.width}x{source_image.height}",
        ]
        if source_image.role:
            pieces.append(f"role={source_image.role}")
        if source_image.summary:
            pieces.append(f"summary={source_image.summary}")
        if source_image.description:
            pieces.append(f"description={source_image.description}")
        return "; ".join(pieces)

    @staticmethod
    def _validate_layout(*, layout: CollageLayoutResult, input_ids: list[str]) -> None:
        input_id_set = set(input_ids)
        layout_ids = [item.artifact_id for item in layout.items]
        duplicate_ids = sorted(
            artifact_id for artifact_id in set(layout_ids) if layout_ids.count(artifact_id) > 1
        )
        if duplicate_ids:
            raise ValueError(f"Collage layout duplicated artifact ids: {duplicate_ids}")
        unknown_ids = [artifact_id for artifact_id in layout_ids if artifact_id not in input_id_set]
        if unknown_ids:
            raise ValueError(f"Collage layout referenced unknown artifact ids: {unknown_ids}")
        missing_ids = [artifact_id for artifact_id in input_ids if artifact_id not in set(layout_ids)]
        if missing_ids:
            raise ValueError(f"Collage layout omitted required artifact ids: {missing_ids}")

        for item in layout.items:
            if item.x + item.width > layout.canvas_width or item.y + item.height > layout.canvas_height:
                raise ValueError(
                    f"Collage item '{item.artifact_id}' exceeds the canvas bounds: "
                    f"item_box=({item.x}, {item.y}, {item.x + item.width}, {item.y + item.height}), "
                    f"canvas=({layout.canvas_width}, {layout.canvas_height})"
                )

    @classmethod
    def _expand_canvas_to_fit(
        cls,
        *,
        layout: CollageLayoutResult,
        source_images: list[SourceImage],
    ) -> CollageLayoutResult:
        required_width = layout.canvas_width
        required_height = layout.canvas_height
        source_by_id = {source_image.artifact_id: source_image for source_image in source_images}

        for item in layout.items:
            required_width = max(required_width, item.x + item.width)
            required_height = max(required_height, item.y + item.height)
            source_image = source_by_id.get(item.artifact_id)
            if source_image is None:
                continue
            layer = cls._build_layer_image(source_image=source_image, item=item)
            required_width = max(required_width, item.x + layer.width)
            required_height = max(required_height, item.y + layer.height)

        if (
            required_width == layout.canvas_width
            and required_height == layout.canvas_height
        ):
            return layout
        if required_width > MAX_CANVAS_WIDTH or required_height > MAX_CANVAS_HEIGHT:
            raise ValueError(
                "Collage layout requires a larger canvas than allowed: "
                f"required=({required_width}, {required_height}), "
                f"max=({MAX_CANVAS_WIDTH}, {MAX_CANVAS_HEIGHT})"
            )
        if required_width * required_height > MAX_CANVAS_PIXELS:
            raise ValueError(
                "Collage layout requires too many pixels after auto-expansion: "
                f"required={required_width * required_height}, max={MAX_CANVAS_PIXELS}"
            )
        return CollageLayoutResult(
            canvas_width=required_width,
            canvas_height=required_height,
            background=layout.background,
            items=[item.model_copy() for item in layout.items],
        )

    @staticmethod
    def _build_layer_image(*, source_image: SourceImage, item: CollageLayoutItem) -> Image.Image:
        with Image.open(source_image.path) as image:
            layer = image.convert("RGBA").resize(
                (item.width, item.height),
                Image.Resampling.LANCZOS,
            )
        if item.rotation_degrees:
            layer = layer.rotate(
                item.rotation_degrees,
                expand=True,
                resample=Image.Resampling.BICUBIC,
            )
        if item.opacity < 1.0:
            alpha = layer.getchannel("A").point(lambda value: int(value * item.opacity))
            layer.putalpha(alpha)
        return layer

    @staticmethod
    def _render_collage(
        *,
        layout: CollageLayoutResult,
        source_images: list[SourceImage],
    ) -> Image.Image:
        canvas = Image.new("RGBA", (layout.canvas_width, layout.canvas_height), (0, 0, 0, 0))
        source_by_id = {source_image.artifact_id: source_image for source_image in source_images}
        for item in sorted(layout.items, key=lambda candidate: candidate.z_index):
            source_image = source_by_id[item.artifact_id]
            layer = CollageTool._build_layer_image(source_image=source_image, item=item)
            if item.x + layer.width > layout.canvas_width or item.y + layer.height > layout.canvas_height:
                raise ValueError(
                    f"Collage item '{item.artifact_id}' exceeds the canvas bounds after rotation: "
                    f"item_box=({item.x}, {item.y}, {item.x + layer.width}, {item.y + layer.height}), "
                    f"canvas=({layout.canvas_width}, {layout.canvas_height})"
                )
            canvas.alpha_composite(layer, dest=(item.x, item.y))
        return canvas

    @staticmethod
    def _write_collage(*, canvas: Image.Image, task_id: str, loop_index: int) -> Path:
        output_dir = Path("generated") / "collage"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{loop_index:03d}.png"
        canvas.save(output_path)
        return output_path
