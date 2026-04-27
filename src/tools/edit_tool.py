"""Real edit tool implementation with a backend-generated image output."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import (
    ArtifactKind,
    EditArgs,
    ImageArtifact,
    ToolInvocationRecord,
    ToolName,
)
from vision_backends.firered_edit_backend import (
    FireRedBackendError,
    backend_config_snapshot,
    edit_images,
)
from vision_backends.remote_client import (
    RemoteBackendError,
    request_firered_edit,
    resolve_edit_backend,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class EditTool:
    name = ToolName.EDIT

    def _resolve_image_artifact(self, state, image_ref: str):
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
        return artifact

    def _load_images(self, state, image_refs: list[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for image_ref in image_refs:
            artifact = self._resolve_image_artifact(state, image_ref)
            with Image.open(Path(artifact.uri)) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _write_output(self, image: Image.Image, *, task_id: str, loop_index: int) -> str:
        output_dir = Path("generated") / "edit"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{loop_index:03d}.png"
        image.save(output_path)
        return str(output_path)

    def _build_output_path(self, *, task_id: str, loop_index: int) -> str:
        output_dir = Path("generated") / "edit"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{task_id}_{loop_index:03d}.png")

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: EditArgs,
    ) -> ToolExecutionResult:
        dumped = args.model_dump()
        task_instruction = resolve_active_instruction_text(state, task_id)
        backend_name = resolve_edit_backend()
        backend_snapshot = backend_config_snapshot()
        if backend_name == "remote":
            image_paths = [str(self._resolve_image_artifact(state, image_ref).uri) for image_ref in args.image_refs]
            requested_output_path = self._build_output_path(task_id=task_id, loop_index=loop_index)
            try:
                remote_result = request_firered_edit(
                    image_paths=image_paths,
                    instruction=args.instruction,
                    output_path=requested_output_path,
                )
            except RemoteBackendError as exc:
                raise RuntimeError("edit could not run the remote FireRed backend") from exc
            output_path = remote_result.output_path
            backend_snapshot = remote_result.backend_config
        elif backend_name == "local":
            images = self._load_images(state, args.image_refs)
            try:
                edited_image = edit_images(images=images, instruction=args.instruction)
            except FireRedBackendError as exc:
                raise RuntimeError(
                    "edit could not run the FireRed backend. "
                    "Check backend dependencies, runtime configuration, and input image paths."
                ) from exc
            output_path = self._write_output(
                edited_image,
                task_id=task_id,
                loop_index=loop_index,
            )
        else:
            raise ValueError("EDIT_BACKEND must be 'local' or 'remote'")

        if not Path(output_path).is_file():
            raise FileNotFoundError(f"edit backend output path does not exist: {output_path}")
        primary_image_ref = args.image_refs[0]
        auxiliary_image_refs = list(args.image_refs[1:])
        artifact = ImageArtifact(
            id=next_artifact_id(state, ArtifactKind.IMAGE),
            uri=output_path,
            payload={
                "role": "candidate_image",
                "primary_image_ref": primary_image_ref,
                "auxiliary_image_refs": auxiliary_image_refs,
                "backend_name": "firered_remote" if backend_name == "remote" else "firered",
                "backend_config_snapshot": backend_snapshot,
                "source": "edit_output",
                "task_instruction": task_instruction,
            },
            source_ids=list(args.image_refs),
            created_by=self.name.value,
            scope="task",
        )
        invocation = ToolInvocationRecord(
            id=next_operation_id(state, self.name),
            task_id=task_id,
            loop_index=loop_index,
            tool_name=self.name,
            args=dumped,
            status="succeeded",
            output_refs=[artifact.id],
            result_payload={
                "instruction": args.instruction,
                "image_refs": list(args.image_refs),
                "output_image_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
