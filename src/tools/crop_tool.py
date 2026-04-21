"""Minimal crop tool implementation."""

from __future__ import annotations

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, CropArgs, ImageArtifact, ToolInvocationRecord, ToolName

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class CropTool:
    name = ToolName.CROP

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: CropArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        artifact = ImageArtifact(
            id=next_artifact_id(state, ArtifactKind.IMAGE),
            uri=f"store://generated/{task_id}/crop_{loop_index}.png",
            payload={
                "role": "cropped_preview",
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "task_instruction": task_instruction,
                "source": "crop_preview",
            },
            source_ids=[args.image_ref, args.mask_ref],
            created_by=self.name.value,
            scope="task",
        )
        invocation = ToolInvocationRecord(
            id=next_operation_id(state, self.name),
            task_id=task_id,
            loop_index=loop_index,
            tool_name=self.name,
            args=args.model_dump(),
            status="succeeded",
            output_refs=[artifact.id],
            result_payload={
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "crop_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
