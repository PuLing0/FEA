"""Minimal collage tool implementation."""

from __future__ import annotations

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, CollageArgs, ImageArtifact, ToolInvocationRecord, ToolName

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


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
        block_count = len(args.block_artifact_ids)
        cell_width = max(240, 880 // max(1, block_count))
        layers = []
        for index, artifact_id in enumerate(args.block_artifact_ids, start=1):
            layers.append(
                {
                    "source_artifact_id": artifact_id,
                    "layer_index": index,
                    "position": {"x": 40 + (index - 1) * cell_width, "y": 120},
                    "size": {"width": min(320, cell_width - 20), "height": 420},
                    "scale": {"x": 1.0, "y": 1.0},
                    "rotation_degrees": 0,
                    "opacity": 1.0,
                }
            )

        source_ids = list(args.block_artifact_ids)
        if args.previous_collage_ref:
            source_ids.append(args.previous_collage_ref)

        artifact = ImageArtifact(
            id=next_artifact_id(state, ArtifactKind.IMAGE),
            uri=f"store://generated/{task_id}/collage_{loop_index}.png",
            payload={
                "role": "collage_reference",
                "task_instruction": task_instruction,
                "layout_goal": args.layout_goal,
                "block_artifact_ids": list(args.block_artifact_ids),
                "previous_collage_ref": args.previous_collage_ref,
                "canvas": {
                    "width": 1024,
                    "height": 1024,
                    "background": "transparent",
                },
                "layers": layers,
                "source": "collage_output",
            },
            source_ids=source_ids,
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
                "layout_goal": args.layout_goal,
                "block_artifact_ids": list(args.block_artifact_ids),
                "collage_artifact_id": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
