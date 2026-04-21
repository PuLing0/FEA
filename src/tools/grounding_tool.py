"""Minimal grounding tool implementation."""

from __future__ import annotations

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import (
    ArtifactKind,
    GeometryArtifact,
    GroundingArgs,
    ToolInvocationRecord,
    ToolName,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class GroundingTool:
    name = ToolName.GROUNDING

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: GroundingArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        top_k = args.top_k or 1
        candidates = []
        for index in range(top_k):
            offset = index * 24
            candidates.append(
                {
                    "bbox": [160 + offset, 180 + offset, 640 + offset, 860 + offset],
                    "score": round(max(0.5, 0.9 - index * 0.12), 2),
                }
            )

        artifact = GeometryArtifact(
            id=next_artifact_id(state, ArtifactKind.GEOMETRY),
            payload={
                "image_artifact_id": args.image_ref,
                "target_description": args.target_description,
                "task_instruction": task_instruction,
                "candidates": candidates,
            },
            source_ids=[args.image_ref],
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
                "target_description": args.target_description,
                "geometry_artifact_id": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
