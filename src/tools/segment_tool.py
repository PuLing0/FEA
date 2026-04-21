"""Minimal segment tool implementation."""

from __future__ import annotations

from schema import MaskArtifact, SegmentArgs, ToolInvocationRecord, ToolName

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id
from schema import ArtifactKind


class SegmentTool:
    name = ToolName.SEGMENT

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: SegmentArgs,
    ) -> ToolExecutionResult:
        artifact = MaskArtifact(
            id=next_artifact_id(state, ArtifactKind.MASK),
            uri=f"store://generated/{task_id}/mask_{loop_index}.png",
            payload={
                "image_ref": args.image_ref,
                "target": args.target,
                "region_hint": args.region_hint,
                "mask_score": 0.9,
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
                "target": args.target,
                "region_hint": args.region_hint,
                "mask_score": 0.9,
                "mask_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
