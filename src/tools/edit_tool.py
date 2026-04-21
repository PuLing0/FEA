"""Minimal edit tool implementation."""

from __future__ import annotations

from typing import cast

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import (
    ArtifactKind,
    EditArgs,
    EditMode,
    ImageArtifact,
    ToolInvocationRecord,
    ToolName,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class EditTool:
    name = ToolName.EDIT

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: EditArgs,
    ) -> ToolExecutionResult:
        dumped = cast(dict, args.model_dump())
        task_instruction = resolve_active_instruction_text(state, task_id)
        artifact = ImageArtifact(
            id=next_artifact_id(state, ArtifactKind.IMAGE),
            uri=f"store://generated/{task_id}/candidate_{loop_index}.png",
            payload={
                "role": "candidate_image",
                "mode": dumped["mode"],
                "source_image_ref": dumped["image_ref"],
                "task_instruction": task_instruction,
                "mask_ref": dumped.get("mask_ref"),
                "reference_refs": dumped.get("reference_refs", []),
            },
            source_ids=[
                dumped["image_ref"],
                *dumped.get("reference_refs", []),
                *([dumped["mask_ref"]] if dumped.get("mask_ref") else []),
            ],
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
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
