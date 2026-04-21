"""Minimal understand tool implementation."""

from __future__ import annotations

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, ToolInvocationRecord, ToolName, UnderstandArgs, UnderstandingArtifact

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class UnderstandTool:
    name = ToolName.UNDERSTAND

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: UnderstandArgs,
    ) -> ToolExecutionResult:
        task_instruction = (
            state["input"]["instruction_text"]
            if task_id == "bootstrap"
            else resolve_active_instruction_text(state, task_id)
        )
        summary = args.question or "prompt-conditioned understanding for the input image"
        artifact = UnderstandingArtifact(
            id=next_artifact_id(state, ArtifactKind.UNDERSTANDING),
            payload={
                "image_ref": args.image_ref,
                "task_instruction": task_instruction,
                "summary": summary,
            },
            source_ids=[args.image_ref],
            created_by=self.name.value,
            scope="task" if task_id != "bootstrap" else "session",
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
                "task_instruction": task_instruction,
                "summary": summary,
                "understanding_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
