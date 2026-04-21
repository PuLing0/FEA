"""Minimal evaluate tool implementation."""

from __future__ import annotations

from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, EvaluateArgs, EvaluationArtifact, ToolInvocationRecord, ToolName

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class EvaluateTool:
    name = ToolName.EVALUATE

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: EvaluateArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        artifact = EvaluationArtifact(
            id=next_artifact_id(state, ArtifactKind.EVALUATION),
            payload={
                "candidate_refs": args.candidate_refs,
                "task_instruction": task_instruction,
                "checks": args.checks,
                "verdict": "needs_review",
                "summary": "minimal evaluation result",
            },
            source_ids=[*args.candidate_refs],
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
                "candidate_refs": args.candidate_refs,
                "task_instruction": task_instruction,
                "checks": args.checks,
                "verdict": "needs_review",
                "summary": "minimal evaluation result",
                "evaluation_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
