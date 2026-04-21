"""Prompt reconstruction tool for execution-time instruction rewriting."""

from __future__ import annotations

from llm import invoke_llm
from schema import (
    ArtifactKind,
    InstructionArtifact,
    PromptReconstructArgs,
    ToolInvocationRecord,
    ToolName,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class PromptReconstructTool:
    name = ToolName.PROMPT_RECONSTRUCT

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: PromptReconstructArgs,
    ) -> ToolExecutionResult:
        context_text = state["_prompt_reconstruct_context"]
        response = invoke_llm(
            system_prompt=(
                "You rewrite execution prompts for an image editing task. "
                "Make the instruction clearer, more specific, and replace ambiguous image references "
                "with explicit artifact ids when available. Output only the final rewritten instruction text."
            ),
            user_prompt=context_text,
        )
        instruction_text = str(response.content).strip()
        artifact = InstructionArtifact(
            id=next_artifact_id(state, ArtifactKind.INSTRUCTION),
            payload={"instruction_text": instruction_text},
            source_ids=list(args.input_artifact_ids),
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
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
