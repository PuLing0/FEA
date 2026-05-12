"""Prompt reconstruction tool for execution-time instruction rewriting."""

from __future__ import annotations

from llm import invoke_llm
from runtime.prompts import PROMPT_RECONSTRUCT_SYSTEM_PROMPT
from schema import (
    ArtifactKind,
    InstructionArtifact,
    PromptReconstructArgs,
    ToolName,
)

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id


class PromptReconstructTool(BaseTool):
    name = ToolName.PROMPT_RECONSTRUCT
    args_schema = PromptReconstructArgs

    def execute(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: PromptReconstructArgs,
    ) -> ToolExecutionResult:
        context_text = state["_prompt_reconstruct_context"]
        response = invoke_llm(
            system_prompt=PROMPT_RECONSTRUCT_SYSTEM_PROMPT,
            user_prompt=context_text,
        )
        instruction_text = str(response.content).strip()
        artifact = InstructionArtifact(
            id=next_artifact_id(state, ArtifactKind.INSTRUCTION),
            summary=instruction_text,
            payload={"instruction_text": instruction_text},
            source_ids=list(args.input_artifact_ids),
            created_by=self.name.value,
            role="rewritten_instruction",
            scope="task",
        )
        return ToolExecutionResult(
            artifacts=[artifact],
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
