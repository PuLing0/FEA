"""Minimal understand tool implementation."""

from __future__ import annotations

from pathlib import Path

from llm import invoke_multimodal_llm
from runtime.instruction_resolver import resolve_active_instruction_text
from runtime.prompts import UNDERSTAND_SYSTEM_PROMPT, build_understand_user_prompt
from schema import ArtifactKind, ToolInvocationRecord, ToolName, UnderstandArgs, UnderstandingArtifact

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class UnderstandTool(BaseTool):
    name = ToolName.UNDERSTAND
    args_schema = UnderstandArgs
    is_expensive = True

    def _resolve_local_image_path(self, state, image_ref: str) -> str:
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
        return str(path)

    def execute(
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
        image_path = self._resolve_local_image_path(state, args.image_ref)
        response = invoke_multimodal_llm(
            system_prompt=UNDERSTAND_SYSTEM_PROMPT,
            user_prompt=build_understand_user_prompt(
                task_instruction=task_instruction,
                question=args.question,
            ),
            image_paths=[image_path],
        )
        summary = str(response.content).strip()
        artifact = UnderstandingArtifact(
            id=next_artifact_id(state, ArtifactKind.UNDERSTANDING),
            summary=summary,
            payload={
                "image_ref": args.image_ref,
                "task_instruction": task_instruction,
                "summary": summary,
            },
            source_ids=[args.image_ref],
            created_by=self.name.value,
            role="image_understanding",
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
