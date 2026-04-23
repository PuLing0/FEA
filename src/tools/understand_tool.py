"""Minimal understand tool implementation."""

from __future__ import annotations

from pathlib import Path

from llm import invoke_multimodal_llm
from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, ToolInvocationRecord, ToolName, UnderstandArgs, UnderstandingArtifact

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class UnderstandTool:
    name = ToolName.UNDERSTAND

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
        image_path = self._resolve_local_image_path(state, args.image_ref)
        response = invoke_multimodal_llm(
            system_prompt=(
                "You understand an input image for an image-editing agent. "
                "Return one concise factual summary focused on what is visible and relevant to the task."
            ),
            user_prompt=(
                f"Task instruction: {task_instruction}\n"
                f"Question: {args.question or 'Summarize the visible contents relevant to the task.'}\n"
                "Return only the summary."
            ),
            image_paths=[image_path],
        )
        summary = str(response.content).strip()
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
