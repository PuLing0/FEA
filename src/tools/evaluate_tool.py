"""Minimal evaluate tool implementation."""

from __future__ import annotations

from pathlib import Path

from llm import invoke_multimodal_llm
from runtime.instruction_resolver import resolve_active_instruction_text
from schema import ArtifactKind, EvaluateArgs, EvaluationArtifact, ToolInvocationRecord, ToolName

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


class EvaluateTool:
    name = ToolName.EVALUATE

    def _resolve_candidate_image_paths(self, state, candidate_refs: list[str]) -> list[str]:
        image_paths: list[str] = []
        for candidate_ref in candidate_refs:
            artifact = state["artifacts"].get(candidate_ref)
            if artifact is None:
                raise ValueError(f"Unknown candidate artifact ref: {candidate_ref}")
            if artifact.kind != ArtifactKind.IMAGE:
                raise ValueError(f"Candidate artifact is not an image: {candidate_ref}")
            if not artifact.uri:
                raise ValueError(f"Candidate image has no uri: {candidate_ref}")
            if "://" in artifact.uri:
                raise ValueError(
                    f"Candidate image uri is not a local file path: {artifact.uri}"
                )
            path = Path(artifact.uri)
            if not path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {artifact.uri}")
            image_paths.append(str(path))
        return image_paths

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: EvaluateArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        image_paths = self._resolve_candidate_image_paths(state, args.candidate_refs)
        response = invoke_multimodal_llm(
            system_prompt=(
                "You evaluate candidate images for an image-editing task. "
                "Check whether the visible result satisfies the acceptance criteria. "
                "Return one concise factual evaluation summary."
            ),
            user_prompt=(
                f"Task instruction: {task_instruction}\n"
                f"Acceptance criteria: {args.checks}\n"
                "Evaluate the candidate image(s) and summarize whether they satisfy the checks."
            ),
            image_paths=image_paths,
        )
        evaluation_summary = str(response.content).strip()
        artifact = EvaluationArtifact(
            id=next_artifact_id(state, ArtifactKind.EVALUATION),
            payload={
                "candidate_refs": args.candidate_refs,
                "task_instruction": task_instruction,
                "checks": args.checks,
                "verdict": "needs_review",
                "summary": evaluation_summary,
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
                "summary": evaluation_summary,
                "evaluation_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
