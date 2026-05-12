"""Minimal grounding tool implementation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from llm import invoke_structured_multimodal_llm
from runtime.instruction_resolver import resolve_active_instruction_text
from runtime.prompts import GROUNDING_SYSTEM_PROMPT, build_grounding_user_prompt
from schema import (
    ArtifactKind,
    GeometryArtifact,
    GroundingArgs,
    GroundingCandidate,
    GroundingLLMOutput,
    GroundingPoint,
    ToolName,
)

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id


class GroundingTool(BaseTool):
    name = ToolName.GROUNDING
    args_schema = GroundingArgs
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

    def _clamp_int(self, value: int | float, lower: int, upper: int) -> int:
        return max(lower, min(int(value), upper))

    def _normalize_point(
        self,
        point: GroundingPoint,
        *,
        width: int,
        height: int,
    ) -> GroundingPoint:
        return GroundingPoint(
            x=self._clamp_int(point.x, 0, max(0, width - 1)),
            y=self._clamp_int(point.y, 0, max(0, height - 1)),
        )

    def _normalize_candidate(
        self,
        candidate: GroundingCandidate,
        *,
        width: int,
        height: int,
    ) -> GroundingCandidate | None:
        x1, y1, x2, y2 = candidate.bbox
        x1 = self._clamp_int(x1, 0, max(0, width - 1))
        y1 = self._clamp_int(y1, 0, max(0, height - 1))
        x2 = self._clamp_int(x2, 0, max(0, width - 1))
        y2 = self._clamp_int(y2, 0, max(0, height - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 >= x2 or y1 >= y2:
            return None
        return GroundingCandidate(
            label=candidate.label,
            bbox=[x1, y1, x2, y2],
            score=candidate.score,
            positive_points=[
                self._normalize_point(point, width=width, height=height)
                for point in candidate.positive_points
            ],
            negative_points=[
                self._normalize_point(point, width=width, height=height)
                for point in candidate.negative_points
            ],
        )

    def execute(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: GroundingArgs,
    ) -> ToolExecutionResult:
        task_instruction = resolve_active_instruction_text(state, task_id)
        image_path = self._resolve_local_image_path(state, args.image_ref)
        top_k = args.top_k or 1
        with Image.open(image_path) as image:
            width, height = image.size

        llm_output = invoke_structured_multimodal_llm(
            system_prompt=GROUNDING_SYSTEM_PROMPT,
            user_prompt=build_grounding_user_prompt(
                grounding_query=args.grounding_query,
                task_instruction=task_instruction,
                width=width,
                height=height,
                top_k=top_k,
            ),
            image_paths=[image_path],
            output_schema=GroundingLLMOutput,
        )

        normalized_candidates: list[GroundingCandidate] = []
        for candidate in llm_output.candidates:
            normalized = self._normalize_candidate(
                candidate,
                width=width,
                height=height,
            )
            if normalized is not None:
                normalized_candidates.append(normalized)
            if len(normalized_candidates) >= top_k:
                break
        if not normalized_candidates:
            raise ValueError("Grounding produced no valid candidates after normalization")

        artifact = GeometryArtifact(
            id=next_artifact_id(state, ArtifactKind.GEOMETRY),
            payload={
                "image_artifact_id": args.image_ref,
                "grounding_query": args.grounding_query,
                "candidates": [
                    {
                        "label": candidate.label,
                        "bbox": list(candidate.bbox),
                        "score": candidate.score,
                        "positive_points": [
                            point.model_dump() for point in candidate.positive_points
                        ],
                        "negative_points": [
                            point.model_dump() for point in candidate.negative_points
                        ],
                    }
                    for candidate in normalized_candidates
                ],
            },
            source_ids=[args.image_ref],
            created_by=self.name.value,
            scope="task",
        )
        return ToolExecutionResult(
            artifacts=[artifact],
            result_payload={
                "image_ref": args.image_ref,
                "grounding_query": args.grounding_query,
                "geometry_artifact_id": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
