"""Single-path segmentation tool using SAM 3.1 text prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
from PIL import Image

from schema import ArtifactKind, MaskArtifact, SegmentArgs, ToolInvocationRecord, ToolName
from vision_backends.sam3_point_backend import (
    Sam3BackendError,
    predict_text_prompt_candidates as sam31_predict_text_prompt_candidates,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


@dataclass(slots=True)
class SegmentCandidate:
    name: str
    mask: np.ndarray
    score: float
    source_stage: str
    mask_logits: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class SegmentTool:
    name = ToolName.SEGMENT

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

    def _predict_text_prompt_candidates(
        self,
        *,
        image_array: np.ndarray,
        text_prompt: str,
        source_stage: str,
    ) -> list[SegmentCandidate]:
        if not text_prompt.strip():
            return []
        raw_candidates = sam31_predict_text_prompt_candidates(
            image_array=image_array,
            text_prompt=text_prompt,
        )
        return [
            SegmentCandidate(
                name=str(item.get("name", f"{source_stage}_{index}")),
                mask=np.asarray(item.get("mask"), dtype=bool),
                mask_logits=(
                    np.asarray(item["mask_logits"], dtype=np.float32)
                    if item.get("mask_logits") is not None
                    else None
                ),
                score=float(item.get("score", 0.0)),
                source_stage=source_stage,
            )
            for index, item in enumerate(raw_candidates)
        ]

    def _is_sam31_text_prompt_backend_available(self, *, backend_name: str | None) -> bool:
        resolved_backend = (backend_name or "sam31").strip().lower()
        if resolved_backend != "sam31":
            return False
        checkpoint_path = os.getenv("SAM3_CHECKPOINT_PATH")
        if not checkpoint_path:
            return False
        return Path(checkpoint_path).is_file()

    def _bbox_from_mask(self, mask: np.ndarray) -> list[int] | None:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        if not binary.any():
            return binary
        height, width = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        best_coords: list[tuple[int, int]] = []

        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                coords: list[tuple[int, int]] = []
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(coords) > len(best_coords):
                    best_coords = coords

        result = np.zeros_like(binary, dtype=bool)
        for y, x in best_coords:
            result[y, x] = True
        return result

    def _postprocess_mask(self, *, mask: np.ndarray) -> np.ndarray:
        cleaned = self._largest_component(mask)
        if not cleaned.any():
            return cleaned
        return cleaned

    def _score_candidate(
        self,
        *,
        candidate: SegmentCandidate,
    ) -> dict[str, float] | None:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            return None
        mask_area = int(mask.sum())
        mask_bbox = self._bbox_from_mask(mask)
        if mask_bbox is None:
            return None
        cand_left, cand_top, cand_right, cand_bottom = mask_bbox
        largest_component_ratio = float(self._largest_component(mask).sum()) / float(mask_area)
        image_area = float(mask.shape[0] * mask.shape[1])
        mask_area_ratio_to_image = float(mask_area) / image_area
        aspect_ratio = float(cand_bottom - cand_top) / float(max(1, cand_right - cand_left))
        if mask_area_ratio_to_image < 0.01 or mask_area_ratio_to_image > 0.75:
            return None
        if largest_component_ratio < 0.7:
            return None
        if aspect_ratio < 0.35:
            return None

        final_score = (
            0.30 * float(candidate.score)
            + 0.40 * largest_component_ratio
            - 0.15 * abs(mask_area_ratio_to_image - 0.22)
            - 0.05 * abs(aspect_ratio - 1.15)
        )
        return {
            "final_score": final_score,
            "sam_score": float(candidate.score),
            "mask_area_ratio_to_image": mask_area_ratio_to_image,
            "largest_component_ratio": largest_component_ratio,
            "aspect_ratio": aspect_ratio,
        }

    def _select_best_candidate(
        self,
        *,
        candidates: list[SegmentCandidate],
    ) -> SegmentCandidate | None:
        ranked: list[SegmentCandidate] = []
        for candidate in candidates:
            mask = np.asarray(candidate.mask, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            metrics = self._score_candidate(candidate=candidate)
            if metrics is None:
                continue
            candidate.metrics = metrics
            ranked.append(candidate)
        if not ranked:
            return None
        ranked.sort(key=lambda item: item.metrics["final_score"], reverse=True)
        return ranked[0]

    def _write_mask(self, *, mask: np.ndarray, task_id: str, loop_index: int) -> str:
        output_dir = Path("generated") / "segment"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{loop_index:03d}_mask.png"
        mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
        Image.fromarray(mask_uint8).save(output_path)
        return str(output_path)

    def run(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: SegmentArgs,
    ) -> ToolExecutionResult:
        image_path = self._resolve_local_image_path(state, args.image_ref)
        image_array = np.asarray(Image.open(Path(image_path)).convert("RGB"), dtype=np.uint8)
        text_prompt = str(args.prompt).strip()
        if not text_prompt:
            raise ValueError("segment requires a non-empty prompt")

        if not self._is_sam31_text_prompt_backend_available(
            backend_name=args.backend_name
        ):
            raise ValueError(
                "segment only supports the SAM 3.1 text-prompt path; set "
                "backend_name='sam31' and provide a valid SAM3_CHECKPOINT_PATH"
            )

        try:
            text_prompt_candidates = self._predict_text_prompt_candidates(
                image_array=image_array,
                text_prompt=text_prompt,
                source_stage="sam_text_only",
            )
        except Sam3BackendError as exc:
            raise ValueError(
                "segment failed while running the SAM 3.1 text-prompt backend"
            ) from exc

        final_candidate = self._select_best_candidate(
            candidates=text_prompt_candidates,
        )
        if final_candidate is None:
            raise ValueError(
                "segment could not find an acceptable SAM 3.1 text-prompt mask candidate"
            )
        final_candidate.mask = self._postprocess_mask(mask=final_candidate.mask)

        mask_path = self._write_mask(
            mask=final_candidate.mask,
            task_id=task_id,
            loop_index=loop_index,
        )

        artifact = MaskArtifact(
            id=next_artifact_id(state, ArtifactKind.MASK),
            uri=mask_path,
            payload={
                "image_ref": args.image_ref,
                "prompt": text_prompt,
                "mask_score": final_candidate.score,
                "selection_metrics": final_candidate.metrics,
                "source_stage": final_candidate.source_stage,
                "text_prompt": text_prompt,
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
                "prompt": text_prompt,
                "mask_score": final_candidate.score,
                "mask_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
