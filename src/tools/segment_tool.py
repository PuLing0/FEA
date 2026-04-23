"""Segmentation tool with grounding-guided mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from schema import GroundingPoint
from schema import MaskArtifact, SegmentArgs, ToolInvocationRecord, ToolName
from vision_backends.grabcut_refinement import (
    GrabCutRefinementError,
    RefinementSeedCandidate,
    refine_candidates as grabcut_refine_candidates,
)
from vision_backends.sam3_point_backend import (
    Sam3BackendError,
    predict_candidates as sam31_predict_candidates,
)

from .base import ToolExecutionResult
from .utils import next_artifact_id, next_operation_id
from schema import ArtifactKind


@dataclass(slots=True)
class SegmentCandidate:
    name: str
    mask: np.ndarray
    score: float


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

    def _resolve_grounding_payload(self, state, grounding_ref: str, image_ref: str) -> dict:
        artifact = state["artifacts"].get(grounding_ref)
        if artifact is None:
            raise ValueError(f"Unknown grounding artifact ref: {grounding_ref}")
        if artifact.kind != ArtifactKind.GEOMETRY:
            raise ValueError(f"Artifact is not geometry: {grounding_ref}")
        if artifact.payload.get("image_artifact_id") != image_ref:
            raise ValueError("grounding artifact does not belong to the requested image")
        candidates = artifact.payload.get("candidates", [])
        if not candidates:
            raise ValueError("grounding artifact has no candidates")
        return artifact.payload

    def _predict_candidates(
        self,
        *,
        image_array: np.ndarray,
        positive_points: list[GroundingPoint],
        negative_points: list[GroundingPoint],
        bbox: list[int],
        backend_name: str | None,
    ) -> list[SegmentCandidate]:
        resolved_backend = (backend_name or "sam31").strip().lower()
        if resolved_backend != "sam31":
            raise ValueError(f"Unsupported segment backend: {backend_name}")
        return self._predict_candidates_via_sam31(
            image_array=image_array,
            positive_points=positive_points,
            negative_points=negative_points,
            bbox=bbox,
        )

    def _predict_candidates_via_sam31(
        self,
        *,
        image_array: np.ndarray,
        positive_points: list[GroundingPoint],
        negative_points: list[GroundingPoint],
        bbox: list[int],
    ) -> list[SegmentCandidate]:
        left, top, right, bottom = bbox
        height, width = image_array.shape[:2]
        roi_positive = [
            point
            for point in positive_points
            if left <= point.x < right and top <= point.y < bottom
        ] or positive_points
        roi_negative = [
            point
            for point in negative_points
            if left <= point.x < right and top <= point.y < bottom
        ]
        try:
            raw_candidates = sam31_predict_candidates(
                image_array=image_array,
                positive_points=roi_positive,
                negative_points=roi_negative,
            )
        except Sam3BackendError:
            raise
        return [
            SegmentCandidate(
                name=str(item.get("name", f"sam31_{index}")),
                mask=np.asarray(item.get("mask"), dtype=bool),
                score=float(item.get("score", 0.0)),
            )
            for index, item in enumerate(raw_candidates)
        ]

    def _refine_candidates_with_grabcut(
        self,
        *,
        image_array: np.ndarray,
        candidates: list[SegmentCandidate],
        positive_points: list[GroundingPoint],
        negative_points: list[GroundingPoint],
    ) -> list[SegmentCandidate]:
        try:
            refined_raw = grabcut_refine_candidates(
                image_array=image_array,
                positive_points=positive_points,
                negative_points=negative_points,
                seed_candidates=[
                    RefinementSeedCandidate(
                        name=item.name,
                        mask=np.asarray(item.mask, dtype=bool),
                        score=float(item.score),
                    )
                    for item in candidates
                ],
            )
        except GrabCutRefinementError:
            return candidates
        if not refined_raw:
            return candidates
        refined: list[SegmentCandidate] = []
        for index, item in enumerate(refined_raw):
            refined.append(
                SegmentCandidate(
                    name=str(item.get("name", f"grabcut_{index}")),
                    mask=np.asarray(item.get("mask"), dtype=bool),
                    score=float(item.get("score", 0.0)),
                )
            )
        return refined or candidates

    def _select_best_candidate(
        self,
        *,
        candidates: list[SegmentCandidate],
        positive_points: list[GroundingPoint],
        negative_points: list[GroundingPoint],
    ) -> SegmentCandidate:
        valid: list[SegmentCandidate] = []
        for candidate in candidates:
            mask = np.asarray(candidate.mask, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            if not all(
                0 <= int(point.x) < mask.shape[1]
                and 0 <= int(point.y) < mask.shape[0]
                and mask[int(point.y), int(point.x)]
                for point in positive_points
            ):
                continue
            if any(
                0 <= int(point.x) < mask.shape[1]
                and 0 <= int(point.y) < mask.shape[0]
                and mask[int(point.y), int(point.x)]
                for point in negative_points
            ):
                continue
            valid.append(candidate)
        if not valid:
            raise ValueError("No candidate mask satisfied grounding point constraints")
        valid.sort(key=lambda item: (-item.score, int(np.asarray(item.mask).sum())))
        return valid[0]

    def _write_mask(self, *, mask: np.ndarray, task_id: str, loop_index: int) -> str:
        output_dir = Path("generated") / "segment"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{loop_index:03d}_mask.png"
        mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
        Image.fromarray(mask_uint8, mode="L").save(output_path)
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
        grounding_payload = self._resolve_grounding_payload(
            state,
            args.grounding_ref,
            args.image_ref,
        )
        first_candidate = grounding_payload["candidates"][0]
        bbox = list(first_candidate["bbox"])
        positive_points = [
            GroundingPoint.model_validate(item)
            for item in first_candidate.get("positive_points", [])
        ]
        negative_points = [
            GroundingPoint.model_validate(item)
            for item in first_candidate.get("negative_points", [])
        ]

        image_array = np.asarray(Image.open(Path(image_path)).convert("RGB"), dtype=np.uint8)
        candidates = self._predict_candidates(
            image_array=image_array,
            positive_points=positive_points,
            negative_points=negative_points,
            bbox=bbox,
            backend_name=args.backend_name,
        )
        candidates = self._refine_candidates_with_grabcut(
            image_array=image_array,
            candidates=candidates,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        selected = self._select_best_candidate(
            candidates=candidates,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        mask_path = self._write_mask(
            mask=selected.mask,
            task_id=task_id,
            loop_index=loop_index,
        )

        artifact = MaskArtifact(
            id=next_artifact_id(state, ArtifactKind.MASK),
            uri=mask_path,
            payload={
                "image_ref": args.image_ref,
                "grounding_ref": args.grounding_ref,
                "target": args.target,
                "positive_points": [point.model_dump() for point in positive_points],
                "negative_points": [point.model_dump() for point in negative_points],
                "mask_score": selected.score,
            },
            source_ids=[args.image_ref, args.grounding_ref],
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
                "grounding_ref": args.grounding_ref,
                "target": args.target,
                "mask_score": selected.score,
                "mask_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
