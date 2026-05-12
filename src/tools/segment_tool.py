"""Single-path segmentation tool using SAM 3.1 text prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from schema import ArtifactKind, MaskArtifact, SegmentArgs, ToolName
from vision_backends.sam3_point_backend import (
    Sam3BackendError,
    predict_text_prompt_candidates as sam31_predict_text_prompt_candidates,
)
from vision_backends.remote_client import (
    RemoteBackendError,
    request_sam31_segment,
    resolve_segment_backend,
)

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id


@dataclass(slots=True)
class SegmentCandidate:
    name: str
    mask: np.ndarray
    score: float
    source_stage: str
    mask_logits: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class SegmentTool(BaseTool):
    name = ToolName.SEGMENT
    args_schema = SegmentArgs
    is_expensive = True
    requires_backend = True

    _SEGMENT_PROMPT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("person", ("person", "face", "woman", "man", "girl", "boy", "subject", "model", "identity")),
        ("top", ("top", "shirt", "blouse", "jacket", "upper body", "upper-body")),
        ("skirt", ("skirt", "dress", "lower body", "lower-body")),
        ("background", ("background", "scene", "temple", "outdoor", "indoor", "location")),
    )

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

    def _normalize_prompt_text(self, prompt: str) -> str:
        return re.sub(r"\s+", " ", str(prompt).strip())

    def _build_prompt_candidates(self, prompt: str) -> list[str]:
        normalized_prompt = self._normalize_prompt_text(prompt)
        if not normalized_prompt:
            return []

        candidates: list[str] = []

        def add(candidate: str) -> None:
            cleaned = self._normalize_prompt_text(candidate)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        if len(normalized_prompt.split()) <= 8 and len(normalized_prompt) <= 64:
            add(normalized_prompt)

        lowered = normalized_prompt.lower()
        for canonical, keywords in self._SEGMENT_PROMPT_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                add(canonical)

        quoted_terms = re.findall(r"['\"]([^'\"]+)['\"]", normalized_prompt)
        for term in quoted_terms:
            if len(term.split()) <= 4:
                add(term)

        if not candidates:
            add("person")
            add("top")
            add("skirt")
            add("background")
        return candidates

    def _write_mask_file(self, *, mask: np.ndarray, output_path: str) -> str:
        mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
        Image.fromarray(mask_uint8).save(output_path)
        return output_path

    def _build_full_image_mask(self, *, image_path: str) -> np.ndarray:
        with Image.open(Path(image_path)).convert("RGB") as image:
            return np.ones((image.height, image.width), dtype=bool)

    def _build_full_image_mask_result(
        self,
        *,
        image_path: str,
        mask_path: str,
        prompt: str,
    ) -> tuple[str, float, dict[str, float | str], str, str]:
        self._write_mask_file(
            mask=self._build_full_image_mask(image_path=image_path),
            output_path=mask_path,
        )
        return (
            mask_path,
            0.0,
            {
                "final_score": 0.0,
                "fallback_reason": "full_image_mask_after_segment_prompt_failures",
            },
            "fallback_full_image",
            prompt,
        )

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

    def _build_mask_path(self, *, task_id: str, loop_index: int) -> str:
        output_dir = Path("generated") / "segment"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{task_id}_{loop_index:03d}_mask.png")

    def _is_retryable_prompt_failure(self, message: str) -> bool:
        normalized = str(message).lower()
        return (
            "no acceptable mask candidate" in normalized
            or "found no acceptable mask candidate" in normalized
            or "text_prompt must be non-empty" in normalized
        )

    def execute(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: SegmentArgs,
    ) -> ToolExecutionResult:
        image_path = self._resolve_local_image_path(state, args.image_ref)
        text_prompt = str(args.prompt).strip()
        if not text_prompt:
            raise ValueError("segment requires a non-empty prompt")
        prompt_candidates = self._build_prompt_candidates(text_prompt)
        if not prompt_candidates:
            raise ValueError("segment requires at least one usable prompt candidate")
        backend_name = resolve_segment_backend()
        if backend_name == "remote":
            requested_mask_path = self._build_mask_path(task_id=task_id, loop_index=loop_index)
            remote_error_messages: list[str] = []
            remote_result = None
            for prompt_candidate in prompt_candidates:
                try:
                    remote_result = request_sam31_segment(
                        image_path=image_path,
                        prompt=prompt_candidate,
                        output_path=requested_mask_path,
                    )
                    break
                except RemoteBackendError as exc:
                    remote_error_messages.append(str(exc))
                    if not self._is_retryable_prompt_failure(str(exc)):
                        raise ValueError(
                            f"segment failed while running the remote SAM 3.1 backend: {exc}"
                        ) from exc
            if remote_result is None:
                mask_path, mask_score, selection_metrics, source_stage, text_prompt = (
                    self._build_full_image_mask_result(
                        image_path=image_path,
                        mask_path=requested_mask_path,
                        prompt=prompt_candidates[0],
                    )
                )
                selection_metrics["remote_errors"] = " | ".join(remote_error_messages)
            else:
                mask_path = remote_result.mask_path
                mask_score = remote_result.mask_score
                selection_metrics = remote_result.selection_metrics
                source_stage = remote_result.source_stage
                text_prompt = remote_result.text_prompt
        elif backend_name == "local":
            image_array = np.asarray(Image.open(Path(image_path)).convert("RGB"), dtype=np.uint8)
            if not self._is_sam31_text_prompt_backend_available(
                backend_name=args.backend_name
            ):
                raise ValueError(
                    "segment only supports the SAM 3.1 text-prompt path; set "
                    "backend_name='sam31' and provide a valid SAM3_CHECKPOINT_PATH"
                )

            final_candidate = None
            local_error_messages: list[str] = []
            for prompt_candidate in prompt_candidates:
                try:
                    text_prompt_candidates = self._predict_text_prompt_candidates(
                        image_array=image_array,
                        text_prompt=prompt_candidate,
                        source_stage="sam_text_only",
                    )
                except Sam3BackendError as exc:
                    local_error_messages.append(str(exc))
                    continue

                final_candidate = self._select_best_candidate(
                    candidates=text_prompt_candidates,
                )
                if final_candidate is not None:
                    final_candidate.mask = self._postprocess_mask(mask=final_candidate.mask)
                    text_prompt = prompt_candidate
                    break

            if final_candidate is None:
                mask_path, mask_score, selection_metrics, source_stage, text_prompt = (
                    self._build_full_image_mask_result(
                        image_path=image_path,
                        mask_path=self._build_mask_path(task_id=task_id, loop_index=loop_index),
                        prompt=prompt_candidates[0],
                    )
                )
                if local_error_messages:
                    selection_metrics["local_errors"] = " | ".join(local_error_messages)
            else:
                mask_path = self._write_mask(
                    mask=final_candidate.mask,
                    task_id=task_id,
                    loop_index=loop_index,
                )
                mask_score = final_candidate.score
                selection_metrics = final_candidate.metrics
                source_stage = final_candidate.source_stage
        else:
            raise ValueError("SEGMENT_BACKEND must be 'local' or 'remote'")

        if not Path(mask_path).is_file():
            raise FileNotFoundError(f"segment backend output path does not exist: {mask_path}")

        artifact = MaskArtifact(
            id=next_artifact_id(state, ArtifactKind.MASK),
            uri=mask_path,
            payload={
                "image_ref": args.image_ref,
                "prompt": text_prompt,
                "mask_score": mask_score,
                "selection_metrics": selection_metrics,
                "source_stage": source_stage,
                "text_prompt": text_prompt,
            },
            source_ids=[args.image_ref],
            created_by=self.name.value,
            scope="task",
        )
        return ToolExecutionResult(
            artifacts=[artifact],
            result_payload={
                "image_ref": args.image_ref,
                "prompt": text_prompt,
                "mask_score": mask_score,
                "mask_ref": artifact.id,
            },
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
