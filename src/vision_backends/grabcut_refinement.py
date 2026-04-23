"""GrabCut-based mask refinement for point-guided segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from schema import GroundingPoint


class GrabCutRefinementError(RuntimeError):
    """Raised when GrabCut refinement cannot be initialized or executed."""


@dataclass(slots=True)
class RefinementSeedCandidate:
    name: str
    mask: np.ndarray
    score: float


@dataclass(slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int


def refine_candidates(
    *,
    image_array: np.ndarray,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
    seed_candidates: list[RefinementSeedCandidate],
) -> list[dict[str, object]]:
    if not positive_points:
        return []

    cv2 = _load_cv2()
    image = np.asarray(image_array, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise GrabCutRefinementError(
            "GrabCut refinement expects an RGB image array with shape [H, W, 3]."
        )

    candidates: list[dict[str, object]] = []
    prompt_only = _run_single_refinement(
        cv2=cv2,
        image=image,
        positive_points=positive_points,
        negative_points=negative_points,
        seed_candidate=None,
        score_bonus=0.12,
    )
    if prompt_only is not None:
        candidates.append(prompt_only)

    for seed_candidate in seed_candidates[:3]:
        refined = _run_single_refinement(
            cv2=cv2,
            image=image,
            positive_points=positive_points,
            negative_points=negative_points,
            seed_candidate=seed_candidate,
            score_bonus=0.08,
        )
        if refined is not None:
            candidates.append(refined)

    return _deduplicate_candidates(candidates)


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception as exc:
        raise GrabCutRefinementError(
            "GrabCut refinement requires OpenCV (cv2) in the active environment."
        ) from exc


def _run_single_refinement(
    *,
    cv2,
    image: np.ndarray,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
    seed_candidate: RefinementSeedCandidate | None,
    score_bonus: float,
) -> dict[str, object] | None:
    image_height, image_width = image.shape[:2]
    prompt_roi_bbox = _build_prompt_roi_bbox(
        image_width=image_width,
        image_height=image_height,
        positive_points=positive_points,
        negative_points=negative_points,
    )
    if prompt_roi_bbox is None:
        return None
    seed_mask = (
        _normalize_seed_mask(seed_candidate.mask, image.shape[:2])
        if seed_candidate is not None
        else None
    )
    usable_seed_mask = _prepare_seed_mask_for_prompt_roi(
        seed_mask=seed_mask,
        prompt_roi_bbox=prompt_roi_bbox,
    )
    roi_bbox = _merge_prompt_and_seed_roi(
        prompt_roi_bbox=prompt_roi_bbox,
        seed_mask=usable_seed_mask,
        image_width=image_width,
        image_height=image_height,
    )

    crop = image[roi_bbox.top:roi_bbox.bottom, roi_bbox.left:roi_bbox.right]
    local_positive = _shift_points(positive_points, roi_bbox)
    local_negative = _shift_points(negative_points, roi_bbox)
    local_seed_mask = (
        usable_seed_mask[roi_bbox.top:roi_bbox.bottom, roi_bbox.left:roi_bbox.right]
        if usable_seed_mask is not None
        else None
    )

    refined_mask = _run_grabcut_on_crop(
        cv2=cv2,
        crop=crop,
        positive_points=local_positive,
        negative_points=local_negative,
        seed_mask=local_seed_mask,
    )
    if refined_mask is None or not refined_mask.any():
        return None

    full_mask = np.zeros((image_height, image_width), dtype=bool)
    full_mask[roi_bbox.top:roi_bbox.bottom, roi_bbox.left:roi_bbox.right] = refined_mask
    if not full_mask.any():
        return None

    base_score = float(seed_candidate.score) if seed_candidate is not None else 0.42
    name = "grabcut_prompt" if seed_candidate is None else f"grabcut_refined_{seed_candidate.name}"
    return {
        "name": name,
        "mask": full_mask,
        "score": min(1.0, max(0.0, base_score + score_bonus)),
    }


def _normalize_seed_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    normalized = np.asarray(mask, dtype=bool)
    if normalized.ndim != 2:
        raise GrabCutRefinementError("GrabCut seed masks must be 2D.")
    if normalized.shape != expected_shape:
        raise GrabCutRefinementError("GrabCut seed mask shape does not match the source image.")
    return normalized


def _build_prompt_roi_bbox(
    *,
    image_width: int,
    image_height: int,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
) -> BoundingBox | None:
    boxes: list[BoundingBox] = []
    point_box = _bbox_from_points(positive_points)
    if point_box is not None:
        boxes.append(point_box)
    negative_box = _bbox_from_points(negative_points)
    if negative_box is not None:
        boxes.append(negative_box)
    if not boxes:
        return None

    roi = _union_boxes(boxes)
    positive_span = _box_span(point_box) if point_box is not None else 0
    roi_span = _box_span(roi)
    padding = max(28, min(160, int(max(positive_span * 0.9 + 20, roi_span * 0.35 + 20))))
    return _expand_box(roi, padding=padding, width=image_width, height=image_height)


def _prepare_seed_mask_for_prompt_roi(
    *,
    seed_mask: np.ndarray | None,
    prompt_roi_bbox: BoundingBox,
) -> np.ndarray | None:
    if seed_mask is None or not seed_mask.any():
        return None
    clipped_seed_mask = np.zeros_like(seed_mask, dtype=bool)
    clipped_seed_mask[
        prompt_roi_bbox.top:prompt_roi_bbox.bottom,
        prompt_roi_bbox.left:prompt_roi_bbox.right,
    ] = seed_mask[
        prompt_roi_bbox.top:prompt_roi_bbox.bottom,
        prompt_roi_bbox.left:prompt_roi_bbox.right,
    ]
    if not clipped_seed_mask.any():
        return None

    prompt_area = max(
        1,
        (prompt_roi_bbox.right - prompt_roi_bbox.left)
        * (prompt_roi_bbox.bottom - prompt_roi_bbox.top),
    )
    if float(clipped_seed_mask.sum()) / float(prompt_area) >= 0.75:
        return None
    return clipped_seed_mask


def _merge_prompt_and_seed_roi(
    *,
    prompt_roi_bbox: BoundingBox,
    seed_mask: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    seed_box = _bbox_from_mask(seed_mask) if seed_mask is not None else None
    if seed_box is None:
        return prompt_roi_bbox
    merged = _union_boxes([prompt_roi_bbox, seed_box])
    return _expand_box(merged, padding=16, width=image_width, height=image_height)


def _run_grabcut_on_crop(
    *,
    cv2,
    crop: np.ndarray,
    positive_points: list[GroundingPoint],
    negative_points: list[GroundingPoint],
    seed_mask: np.ndarray | None,
) -> np.ndarray | None:
    crop_height, crop_width = crop.shape[:2]
    if crop_height < 2 or crop_width < 2:
        return None

    mask = np.full((crop_height, crop_width), cv2.GC_PR_BGD, dtype=np.uint8)
    likely_fg_box = _build_likely_foreground_box(
        width=crop_width,
        height=crop_height,
        positive_points=positive_points,
        seed_mask=seed_mask,
    )
    mask[likely_fg_box.top:likely_fg_box.bottom, likely_fg_box.left:likely_fg_box.right] = cv2.GC_PR_FGD

    if seed_mask is not None and seed_mask.any():
        mask[seed_mask] = cv2.GC_PR_FGD
        sure_seed_mask = _erode_binary_mask(cv2=cv2, mask=seed_mask, kernel_size=5, iterations=1)
        if sure_seed_mask.any():
            mask[sure_seed_mask] = cv2.GC_FGD

    fg_radius = _point_radius(width=crop_width, height=crop_height, scale=0.045, minimum=8, maximum=18)
    bg_radius = _point_radius(width=crop_width, height=crop_height, scale=0.05, minimum=10, maximum=22)
    _draw_point_disks(cv2=cv2, canvas=mask, points=positive_points, radius=fg_radius, value=cv2.GC_FGD)
    _draw_point_disks(cv2=cv2, canvas=mask, points=negative_points, radius=bg_radius, value=cv2.GC_BGD)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, None, bg_model, fg_model, 6, cv2.GC_INIT_WITH_MASK)

    result_mask = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return result_mask if result_mask.any() else None


def _bbox_from_points(points: list[GroundingPoint]) -> BoundingBox | None:
    if not points:
        return None
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return BoundingBox(left=min(xs), top=min(ys), right=max(xs) + 1, bottom=max(ys) + 1)


def _bbox_from_mask(mask: np.ndarray | None) -> BoundingBox | None:
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return BoundingBox(left=int(xs.min()), top=int(ys.min()), right=int(xs.max()) + 1, bottom=int(ys.max()) + 1)


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        right=max(box.right for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def _box_span(box: BoundingBox) -> int:
    return max(box.right - box.left, box.bottom - box.top)


def _expand_box(box: BoundingBox, *, padding: int, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        left=max(0, box.left - padding),
        top=max(0, box.top - padding),
        right=min(width, box.right + padding),
        bottom=min(height, box.bottom + padding),
    )


def _shift_points(points: list[GroundingPoint], bbox: BoundingBox) -> list[GroundingPoint]:
    return [GroundingPoint(x=point.x - bbox.left, y=point.y - bbox.top) for point in points]


def _build_likely_foreground_box(
    *,
    width: int,
    height: int,
    positive_points: list[GroundingPoint],
    seed_mask: np.ndarray | None,
) -> BoundingBox:
    point_box = _bbox_from_points(positive_points)
    seed_box = _bbox_from_mask(seed_mask) if seed_mask is not None else None
    if point_box is not None and seed_box is not None:
        base = _union_boxes([point_box, seed_box])
    else:
        base = point_box or seed_box or BoundingBox(0, 0, width, height)
    padding = max(8, min(24, int(_box_span(base) * 0.2 + 8)))
    return _expand_box(base, padding=padding, width=width, height=height)


def _point_radius(*, width: int, height: int, scale: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, max(width, height) * scale)))


def _draw_point_disks(*, cv2, canvas: np.ndarray, points: list[GroundingPoint], radius: int, value: int) -> None:
    for point in points:
        cv2.circle(canvas, (int(point.x), int(point.y)), radius, value, thickness=-1)


def _erode_binary_mask(*, cv2, mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded.astype(bool)


def _deduplicate_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for item in candidates:
        mask = np.asarray(item.get("mask"), dtype=bool)
        if mask.ndim != 2 or not mask.any():
            continue
        signature = mask.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(
            {
                "name": str(item.get("name", "candidate")),
                "mask": mask,
                "score": float(item.get("score", 0.0)),
            }
        )
    return unique


__all__ = ["GrabCutRefinementError", "RefinementSeedCandidate", "refine_candidates"]
