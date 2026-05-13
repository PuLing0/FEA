"""Structured multimodal evaluate tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm import invoke_structured_multimodal_llm
from PIL import Image, ImageDraw, ImageFont, ImageOps
from runtime.instruction_resolver import resolve_active_instruction_text
from runtime.prompts import EVALUATE_SYSTEM_PROMPT, build_evaluate_user_prompt
from schema import (
    ArtifactKind,
    EvaluateArgs,
    EvaluateLLMOutput,
    EvaluationArtifact,
    ToolName,
)

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id, tool_artifact_dir


REFERENCE_BOARD_MAX_SIDE = 2048
REFERENCE_BOARD_PADDING = 8
REFERENCE_BOARD_LAYOUT_LIMIT = 48
SUPPORTED_VERDICTS = {"pass", "pass_with_issues", "needs_revision", "replan"}


@dataclass(frozen=True)
class _BoardImage:
    artifact_ref: str
    image: Image.Image


@dataclass(frozen=True)
class _BoardPlacement:
    index: int
    x: int
    y: int


@dataclass(frozen=True)
class _BoardLayout:
    width: int
    height: int
    placements: tuple[_BoardPlacement, ...]


class EvaluateTool(BaseTool):
    name = ToolName.EVALUATE
    args_schema = EvaluateArgs
    is_expensive = True

    def _resolve_image_artifact(self, state, artifact_ref: str, *, label: str):
        artifact = state["artifacts"].get(artifact_ref)
        if artifact is None:
            raise ValueError(f"Unknown {label} artifact ref: {artifact_ref}")
        if artifact.kind != ArtifactKind.IMAGE:
            raise ValueError(f"{label.capitalize()} artifact is not an image: {artifact_ref}")
        if not artifact.uri:
            raise ValueError(f"{label.capitalize()} image has no uri: {artifact_ref}")
        if "://" in artifact.uri:
            raise ValueError(
                f"{label.capitalize()} image uri is not a local file path: {artifact.uri}"
            )
        path = Path(artifact.uri)
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {artifact.uri}")
        return artifact, path

    def _resolve_image_paths(self, state, refs: list[str], *, label: str) -> list[Path]:
        return [
            self._resolve_image_artifact(state, artifact_ref, label=label)[1]
            for artifact_ref in refs
        ]

    def _build_reference_board(
        self,
        *,
        state,
        input_refs: list[str],
        task_id: str,
        loop_index: int,
    ) -> tuple[str, list[str]]:
        if not input_refs:
            raise ValueError("evaluate requires at least one input/reference image")
        input_paths = self._resolve_image_paths(state, input_refs, label="input")

        images: list[_BoardImage] = []
        for artifact_ref, path in zip(input_refs, input_paths, strict=True):
            with Image.open(path) as image:
                images.append(
                    _BoardImage(
                        artifact_ref=artifact_ref,
                        image=ImageOps.exif_transpose(image).convert("RGB"),
                    )
                )

        layout = self._build_compact_board_layout(images)
        board = Image.new("RGB", (layout.width, layout.height), color=(245, 245, 245))
        for placement in layout.placements:
            board_image = images[placement.index]
            board.paste(board_image.image, (placement.x, placement.y))
        self._draw_reference_labels(board, images=images, layout=layout)

        max_side = max(board.size)
        if max_side > REFERENCE_BOARD_MAX_SIDE:
            scale = REFERENCE_BOARD_MAX_SIDE / float(max_side)
            board = board.resize(
                (int(board.width * scale), int(board.height * scale)),
                Image.Resampling.LANCZOS,
            )

        output_dir = tool_artifact_dir(state, self.name)
        board_path = output_dir / f"{task_id}_{loop_index:03d}_reference_board.png"
        board.save(board_path)
        return str(board_path), [str(board_path)]

    @classmethod
    def _build_compact_board_layout(cls, images: list[_BoardImage]) -> _BoardLayout:
        dimensions = tuple((item.image.width, item.image.height) for item in images)
        if len(dimensions) == 1:
            width, height = dimensions[0]
            return _BoardLayout(
                width=width,
                height=height,
                placements=(_BoardPlacement(index=0, x=0, y=0),),
            )
        if len(dimensions) > 8:
            return cls._build_shelf_board_layout(dimensions)

        full_mask = (1 << len(dimensions)) - 1

        @lru_cache(maxsize=None)
        def best_layouts(mask: int) -> tuple[_BoardLayout, ...]:
            if mask and mask & (mask - 1) == 0:
                index = mask.bit_length() - 1
                width, height = dimensions[index]
                return (
                    _BoardLayout(
                        width=width,
                        height=height,
                        placements=(_BoardPlacement(index=index, x=0, y=0),),
                    ),
                )

            candidates: list[_BoardLayout] = []
            submask = (mask - 1) & mask
            while submask:
                other_mask = mask ^ submask
                if submask < other_mask:
                    for left in best_layouts(submask):
                        for right in best_layouts(other_mask):
                            candidates.append(cls._combine_layouts(left, right, horizontal=True))
                            candidates.append(cls._combine_layouts(left, right, horizontal=False))
                submask = (submask - 1) & mask
            return tuple(cls._trim_layout_candidates(candidates))

        return min(best_layouts(full_mask), key=cls._layout_score)

    @classmethod
    def _build_shelf_board_layout(cls, dimensions: tuple[tuple[int, int], ...]) -> _BoardLayout:
        order = sorted(
            range(len(dimensions)),
            key=lambda index: dimensions[index][0] * dimensions[index][1],
            reverse=True,
        )
        best: _BoardLayout | None = None
        for columns in range(1, len(order) + 1):
            placements: list[_BoardPlacement] = []
            y = 0
            width = 0
            row_specs: list[tuple[list[int], int, int]] = []
            for offset in range(0, len(order), columns):
                row = order[offset : offset + columns]
                row_width = sum(dimensions[index][0] for index in row) + REFERENCE_BOARD_PADDING * (
                    len(row) - 1
                )
                row_height = max(dimensions[index][1] for index in row)
                row_specs.append((row, row_width, row_height))
                width = max(width, row_width)
            height = sum(row_height for _, _, row_height in row_specs) + REFERENCE_BOARD_PADDING * (
                len(row_specs) - 1
            )
            for row, row_width, row_height in row_specs:
                x = (width - row_width) // 2
                for index in row:
                    item_width, item_height = dimensions[index]
                    placements.append(
                        _BoardPlacement(index=index, x=x, y=y + (row_height - item_height) // 2)
                    )
                    x += item_width + REFERENCE_BOARD_PADDING
                y += row_height + REFERENCE_BOARD_PADDING
            candidate = _BoardLayout(width=width, height=height, placements=tuple(placements))
            if best is None or cls._layout_score(candidate) < cls._layout_score(best):
                best = candidate
        assert best is not None
        return best

    @staticmethod
    def _combine_layouts(left: _BoardLayout, right: _BoardLayout, *, horizontal: bool) -> _BoardLayout:
        if horizontal:
            width = left.width + REFERENCE_BOARD_PADDING + right.width
            height = max(left.height, right.height)
            left_dx = 0
            left_dy = (height - left.height) // 2
            right_dx = left.width + REFERENCE_BOARD_PADDING
            right_dy = (height - right.height) // 2
        else:
            width = max(left.width, right.width)
            height = left.height + REFERENCE_BOARD_PADDING + right.height
            left_dx = (width - left.width) // 2
            left_dy = 0
            right_dx = (width - right.width) // 2
            right_dy = left.height + REFERENCE_BOARD_PADDING
        placements = tuple(
            [
                *(
                    _BoardPlacement(index=item.index, x=item.x + left_dx, y=item.y + left_dy)
                    for item in left.placements
                ),
                *(
                    _BoardPlacement(index=item.index, x=item.x + right_dx, y=item.y + right_dy)
                    for item in right.placements
                ),
            ]
        )
        return _BoardLayout(width=width, height=height, placements=placements)

    @classmethod
    def _trim_layout_candidates(cls, layouts: list[_BoardLayout]) -> list[_BoardLayout]:
        unique: dict[tuple[int, int], _BoardLayout] = {}
        for layout in sorted(layouts, key=cls._layout_score):
            unique.setdefault((layout.width, layout.height), layout)
            if len(unique) >= REFERENCE_BOARD_LAYOUT_LIMIT:
                break
        return list(unique.values())

    @staticmethod
    def _layout_score(layout: _BoardLayout) -> tuple[int, int, int]:
        return (
            layout.width * layout.height,
            max(layout.width, layout.height),
            abs(layout.width - layout.height),
        )

    @staticmethod
    def _draw_reference_labels(
        board: Image.Image,
        *,
        images: list[_BoardImage],
        layout: _BoardLayout,
    ) -> None:
        font = ImageFont.load_default()
        label_height = 16
        for placement in layout.placements:
            board_image = images[placement.index]
            label_width = min(
                board_image.image.width,
                max(36, min(160, len(board_image.artifact_ref) * 7 + 8)),
            )
            label_layer = Image.new(
                "RGBA",
                (label_width, min(label_height, board_image.image.height)),
                (0, 0, 0, 150),
            )
            draw = ImageDraw.Draw(label_layer)
            draw.text(
                (4, 3),
                board_image.artifact_ref,
                fill=(255, 255, 255, 255),
                font=font,
            )
            board.paste(label_layer, (placement.x, placement.y), label_layer)

    @staticmethod
    def _derive_verdict(
        *,
        llm_verdict: str | None,
    ) -> str:
        verdict = llm_verdict if llm_verdict in SUPPORTED_VERDICTS else None
        return verdict or "replan"

    def _evaluate_with_llm(
        self,
        *,
        reference_board_path: str,
        candidate_path: str,
        instruction: str,
        checks: list[str],
        input_refs: list[str],
        candidate_ref: str,
    ) -> EvaluateLLMOutput:
        image_paths = [reference_board_path, candidate_path]
        reference_text = (
            "Image 1 is a compact reference board containing the task input/reference images. "
            "Image 2 is the candidate edited output to evaluate."
        )
        return invoke_structured_multimodal_llm(
            system_prompt=EVALUATE_SYSTEM_PROMPT,
            user_prompt=build_evaluate_user_prompt(
                reference_text=reference_text,
                input_refs=input_refs,
                candidate_ref=candidate_ref,
                instruction=instruction,
                checks=checks,
            ),
            image_paths=image_paths,
            output_schema=EvaluateLLMOutput,
        )

    def _resolve_instruction(self, state, task_id: str, args: EvaluateArgs) -> str:
        if args.instruction:
            return args.instruction
        return resolve_active_instruction_text(state, task_id)

    def _resolve_input_refs(self, state, task_id: str, args: EvaluateArgs) -> list[str]:
        if args.input_refs:
            return list(args.input_refs)
        task_state = state["session"].task_states.get(task_id)
        if task_state and task_state.resolved_input_artifact_ids:
            return [
                artifact_id
                for artifact_id in task_state.resolved_input_artifact_ids
                if state["artifacts"].get(artifact_id) is not None
                and state["artifacts"][artifact_id].kind == ArtifactKind.IMAGE
            ]
        task = state.get("tasks", {}).get(task_id)
        if task:
            return [
                artifact_id
                for artifact_id in task.input_artifact_ids
                if state["artifacts"].get(artifact_id) is not None
                and state["artifacts"][artifact_id].kind == ArtifactKind.IMAGE
            ]
        return []

    def execute(
        self,
        state,
        *,
        task_id: str,
        loop_index: int,
        args: EvaluateArgs,
    ) -> ToolExecutionResult:
        candidate_ref = args.candidate_ref
        assert candidate_ref is not None
        _, candidate_path = self._resolve_image_artifact(state, candidate_ref, label="candidate")
        candidate_refs = set(args.candidate_refs or [candidate_ref])
        input_refs = [
            artifact_ref
            for artifact_ref in self._resolve_input_refs(state, task_id, args)
            if artifact_ref not in candidate_refs
        ]
        instruction = self._resolve_instruction(state, task_id, args)
        reference_board_path, generated_refs = self._build_reference_board(
            state=state,
            input_refs=input_refs,
            task_id=task_id,
            loop_index=loop_index,
        )
        llm_output = self._evaluate_with_llm(
            reference_board_path=reference_board_path,
            candidate_path=str(candidate_path),
            instruction=instruction,
            checks=args.checks,
            input_refs=input_refs,
            candidate_ref=candidate_ref,
        )
        verdict = self._derive_verdict(
            llm_verdict=llm_output.verdict,
        )
        payload: dict[str, Any] = {
            "input_refs": input_refs,
            "candidate_ref": candidate_ref,
            "candidate_refs": list(args.candidate_refs),
            "reference_board_uri": reference_board_path,
            "instruction": instruction,
            "checks": args.checks,
            "verdict": verdict,
            "reason": llm_output.reason,
        }
        artifact = EvaluationArtifact(
            id=next_artifact_id(state, ArtifactKind.EVALUATION),
            summary=llm_output.reason,
            payload=payload,
            source_ids=[*input_refs, candidate_ref],
            created_by=self.name.value,
            role="evaluation_feedback",
            scope="task",
        )
        return ToolExecutionResult(
            artifacts=[artifact],
            result_payload={**payload, "evaluation_ref": artifact.id, "generated_refs": generated_refs},
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
