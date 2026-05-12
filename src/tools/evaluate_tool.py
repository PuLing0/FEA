"""Structured multimodal evaluate tool implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import invoke_structured_multimodal_llm
from PIL import Image, ImageDraw, ImageFont
from runtime.instruction_resolver import resolve_active_instruction_text
from runtime.prompts import EVALUATE_SYSTEM_PROMPT, build_evaluate_user_prompt
from schema import (
    ArtifactKind,
    EvaluateArgs,
    EvaluateLLMOutput,
    EvaluationArtifact,
    ToolInvocationRecord,
    ToolName,
)

from .base import BaseTool, ToolExecutionResult
from .utils import next_artifact_id, next_operation_id


REFERENCE_BOARD_MAX_SIDE = 2048
SUPPORTED_VERDICTS = {"pass", "pass_with_issues", "needs_revision", "replan"}


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
    ) -> tuple[str | None, list[str]]:
        if not input_refs:
            return None, []
        input_paths = self._resolve_image_paths(state, input_refs, label="input")
        if len(input_paths) == 1:
            return str(input_paths[0]), []

        images = []
        for artifact_ref, path in zip(input_refs, input_paths, strict=True):
            with Image.open(path) as image:
                images.append((artifact_ref, image.convert("RGB")))

        cell_width = max(image.width for _, image in images)
        cell_height = max(image.height for _, image in images)
        label_height = 36
        columns = 2 if len(images) <= 4 else 3
        rows = (len(images) + columns - 1) // columns
        board = Image.new(
            "RGB",
            (columns * cell_width, rows * (cell_height + label_height)),
            color=(245, 245, 245),
        )
        draw = ImageDraw.Draw(board)
        font = ImageFont.load_default()

        for index, (artifact_ref, image) in enumerate(images):
            row = index // columns
            col = index % columns
            x = col * cell_width
            y = row * (cell_height + label_height)
            board.paste(image, (x, y + label_height))
            draw.rectangle((x, y, x + cell_width, y + label_height), fill=(230, 230, 230))
            draw.text((x + 8, y + 10), artifact_ref, fill=(0, 0, 0), font=font)

        max_side = max(board.size)
        if max_side > REFERENCE_BOARD_MAX_SIDE:
            scale = REFERENCE_BOARD_MAX_SIDE / float(max_side)
            board = board.resize(
                (int(board.width * scale), int(board.height * scale)),
                Image.Resampling.LANCZOS,
            )

        output_dir = Path("generated") / "evaluate"
        output_dir.mkdir(parents=True, exist_ok=True)
        board_path = output_dir / f"{task_id}_{loop_index:03d}_reference_board.png"
        board.save(board_path)
        return str(board_path), [str(board_path)]

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
        reference_board_path: str | None,
        candidate_path: str,
        instruction: str,
        checks: list[str],
        input_refs: list[str],
        candidate_ref: str,
    ) -> EvaluateLLMOutput:
        image_paths = [path for path in [reference_board_path, candidate_path] if path]
        reference_text = (
            "Image 1 is a reference board containing all task input/reference images. "
            "Image 2 is the candidate edited result."
            if reference_board_path
            else "Only one image is provided: the candidate edited result."
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
        input_refs = self._resolve_input_refs(state, task_id, args)
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
        invocation = ToolInvocationRecord(
            id=next_operation_id(state, self.name),
            task_id=task_id,
            loop_index=loop_index,
            tool_name=self.name,
            args=args.model_dump(),
            status="succeeded",
            output_refs=[artifact.id],
            result_payload={**payload, "evaluation_ref": artifact.id, "generated_refs": generated_refs},
            raw_output_uri=f"runs/{task_id}/{self.name.value}.json",
        )
        return ToolExecutionResult(invocation=invocation, artifacts=[artifact])
