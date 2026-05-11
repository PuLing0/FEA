from __future__ import annotations

import pytest

from tests.regression.common import (
    EvaluationArtifact,
    ImageArtifact,
    MaskArtifact,
    ToolInvocationRecord,
    ToolName,
    UnderstandingArtifact,
)


REAL_TOOL_TESTS = {
    "test_grounding_tool_returns_geometry_artifact",
    "test_grounding_tool_rejects_non_local_image_uri",
    "test_collage_tool_returns_image_artifact",
    "test_crop_tool_uses_mask_cutout_branch",
    "test_crop_tool_uses_unique_output_paths_within_same_loop",
    "test_segment_tool_uses_grounding_and_writes_mask_file",
    "test_crop_tool_uses_grounding_preview_branch",
    "test_crop_tool_rejects_mismatched_mask_source",
    "test_understand_tool_uses_multimodal_llm",
    "test_evaluate_tool_uses_multimodal_llm",
    "test_edit_tool_generates_local_candidate_image_with_unified_args",
    "test_edit_tool_uses_remote_backend_when_configured",
    "test_segment_tool_uses_remote_backend_when_configured",
    "test_segment_tool_remote_retries_with_shorter_prompt",
    "test_segment_tool_remote_falls_back_to_full_image_mask_after_prompt_failures",
    "test_firered_edit_server_endpoint_writes_output",
    "test_sam31_segment_server_endpoint_writes_output",
}


@pytest.fixture(autouse=True)
def fake_model_tools_for_schema_regression_tests(mocker, monkeypatch, request):
    """Keep schema/runtime regression tests from loading real model backends."""

    monkeypatch.setenv("EDIT_BACKEND", "local")
    monkeypatch.setenv("SEGMENT_BACKEND", "local")

    if request.node.name in REAL_TOOL_TESTS:
        return

    from tools.crop_tool import CropTool
    from tools.edit_tool import EditTool
    from tools.evaluate_tool import EvaluateTool
    from tools.segment_tool import SegmentTool
    from tools.understand_tool import UnderstandTool

    def fake_edit_execute(self, state, *, task_id, loop_index, args):
        invocation_args = args.model_dump()
        if len(args.image_refs) > 1:
            invocation_args["reference_refs"] = list(args.image_refs[1:])
        artifact = ImageArtifact(
            id=f"art_image_fake_edit_{len(state.get('artifacts', {})) + 1:03d}",
            uri=f"store://generated/{task_id}/fake_edit_{loop_index:03d}.png",
            payload={
                "role": "candidate_image",
                "source": "fake_edit_output",
                "task_instruction": state["tasks"][task_id].instruction,
            },
            source_ids=list(args.image_refs),
            created_by=ToolName.EDIT.value,
            scope="task",
        )
        return type(
            "FakeToolExecutionResult",
            (),
            {
                "invocation": ToolInvocationRecord(
                    id=f"op_fake_edit_{loop_index:03d}",
                    task_id=task_id,
                    loop_index=loop_index,
                    tool_name=ToolName.EDIT,
                    args=invocation_args,
                    status="succeeded",
                    output_refs=[artifact.id],
                    result_payload={
                        "instruction": args.instruction,
                        "image_refs": list(args.image_refs),
                        "reference_refs": list(args.image_refs[1:]),
                        "output_image_ref": artifact.id,
                    },
                ),
                "artifacts": [artifact],
            },
        )()

    def fake_crop_execute(self, state, *, task_id, loop_index, args):
        artifact = ImageArtifact(
            id=f"art_crop_fake_{loop_index:03d}",
            uri=f"store://generated/{task_id}/fake_crop_{loop_index:03d}.png",
            payload={
                "role": "cropped_preview",
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "grounding_ref": args.grounding_ref,
                "source": "crop_preview",
                "crop_mode": "mask_cutout",
            },
            source_ids=[
                args.image_ref,
                *([args.mask_ref] if args.mask_ref else []),
                *([args.grounding_ref] if args.grounding_ref else []),
            ],
            created_by=ToolName.CROP.value,
            scope="task",
        )
        return type(
            "FakeToolExecutionResult",
            (),
            {
                "invocation": ToolInvocationRecord(
                    id=f"op_fake_crop_{loop_index:03d}",
                    task_id=task_id,
                    loop_index=loop_index,
                    tool_name=ToolName.CROP,
                    args=args.model_dump(),
                    status="succeeded",
                    output_refs=[artifact.id],
                ),
                "artifacts": [artifact],
            },
        )()

    def fake_segment_execute(self, state, *, task_id, loop_index, args):
        artifact = MaskArtifact(
            id=f"art_mask_fake_{loop_index:03d}",
            uri=f"store://generated/{task_id}/fake_mask_{loop_index:03d}.png",
            payload={
                "image_ref": args.image_ref,
                "image_artifact_id": args.image_ref,
                "prompt": args.prompt,
                "source": "fake_segment_output",
            },
            source_ids=[args.image_ref],
            created_by=ToolName.SEGMENT.value,
            scope="task",
        )
        return type(
            "FakeToolExecutionResult",
            (),
            {
                "invocation": ToolInvocationRecord(
                    id=f"op_fake_segment_{loop_index:03d}",
                    task_id=task_id,
                    loop_index=loop_index,
                    tool_name=ToolName.SEGMENT,
                    args=args.model_dump(),
                    status="succeeded",
                    output_refs=[artifact.id],
                ),
                "artifacts": [artifact],
            },
        )()

    def fake_understand_execute(self, state, *, task_id, loop_index, args):
        artifact = UnderstandingArtifact(
            id=f"art_understanding_fake_{loop_index:03d}",
            payload={
                "image_ref": args.image_ref,
                "task_instruction": state["tasks"][task_id].instruction if task_id != "bootstrap" else state["input"]["instruction_text"],
                "summary": f"Fake understanding for {args.image_ref}",
            },
            source_ids=[args.image_ref],
            created_by=ToolName.UNDERSTAND.value,
            scope="session" if task_id == "bootstrap" else "task",
        )
        return type(
            "FakeToolExecutionResult",
            (),
            {
                "invocation": ToolInvocationRecord(
                    id=f"op_fake_understand_{loop_index:03d}",
                    task_id=task_id,
                    loop_index=loop_index,
                    tool_name=ToolName.UNDERSTAND,
                    args=args.model_dump(),
                    status="succeeded",
                    output_refs=[artifact.id],
                ),
                "artifacts": [artifact],
            },
        )()

    def fake_evaluate_execute(self, state, *, task_id, loop_index, args):
        desired_route = state.get("input", {}).get("desired_decision_route", "pass")
        if desired_route == "continue_execute":
            verdict = "replan" if state["session"].task_states[task_id].evaluator_checkpoint_count > 3 else "needs_revision"
            is_satisfied = True
            base_score = 3
        elif desired_route in {"replan", "fail"}:
            verdict = "replan"
            is_satisfied = False
            base_score = 2
        else:
            verdict = "pass"
            is_satisfied = True
            base_score = 4
        artifact = EvaluationArtifact(
            id=f"art_eval_fake_{loop_index:03d}",
            payload={
                "input_refs": [],
                "candidate_ref": args.candidate_ref,
                "candidate_refs": list(args.candidate_refs),
                "checks": list(args.checks),
                "is_satisfied": is_satisfied,
                "verdict": verdict,
                "reason": "Fake evaluation for schema test.",
                "issues": [],
                "scores": {
                    "instruction_success": base_score,
                    "reference_consistency": base_score,
                    "overediting": base_score,
                    "naturalness": base_score,
                    "artifacts": base_score,
                    "semantic_score": base_score,
                    "quality_score": base_score,
                    "weighted_score": base_score,
                    "overall_score": base_score,
                },
            },
            source_ids=list(args.candidate_refs),
            created_by=ToolName.EVALUATE.value,
            scope="task",
        )
        return type(
            "FakeToolExecutionResult",
            (),
            {
                "invocation": ToolInvocationRecord(
                    id=f"op_fake_evaluate_{loop_index:03d}",
                    task_id=task_id,
                    loop_index=loop_index,
                    tool_name=ToolName.EVALUATE,
                    args=args.model_dump(),
                    status="succeeded",
                    output_refs=[artifact.id],
                ),
                "artifacts": [artifact],
            },
        )()

    mocker.patch.object(EditTool, "execute", fake_edit_execute)
    mocker.patch.object(CropTool, "execute", fake_crop_execute)
    mocker.patch.object(SegmentTool, "execute", fake_segment_execute)
    mocker.patch.object(UnderstandTool, "execute", fake_understand_execute)
    mocker.patch.object(EvaluateTool, "execute", fake_evaluate_execute)
