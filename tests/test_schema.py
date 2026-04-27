from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from agent import agent, create_agent, main as agent_main
from agents import EvaluatorAgent, ExecuteAgent, PlanAgent
from llm import (
    encode_image_path_to_data_url,
    invoke_llm,
    invoke_multimodal_llm,
    invoke_structured_llm,
    invoke_structured_multimodal_llm,
    load_llm_config,
)
from runtime.graph import build_runtime_graph
from runtime.input_selector import (
    TaskInputSelectionOutput,
    build_candidate_image_pool,
    build_candidate_images_text,
)
from runtime.scheduler import select_next_runnable_task
from schema import (
    ArtifactIndex,
    ArtifactKind,
    ArtifactStage,
    CollageArgs,
    CropArgs,
    Decision,
    DecisionLLMOutput,
    DecisionRoute,
    EditArgs,
    EvaluateArgs,
    EvaluateLLMOutput,
    EvaluationScores,
    EvaluationArtifact,
    ExecuteLLMOutput,
    ExecutionOutcome,
    GeometryArtifact,
    GroundingArgs,
    GroundingCandidate,
    GroundingLLMOutput,
    GroundingPoint,
    ImageArtifact,
    InstructionArtifact,
    MaskArtifact,
    ObserveArtifactSummary,
    ObserveLLMOutput,
    Plan,
    PlanLLMOutput,
    PlanTaskSpec,
    ReplanMode,
    ReplanRequest,
    SegmentArgs,
    SessionPhase,
    SessionState,
    Task,
    TaskActRecord,
    TaskLoop,
    TaskRetryAdvice,
    TaskState,
    TaskStatus,
    ToolInvocationRecord,
    ToolName,
    UnderstandArgs,
    UnderstandingArtifact,
)
from tools import build_default_tool_registry
from tools.evaluate_tool import EvaluateTool


REAL_TOOL_TESTS = {
    "test_grounding_tool_returns_geometry_artifact",
    "test_grounding_tool_rejects_non_local_image_uri",
    "test_collage_tool_returns_image_artifact",
    "test_crop_tool_uses_mask_cutout_branch",
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


def _evaluation_scores(score: int = 4, **overrides: int) -> EvaluationScores:
    values = {
        "instruction_success": score,
        "reference_consistency": score,
        "overediting": score,
        "naturalness": score,
        "artifacts": score,
    }
    values.update(overrides)
    return EvaluationScores(**values)


def test_evaluate_tool_derive_verdict_passes_only_satisfied_high_scores() -> None:
    scores = _evaluation_scores(4)
    calculated_scores = EvaluateTool._calculate_scores(scores)

    verdict = EvaluateTool._derive_verdict(
        is_satisfied=True,
        scores=scores,
        calculated_scores=calculated_scores,
        evaluator_checkpoint_count=1,
    )

    assert verdict == "pass"


def test_evaluate_tool_derive_verdict_unsatisfied_low_risk_needs_revision() -> None:
    scores = _evaluation_scores(3)
    calculated_scores = EvaluateTool._calculate_scores(scores)

    verdict = EvaluateTool._derive_verdict(
        is_satisfied=False,
        scores=scores,
        calculated_scores=calculated_scores,
        evaluator_checkpoint_count=1,
    )

    assert verdict == "needs_revision"


def test_evaluate_tool_derive_verdict_severe_subscore_replans() -> None:
    scores = _evaluation_scores(4, artifacts=1)
    calculated_scores = EvaluateTool._calculate_scores(scores)

    verdict = EvaluateTool._derive_verdict(
        is_satisfied=False,
        scores=scores,
        calculated_scores=calculated_scores,
        evaluator_checkpoint_count=1,
    )

    assert verdict == "replan"


def test_evaluate_tool_derive_verdict_max_revision_count_replans_when_not_passed() -> None:
    scores = _evaluation_scores(3)
    calculated_scores = EvaluateTool._calculate_scores(scores)

    verdict = EvaluateTool._derive_verdict(
        is_satisfied=False,
        scores=scores,
        calculated_scores=calculated_scores,
        evaluator_checkpoint_count=3,
    )

    assert verdict == "replan"


@pytest.fixture(autouse=True)
def fake_model_tools_for_schema_tests(mocker, monkeypatch, request):
    """Keep schema/runtime regression tests from loading real model backends."""

    monkeypatch.setenv("EDIT_BACKEND", "local")
    monkeypatch.setenv("SEGMENT_BACKEND", "local")

    if request.node.name in REAL_TOOL_TESTS:
        return

    from tools.edit_tool import EditTool
    from tools.evaluate_tool import EvaluateTool
    from tools.crop_tool import CropTool
    from tools.segment_tool import SegmentTool
    from tools.understand_tool import UnderstandTool

    def fake_edit_run(self, state, *, task_id, loop_index, args):
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

    def fake_crop_run(self, state, *, task_id, loop_index, args):
        artifact = ImageArtifact(
            id=f"art_crop_fake_{loop_index:03d}",
            uri=f"store://generated/{task_id}/fake_crop_{loop_index:03d}.png",
            payload={
                "role": "cropped_preview",
                "image_ref": args.image_ref,
                "mask_ref": args.mask_ref,
                "grounding_ref": args.grounding_ref,
                "source": "fake_crop_output",
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

    def fake_segment_run(self, state, *, task_id, loop_index, args):
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

    def fake_understand_run(self, state, *, task_id, loop_index, args):
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

    def fake_evaluate_run(self, state, *, task_id, loop_index, args):
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

    mocker.patch.object(EditTool, "run", fake_edit_run)
    mocker.patch.object(CropTool, "run", fake_crop_run)
    mocker.patch.object(SegmentTool, "run", fake_segment_run)
    mocker.patch.object(UnderstandTool, "run", fake_understand_run)
    mocker.patch.object(EvaluateTool, "run", fake_evaluate_run)


def test_public_imports_construct_minimal_objects() -> None:
    image = ImageArtifact(
        id="art_img_001",
        uri="store://images/input.png",
        payload={"role": "input"},
        summary="原始输入图",
        stage=ArtifactStage.COMMITTED,
    )
    mask = MaskArtifact(
        id="art_mask_001",
        uri="store://masks/object.png",
        payload={"target": "shirt"},
    )
    geometry = GeometryArtifact(
        id="art_geo_001",
        payload={"image_artifact_id": "art_img_001", "candidates": [{"bbox": [1, 2, 3, 4], "score": 0.9}]},
    )
    instruction = InstructionArtifact(
        id="art_inst_001",
        payload={"raw_instruction": "replace the shirt"},
        created_by="user",
    )
    understanding = UnderstandingArtifact(
        id="art_under_001",
        payload={"image_ref": "art_img_001", "summary": "person in shirt"},
        source_ids=["art_img_001"],
    )
    evaluation = EvaluationArtifact(
        id="art_eval_001",
        payload={"verdict": "good", "issues": []},
        source_ids=["art_img_001"],
    )

    assert image.kind == ArtifactKind.IMAGE
    assert mask.kind == ArtifactKind.MASK
    assert geometry.kind == ArtifactKind.GEOMETRY
    assert instruction.kind == ArtifactKind.INSTRUCTION
    assert understanding.kind == ArtifactKind.UNDERSTANDING
    assert evaluation.kind == ArtifactKind.EVALUATION
    assert image.summary == "原始输入图"


def test_instruction_artifact_returns_instruction_text() -> None:
    artifact = InstructionArtifact(
        id="art_inst_001",
        payload={"instruction_text": "把人物放入背景中，并保持人物身份不变。"},
    )

    assert artifact.get_instruction_text() == "把人物放入背景中，并保持人物身份不变。"


def test_runtime_models_construct() -> None:
    plan = Plan(
        id="plan_001",
        instruction="edit the shirt color",
        task_ids=["task_001"],
        input_artifact_ids=["art_img_001"],
        understanding_artifact_ids=["art_under_001"],
    )
    task = Task(
        id="task_001",
        plan_id="plan_001",
        type="local_edit",
        instruction="segment shirt then edit it",
        input_artifact_ids=["art_img_001"],
        acceptance_criteria=["shirt color changed", "face unchanged"],
    )
    task_state = TaskState(
        task_id="task_001",
        status=TaskStatus.RUNNING,
        latest_execution_outcome=ExecutionOutcome.SUCCESS,
        latest_execute_checkpoint="passed",
        latest_evaluate_checkpoint="passed",
        loop_count=1,
    )
    session = SessionState(
        session_id="sess_001",
        phase=SessionPhase.EXECUTING,
        current_plan_id="plan_001",
        current_task_id="task_001",
        task_states={"task_001": task_state},
        artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_001"]}),
    )

    assert plan.task_ids == ["task_001"]
    assert task.id == "task_001"
    assert session.task_states["task_001"].status == TaskStatus.RUNNING


def test_task_loop_construct() -> None:
    loop = TaskLoop(
        id="loop_task_001_001",
        task_id="task_001",
        loop_index=1,
        thinking="Need to segment before local edit.",
        selected_tools=["segment", "crop", "understand", "edit"],
        output_artifact_ids=["art_image_001"],
        observation="Produced one candidate image.",
    )
    assert loop.loop_index == 1
    assert loop.selected_tools == ["segment", "crop", "understand", "edit"]


def test_task_act_record_construct() -> None:
    record = TaskActRecord(
        task_id="task_001",
        loop_index=1,
        act_index=1,
        thinking_text="Need to rewrite prompt before editing.",
        tool_name="prompt_reconstruct",
        tool_args={"input_artifact_ids": ["art_img_input_001"]},
        output_artifact_ids=["art_instruction_001"],
        observation_text="Produced one rewritten instruction artifact.",
    )
    assert record.tool_name == "prompt_reconstruct"


def test_observe_llm_output_construct() -> None:
    out = ObserveLLMOutput(
        outcome="continue",
        observation="This crop preview is useful but not final yet.",
        artifact_summaries=[
            ObserveArtifactSummary(
                artifact_id="art_image_001",
                summary="局部裁剪预览图，用于下一步判断编辑区域。",
            )
        ],
    )
    assert out.outcome == "continue"
    assert out.artifact_summaries[0].artifact_id == "art_image_001"


def test_tool_registry_contains_prompt_reconstruct() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.PROMPT_RECONSTRUCT) is not None


def test_build_candidate_images_text_uses_compact_scheme_b() -> None:
    state = {
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                summary="上衣单品图",
                scope="session",
            ),
            "art_image_001": ImageArtifact(
                id="art_image_001",
                uri="store://generated/task_001.png",
                payload={"role": "candidate_image"},
                summary="已合成的人物主体图",
                scope="task",
            ),
        },
        "operations": [],
    }
    text = build_candidate_images_text(state, ["art_img_input_001", "art_image_001"])
    assert "[art_img_input_001] 上衣单品图 | 原始输入图片" in text
    assert "[art_image_001] 已合成的人物主体图 | source=generated_result" in text
    assert "uri:" not in text


def test_build_candidate_image_pool_includes_current_plan_inputs() -> None:
    state = {
        "session": SessionState(
            session_id="sess_plan_candidates",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="测试 plan 输入池",
                task_ids=["task_001"],
                input_artifact_ids=[
                    "art_plan_visible_001",
                    "art_not_image_001",
                    "art_img_input_001",
                ],
            )
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="edit",
                instruction="执行当前任务",
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_plan_visible_001": ImageArtifact(
                id="art_plan_visible_001",
                uri="store://images/retained.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
            "art_not_image_001": InstructionArtifact(
                id="art_not_image_001",
                payload={"instruction_text": "not an image"},
                scope="task",
            ),
        },
    }

    assert build_candidate_image_pool(state) == [
        "art_img_input_001",
        "art_plan_visible_001",
    ]


def test_select_next_runnable_task_respects_dependencies() -> None:
    state = {
        "session": SessionState(
            session_id="sched_001",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id=None,
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.PENDING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="dag test",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="compose_subject",
                instruction="first",
            ),
            "task_002": Task(
                id="task_002",
                plan_id="plan_001",
                type="place_subject_in_background",
                instruction="second",
                depends_on=["task_001"],
            ),
        },
    }
    assert select_next_runnable_task(state) == "task_002"


def test_tool_args_construct() -> None:
    understand = UnderstandArgs(
        image_ref="art_img_001",
    )
    grounding = GroundingArgs(
        image_ref="art_img_001",
        grounding_query="Locate the shirt area for editing.",
        top_k=2,
    )
    segment = SegmentArgs(
        image_ref="art_img_001",
        prompt="segment the shirt",
    )
    crop = CropArgs(
        image_ref="art_img_001",
        mask_ref="art_mask_001",
    )
    collage = CollageArgs(
        block_artifact_ids=["art_img_001", "art_img_002"],
        layout_goal="identity and clothing are primary",
    )
    edit = EditArgs(
        instruction="把人物放到背景中",
        image_refs=["art_img_001", "art_img_002"],
    )

    assert understand.image_ref == "art_img_001"
    assert grounding.grounding_query == "Locate the shirt area for editing."
    assert segment.prompt == "segment the shirt"
    assert crop.mask_ref == "art_mask_001"
    assert crop.grounding_ref is None
    assert collage.block_artifact_ids == ["art_img_001", "art_img_002"]
    assert edit.image_refs == ["art_img_001", "art_img_002"]


def test_collage_args_validate_inputs() -> None:
    with pytest.raises(ValidationError):
        CollageArgs(
            block_artifact_ids=["art_img_001"],
            layout_goal="identity and clothing are primary",
        )
    with pytest.raises(ValidationError):
        CollageArgs(
            block_artifact_ids=["art_img_001", "art_img_001"],
            layout_goal="identity and clothing are primary",
        )
    with pytest.raises(ValidationError):
        CollageArgs(
            block_artifact_ids=["art_img_001", "art_img_002"],
            layout_goal="   ",
        )


def test_segment_args_require_prompt() -> None:
    with pytest.raises(ValidationError):
        SegmentArgs(
            image_ref="art_img_001",
        )


def test_crop_args_accepts_grounding_only() -> None:
    crop = CropArgs(
        image_ref="art_img_001",
        grounding_ref="art_geometry_001",
        padding=6,
    )

    assert crop.grounding_ref == "art_geometry_001"
    assert crop.padding == 6


def test_crop_args_requires_mask_or_grounding() -> None:
    with pytest.raises(ValidationError):
        CropArgs(
            image_ref="art_img_001",
        )


def test_grounding_candidate_rejects_invalid_bbox_semantics() -> None:
    with pytest.raises(ValidationError):
        GroundingCandidate(
            label="shirt",
            bbox=[10, 20, 10, 60],
        )


def test_grounding_candidate_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        GroundingCandidate(
            label="shirt",
            bbox=[10, 20, 40, 60],
            score=1.2,
        )


def test_grounding_candidate_rejects_too_many_positive_points() -> None:
    with pytest.raises(ValidationError):
        GroundingCandidate(
            label="shirt",
            bbox=[10, 20, 40, 60],
            positive_points=[
                GroundingPoint(x=1, y=1),
                GroundingPoint(x=2, y=2),
                GroundingPoint(x=3, y=3),
                GroundingPoint(x=4, y=4),
            ],
        )


def test_grounding_llm_output_requires_candidates() -> None:
    with pytest.raises(ValidationError):
        GroundingLLMOutput(candidates=[])


def test_tool_registry_contains_grounding_and_collage() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.GROUNDING) is not None
    assert registry.get(ToolName.COLLAGE) is not None


class EditArgsHolder(BaseModel):
    value: EditArgs


def test_edit_args_accepts_valid_payload() -> None:
    payload = {
        "value": {
            "instruction": "把人物放到背景里",
            "image_refs": ["art_img_001", "art_img_002"],
        }
    }
    parsed = EditArgsHolder.model_validate(payload)
    assert parsed.value.instruction == "把人物放到背景里"
    assert parsed.value.image_refs == ["art_img_001", "art_img_002"]


def test_edit_args_rejects_too_many_images() -> None:
    payload = {
        "value": {
            "instruction": "把人物放到背景里",
            "image_refs": [
                "art_img_001",
                "art_img_002",
                "art_img_003",
                "art_img_004",
            ],
        }
    }
    with pytest.raises(ValidationError):
        EditArgsHolder.model_validate(payload)


def test_edit_args_rejects_duplicate_images() -> None:
    payload = {
        "value": {
            "instruction": "把人物放到背景里",
            "image_refs": ["art_img_001", "art_img_001"],
        }
    }
    with pytest.raises(ValidationError):
        EditArgsHolder.model_validate(payload)


def test_decision_continue_execute_requires_task_retry() -> None:
    decision = Decision(
        id="dec_001",
        route=DecisionRoute.CONTINUE_EXECUTE,
        task_id="task_001",
        summary="Need one more pass",
        source_execution_outcome=ExecutionOutcome.FAILURE,
        task_retry=TaskRetryAdvice(
            reason="mask too loose",
            base_candidate_artifact_id="art_img_001",
            fix_focuses=["tighten mask"],
        ),
    )
    assert decision.task_retry is not None


def test_decision_replan_requires_replan_request() -> None:
    decision = Decision(
        id="dec_002",
        route=DecisionRoute.REPLAN,
        plan_id="plan_001",
        summary="Need to split the task",
        replan=ReplanRequest(
            mode=ReplanMode.SPLIT_TASK,
            reason="single task is too coarse",
        ),
    )
    assert decision.replan is not None


def test_decision_pass_forbids_retry_payloads() -> None:
    with pytest.raises(ValidationError):
        Decision(
            id="dec_003",
            route=DecisionRoute.PASS,
            summary="Looks good",
            task_retry=TaskRetryAdvice(
                reason="should not exist",
                fix_focuses=["none"],
            ),
        )


def test_tool_invocation_record_links_outputs() -> None:
    record = ToolInvocationRecord(
        id="op_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "把人物放到背景里", "image_refs": ["art_img_001"]},
        status="succeeded",
        output_refs=["art_img_002"],
        raw_output_uri="runs/sess_001/task_001/loop_01/edit/result.json",
    )
    assert record.output_refs == ["art_img_002"]


def test_minimal_runtime_graph_pass_path() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_001",
                "image_uri": "store://images/input.png",
                "image_uris": ["store://images/input.png"],
                "instruction_text": "change the shirt color",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_plan_id == "plan_001"
    assert result["session"].current_task_id == "task_001"
    assert len(result["operations"]) == 1
    assert [op.tool_name for op in result["operations"]] == [ToolName.UNDERSTAND]
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]
    assert result["tasks"]["task_001"].type == "compose_subject"
    assert result["tasks"]["task_002"].type == "place_subject_in_background"
    understand_ops = [op for op in result["operations"] if op.tool_name == ToolName.UNDERSTAND]
    assert len(understand_ops) == 1


def test_minimal_runtime_graph_replan_path() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_002",
                "image_uri": "store://images/input.png",
                "image_uris": ["store://images/input.png"],
                "instruction_text": "change the shirt color",
                "desired_decision_route": "replan",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_001"
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]


def test_four_images_generate_task_todo_array() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "sess_003",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "Generate a photo of this person wearing the provided top and skirt in the provided background.",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert len(result["session"].artifact_index.by_type[ArtifactKind.IMAGE]) == 4
    understand_ops = [op for op in result["operations"] if op.tool_name == ToolName.UNDERSTAND]
    assert len(understand_ops) == 4
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002"]
    assert result["tasks"]["task_001"].instruction.startswith("Combine the face")
    assert result["tasks"]["task_002"].instruction.startswith("Place the composed subject")
    assert result["tasks"]["task_001"].depends_on == []
    assert result["tasks"]["task_002"].depends_on == ["task_001"]
    assert not any(artifact_id.startswith("art_image_") for artifact_id in result["tasks"]["task_002"].input_artifact_ids)
    assert result["session"].current_task_id == "task_001"
    assert result["session"].task_states["task_001"].status == TaskStatus.RUNNING
    assert result["session"].task_states["task_002"].status == TaskStatus.PENDING
    assert set(result["session"].artifact_index.by_type.keys()) == {
        ArtifactKind.IMAGE,
    }


def test_single_task_loop_runs_until_success() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "single-task-loop",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.DONE
    assert result["decision"].route == DecisionRoute.PASS
    assert result["session"].final_result_id is not None
    assert len(result["task_loops"]) >= 2
    assert all(loop.selected_tools for loop in result["task_loops"])


def test_all_tasks_run_until_session_done() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "all-task-loop",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].status == TaskStatus.PASSED
    assert result["session"].task_states["task_002"].status == TaskStatus.PASSED
    loop_task_ids = [loop.task_id for loop in result["task_loops"]]
    assert "task_001" in loop_task_ids
    assert "task_002" in loop_task_ids


def test_task_one_result_is_resolved_into_task_two_runtime_inputs() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "inject-task-result",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    resolved_inputs = result["session"].task_states["task_002"].resolved_input_artifact_ids
    assert any(artifact_id.startswith("art_image_") for artifact_id in resolved_inputs)
    assert result["session"].task_states["task_002"].input_selection_reasoning is not None
    assert result["session"].task_states["task_002"].input_validation_summary is not None


def test_task_two_uses_reference_edit_after_task_switch() -> None:
    graph = build_runtime_graph()
    result = graph.invoke(
        {
            "input": {
                "session_id": "reference-edit-switch",
                "image_uri": "examples/fig1.jpg",
                "image_uris": [
                    "examples/fig1.jpg",
                    "examples/fig2.jpg",
                    "examples/fig3.jpg",
                    "examples/fig4.jpg",
                ],
                "instruction_text": "生成一张这个人穿着这个上衣和下衣，并在这个背景里拍照的图片。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    task_two_edit_ops = [
        op for op in result["operations"]
        if op.task_id == "task_002" and op.tool_name == ToolName.EDIT
    ]
    assert task_two_edit_ops
    assert "reference_refs" in task_two_edit_ops[0].args


def test_agent_entrypoint_exports_compiled_graph() -> None:
    compiled = create_agent()
    assert type(compiled).__name__ == "CompiledStateGraph"
    assert type(agent).__name__ == "CompiledStateGraph"


def test_agent_cli_help_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        agent_main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Run the fig edit agent runtime" in captured.out
    assert "--images" in captured.out


def test_agent_cli_runs_rule_based_fallback(mocker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AGENT_LOG_ENABLED", "false")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "false")
    unload_mock = mocker.patch("agent.unload_pipeline")

    exit_code = agent_main(
        [
            "--no-use-llm",
            "--stop-after-first-edit",
            "--session-id",
            "agent-cli-test",
            "--images",
            "examples/fig1.jpg",
            "--instruction",
            "Keep the image unchanged.",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["stop_reason"] == "first_edit_candidate"
    assert summary["final_artifact"]["kind"] == "image"
    assert summary["operations"][-1]["tool_name"] == "edit"
    unload_mock.assert_called_once_with()


def test_firered_backend_optimization_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from vision_backends.firered_edit_backend import backend_config_snapshot

    monkeypatch.delenv("FIRERED_ENABLE_ATTENTION_SLICING", raising=False)
    monkeypatch.delenv("FIRERED_ENABLE_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("FIRERED_ENABLE_WARMUP", raising=False)
    monkeypatch.delenv("FIRERED_WARMUP_STEPS", raising=False)

    snapshot = backend_config_snapshot()

    assert snapshot["enable_attention_slicing"] is True
    assert snapshot["enable_torch_compile"] is True
    assert snapshot["enable_warmup"] is True
    assert snapshot["warmup_steps"] == 4


def test_firered_post_load_optimizations_apply_once_per_cached_load(mocker) -> None:
    from vision_backends import firered_edit_backend as backend

    class FakeTorch:
        class inference_mode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def compile(module, mode=None):
            module.compiled_mode = mode
            return module

    class FakeTransformer:
        def compile_repeated_blocks(self, mode, dynamic):
            self.compile_args = {"mode": mode, "dynamic": dynamic}

    class FakePipe:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.vae = type("FakeVAE", (), {})()
            self.calls = []
            self.slicing_enabled = False

        def enable_attention_slicing(self):
            self.slicing_enabled = True

        def __call__(self, **inputs):
            self.calls.append(inputs)
            return object()

    fake_pipe = FakePipe()
    settings = backend.backend_config_snapshot()
    settings.update(
        {
            "enable_attention_slicing": True,
            "enable_torch_compile": True,
            "enable_warmup": True,
            "warmup_steps": 2,
            "warmup_height": 64,
            "warmup_width": 96,
            "height": 128,
            "width": 160,
            "generator_device": "cpu",
        }
    )

    backend._apply_post_load_optimizations(FakeTorch, fake_pipe, settings)

    assert fake_pipe.slicing_enabled is True
    assert fake_pipe.transformer.compile_args == {"mode": "default", "dynamic": True}
    assert fake_pipe.vae.compiled_mode == "reduce-overhead"
    assert len(fake_pipe.calls) == 1
    assert fake_pipe.calls[0]["num_inference_steps"] == 2
    assert fake_pipe.calls[0]["height"] == 64
    assert fake_pipe.calls[0]["width"] == 96


def test_firered_unload_pipeline_clears_cache(mocker) -> None:
    from vision_backends import firered_edit_backend as backend

    cache_clear = mocker.patch.object(backend.load_pipeline, "cache_clear")
    collect = mocker.patch("vision_backends.firered_edit_backend.gc.collect")
    fake_torch = mocker.Mock()
    fake_torch.cuda.is_available.return_value = True
    import_module = mocker.patch("vision_backends.firered_edit_backend.importlib.import_module", return_value=fake_torch)

    backend.unload_pipeline()

    cache_clear.assert_called_once_with()
    collect.assert_called_once_with()
    import_module.assert_called_once_with("torch")
    fake_torch.cuda.empty_cache.assert_called_once_with()


def test_firered_manual_shard_plan_uses_third_gpu_for_non_transformer_weights(mocker) -> None:
    from vision_backends._vendor import firered_manual_pipeline as manual_pipeline

    mocker.patch.object(
        manual_pipeline,
        "_inspect_transformer_structure",
        return_value={"block_count": 60},
    )
    mocker.patch.object(
        manual_pipeline,
        "_inspect_text_encoder_structure",
        return_value={"visual_block_count": 32, "language_layer_count": 28},
    )

    shard_plan = manual_pipeline.build_manual_shard_plan(
        model_path="mock-model",
        visible_gpu_ids=[0, 1, 2],
        local_files_only=True,
        include_estimates=False,
    )

    assert shard_plan["strategy"] == "manual_grouped_component_shard"
    assert shard_plan["transformer_devices"] == [0, 1]
    assert shard_plan["text_encoder_devices"] == [2]
    assert shard_plan["vae_device"] == 2
    assert shard_plan["transformer_device_map"]["transformer_blocks.0"] == 0
    assert shard_plan["transformer_device_map"]["transformer_blocks.29"] == 0
    assert shard_plan["transformer_device_map"]["transformer_blocks.30"] == 1
    assert shard_plan["transformer_device_map"]["transformer_blocks.59"] == 1
    assert shard_plan["text_encoder_device_map"]["model.visual.patch_embed"] == 2
    assert shard_plan["text_encoder_device_map"]["model.visual.merger"] == 2
    assert shard_plan["text_encoder_device_map"]["model.language_model.embed_tokens"] == 2
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.0"] == 2
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.27"] == 2


def test_firered_manual_shard_plan_preserves_original_four_gpu_layout(mocker) -> None:
    from vision_backends._vendor import firered_manual_pipeline as manual_pipeline

    mocker.patch.object(
        manual_pipeline,
        "_inspect_transformer_structure",
        return_value={"block_count": 60},
    )
    mocker.patch.object(
        manual_pipeline,
        "_inspect_text_encoder_structure",
        return_value={"visual_block_count": 32, "language_layer_count": 28},
    )

    shard_plan = manual_pipeline.build_manual_shard_plan(
        model_path="mock-model",
        visible_gpu_ids=[0, 1, 2, 3],
        local_files_only=True,
        include_estimates=False,
    )

    assert shard_plan["strategy"] == "manual_grouped_component_shard"
    assert shard_plan["transformer_devices"] == [0, 1]
    assert shard_plan["text_encoder_devices"] == [2, 3]
    assert shard_plan["vae_device"] == 3
    assert shard_plan["transformer_device_map"]["transformer_blocks.0"] == 0
    assert shard_plan["transformer_device_map"]["transformer_blocks.29"] == 0
    assert shard_plan["transformer_device_map"]["transformer_blocks.30"] == 1
    assert shard_plan["transformer_device_map"]["transformer_blocks.59"] == 1
    assert shard_plan["text_encoder_device_map"]["model.visual.patch_embed"] == 2
    assert shard_plan["text_encoder_device_map"]["model.visual.merger"] == 3
    assert shard_plan["text_encoder_device_map"]["model.language_model.embed_tokens"] == 3
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.0"] == 3
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.13"] == 3
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.14"] == 2
    assert shard_plan["text_encoder_device_map"]["model.language_model.layers.27"] == 2


def test_load_llm_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    config = load_llm_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.invalid/v1"
    assert config.model_name == "test-model"
    assert config.temperature == 0.2


def test_invoke_llm_uses_langchain_model(mocker) -> None:
    fake_response = mocker.Mock()
    fake_response.__class__.__name__ = "AIMessage"

    fake_model = mocker.Mock()
    fake_model.invoke.return_value = fake_response

    result = invoke_llm(
        user_prompt="hello",
        system_prompt="you are helpful",
        model=fake_model,
    )

    assert result is fake_response
    fake_model.invoke.assert_called_once()


def test_invoke_structured_llm_uses_structured_output(mocker) -> None:
    class OutputSchema(BaseModel):
        answer: str

    parsed = OutputSchema(answer="done")
    fake_result = mocker.Mock()
    fake_result.output = parsed
    fake_agent = mocker.Mock()
    fake_agent.run_sync.return_value = fake_result
    agent_cls = mocker.patch("llm.client.Agent", return_value=fake_agent)

    result = invoke_structured_llm(
        user_prompt="return structured",
        system_prompt="system",
        output_schema=OutputSchema,
        model="openai:gpt-5.4",
    )

    assert result == parsed
    agent_cls.assert_called_once()
    fake_agent.run_sync.assert_called_once_with("return structured")


def test_encode_image_path_to_data_url_reads_local_file(tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    data_url = encode_image_path_to_data_url(str(image_path))

    assert data_url.startswith("data:image/png;base64,")


def test_invoke_multimodal_llm_uses_langchain_model_with_image_content(mocker, tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    fake_response = mocker.Mock()
    fake_response.__class__.__name__ = "AIMessage"
    fake_model = mocker.Mock()
    fake_model.invoke.return_value = fake_response

    result = invoke_multimodal_llm(
        user_prompt="describe image",
        system_prompt="system",
        image_paths=[str(image_path)],
        model=fake_model,
    )

    assert result is fake_response
    fake_model.invoke.assert_called_once()
    messages = fake_model.invoke.call_args.args[0]
    assert messages[-1].content[0]["type"] == "text"
    assert messages[-1].content[1]["type"] == "image_url"
    assert messages[-1].content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_invoke_structured_multimodal_llm_uses_structured_output(mocker, tmp_path) -> None:
    class OutputSchema(BaseModel):
        answer: str

    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    parsed = OutputSchema(answer="done")
    fake_structured_model = mocker.Mock()
    fake_structured_model.invoke.return_value = parsed
    fake_model = mocker.Mock()
    fake_model.with_structured_output.return_value = fake_structured_model

    result = invoke_structured_multimodal_llm(
        user_prompt="return structured multimodal",
        system_prompt="system",
        image_paths=[str(image_path)],
        output_schema=OutputSchema,
        model=fake_model,
    )

    assert result == parsed
    fake_model.with_structured_output.assert_called_once_with(OutputSchema)
    fake_structured_model.invoke.assert_called_once()


def test_plan_agent_uses_llm_when_enabled(mocker) -> None:
    state = build_runtime_graph().invoke(
        {
            "input": {
                "session_id": "plan_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "make the jacket red",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.PLANNING
    state["session"].current_plan_id = None
    state["session"].current_task_id = None
    state["plans"] = {}
    state["tasks"] = {}
    state["session"].task_states = {}
    state["input"]["use_llm"] = True

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction="llm plan",
            tasks=[
                PlanTaskSpec(
                    id="task_002",
                    type="global_edit",
                    instruction="run one global edit",
                    input_artifact_ids=["art_img_input_001"],
                    depends_on=[],
                    acceptance_criteria=["candidate exists"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["plans"]["plan_001"].instruction == "llm plan"
    assert result["tasks"]["task_002"].type == "global_edit"
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.INSTRUCTION
        for artifact_id in result["session"].task_states["task_002"].task_artifact_ids
    )


def test_execute_agent_uses_llm_strategy_when_enabled(mocker) -> None:
    from PIL import Image

    state = {
        "input": {
            "session_id": "exec_llm",
            "instruction_text": "global recolor",
            "desired_decision_route": "pass",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="global recolor",
                input_artifact_ids=["art_img_input_001"],
                acceptance_criteria=["candidate exists"],
            )
        },
        "session": SessionState(
            session_id="exec_llm",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(
                by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}
            ),
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="global recolor",
                task_ids=["task_001"],
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="global edit is enough",
            selected_tools=["edit"],
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )
    mocker.patch(
        "runtime.input_selector.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )
    mocker.patch(
        "runtime.input_selector.invoke_structured_llm",
        return_value=TaskInputSelectionOutput(
            selected_artifact_ids=["art_img_input_001"],
        ),
    )
    mocker.patch(
        "runtime.input_selector.invoke_llm",
        return_value=mocker.Mock(content="Need one base image and no dependency result."),
    )
    mocker.patch("runtime.input_selector.ensure_understanding_for_images")
    mocker.patch(
        "tools.edit_tool.edit_images",
        return_value=Image.new("RGB", (16, 16), color=(0, 128, 255)),
    )
    mocker.patch(
        "tools.edit_tool.backend_config_snapshot",
        return_value={"model_path": "mock-firered"},
    )

    result = ExecuteAgent().run(state)

    assert result["operations"][-1].tool_name == ToolName.EDIT
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
    ]
    assert result["session"].task_states["task_001"].input_selection_reasoning is not None
    latest_refs = result["session"].task_states["task_001"].latest_artifact_ids
    assert len(latest_refs) == 1
    assert latest_refs[0].startswith("art_image_")
    assert result["operations"][-1].args == {
        "instruction": "global recolor",
        "image_refs": ["art_img_input_001"],
    }
    assert result["task_act_records"]


def test_grounding_tool_returns_geometry_artifact() -> None:
    from tools.grounding_tool import GroundingTool

    import tempfile
    from unittest.mock import patch
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_file.name)

        state = {
            "tasks": {
                "task_001": Task(
                    id="task_001",
                    plan_id="plan_001",
                    type="local_edit",
                    instruction="定位人物区域",
                )
            },
            "session": SessionState(
                session_id="sess_grounding",
                phase=SessionPhase.EXECUTING,
                current_plan_id="plan_001",
                current_task_id="task_001",
                task_states={
                    "task_001": TaskState(
                        task_id="task_001",
                        status=TaskStatus.RUNNING,
                    )
                },
                artifact_index=ArtifactIndex(by_type={}),
            ),
            "artifacts": {
                "art_img_001": ImageArtifact(
                    id="art_img_001",
                    uri=image_file.name,
                    payload={"role": "input"},
                )
            },
            "operations": [],
            "task_act_records": [],
        }

        def fake_structured_multimodal(**kwargs):
            return GroundingLLMOutput(
                candidates=[
                    GroundingCandidate(
                        label="torso",
                        bbox=[-10, -5, 8, 9],
                        score=0.91,
                        positive_points=[GroundingPoint(x=5, y=5)],
                        negative_points=[GroundingPoint(x=99, y=99)],
                    )
                ]
            )

        with patch(
            "tools.grounding_tool.invoke_structured_multimodal_llm",
            side_effect=fake_structured_multimodal,
        ):
            execution = GroundingTool().run(
                state,
                task_id="task_001",
                loop_index=1,
                args=GroundingArgs(
                    image_ref="art_img_001",
                    grounding_query="Locate the person torso",
                    top_k=2,
                ),
            )

        artifact = execution.artifacts[0]
        assert artifact.kind == ArtifactKind.GEOMETRY
        assert artifact.payload["image_artifact_id"] == "art_img_001"
        assert artifact.payload["grounding_query"] == "Locate the person torso"
        assert len(artifact.payload["candidates"]) == 1
        assert artifact.payload["candidates"][0]["bbox"] == [0, 0, 3, 3]
        assert artifact.payload["candidates"][0]["positive_points"] == [{"x": 3, "y": 3}]
        assert artifact.payload["candidates"][0]["negative_points"] == [{"x": 3, "y": 3}]
        assert execution.invocation.result_payload["geometry_artifact_id"] == artifact.id


def test_grounding_tool_rejects_non_local_image_uri() -> None:
    from tools.grounding_tool import GroundingTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="定位人物区域",
            )
        },
        "session": SessionState(
            session_id="sess_grounding_bad_uri",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri="store://images/input.png",
                payload={"role": "input"},
            )
        },
        "operations": [],
        "task_act_records": [],
    }

    with pytest.raises(ValueError, match="not a local file path"):
        GroundingTool().run(
            state,
            task_id="task_001",
            loop_index=1,
            args=GroundingArgs(
                image_ref="art_img_001",
                grounding_query="Locate the person torso",
                top_k=1,
            ),
        )


def test_collage_tool_returns_image_artifact(mocker, tmp_path) -> None:
    from PIL import Image
    from tools.collage_tool import CollageLayoutItem, CollageLayoutResult
    from tools.collage_tool import CollageTool

    image_a = tmp_path / "face.png"
    image_b = tmp_path / "cloth.png"
    Image.new("RGBA", (20, 30), color=(255, 0, 0, 255)).save(image_a)
    Image.new("RGBA", (30, 20), color=(0, 255, 0, 255)).save(image_b)

    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=80,
            canvas_height=40,
            background="transparent",
            items=[
                CollageLayoutItem(
                    artifact_id="art_face_001",
                    x=0,
                    y=0,
                    width=20,
                    height=30,
                    z_index=0,
                ),
                CollageLayoutItem(
                    artifact_id="art_cloth_001",
                    x=30,
                    y=10,
                    width=30,
                    height=20,
                    z_index=1,
                ),
            ],
        ),
    )

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="整理参考图",
            )
        },
        "session": SessionState(
            session_id="sess_collage",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_a),
                summary="face reference",
                payload={"role": "identity_reference"},
            ),
            "art_cloth_001": ImageArtifact(
                id="art_cloth_001",
                uri=str(image_b),
                payload={"role": "clothing_reference", "description": "green garment"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = CollageTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="identity and clothing are primary",
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert Path(artifact.uri).is_file()
    with Image.open(artifact.uri) as output_image:
        assert output_image.size == (80, 40)
    assert artifact.payload["role"] == "collage_reference"
    assert len(artifact.payload["layers"]) == 2
    assert artifact.source_ids == ["art_face_001", "art_cloth_001"]
    assert artifact.summary is None
    assert "previous_collage_ref" not in artifact.payload
    assert execution.invocation.result_payload["collage_artifact_id"] == artifact.id


def test_crop_tool_uses_mask_cutout_branch(tmp_path) -> None:
    from PIL import Image
    from tools.crop_tool import CropTool

    image_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGBA", (10, 10), color=(255, 0, 0, 255)).save(image_path)
    mask = Image.new("L", (10, 10), color=0)
    for x in range(2, 7):
        for y in range(3, 9):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="裁剪前景",
            )
        },
        "session": SessionState(
            session_id="sess_crop_mask",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
            "art_mask_001": MaskArtifact(
                id="art_mask_001",
                uri=str(mask_path),
                payload={"image_ref": "art_img_001"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = CropTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=CropArgs(
            image_ref="art_img_001",
            mask_ref="art_mask_001",
            padding=1,
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert artifact.payload["crop_mode"] == "mask_cutout"
    assert artifact.payload["bbox"] == [1, 2, 8, 10]
    assert artifact.uri.endswith(".png")
    assert Path(artifact.uri).is_file()
    with Image.open(artifact.uri) as cropped:
        assert cropped.mode == "RGBA"


def test_segment_tool_uses_grounding_and_writes_mask_file(tmp_path, mocker) -> None:
    from PIL import Image
    from tools.segment_tool import SegmentTool
    import os

    image_path = tmp_path / "source.png"
    Image.new("RGB", (12, 12), color=(255, 255, 255)).save(image_path)
    mocker.patch.dict(os.environ, {"SAM3_CHECKPOINT_PATH": str(image_path)})

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="分割主体",
            )
        },
        "session": SessionState(
            session_id="sess_segment",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    mocker.patch(
        "tools.segment_tool.sam31_predict_text_prompt_candidates",
        return_value=[
            {
                "name": "sam31_text_0",
                "mask": np.pad(np.ones((8, 8), dtype=bool), 2),
                "mask_logits": np.pad(np.ones((8, 8), dtype=np.float32), 2)[None, ...],
                "score": 0.9,
            }
        ],
    )

    execution = SegmentTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="subject",
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.MASK
    assert artifact.payload["image_ref"] == "art_img_001"
    assert artifact.payload["prompt"] == "subject"
    assert "mask_score" in artifact.payload
    assert Path(artifact.uri).is_file()
    with Image.open(artifact.uri) as mask_image:
        assert mask_image.mode == "L"


def test_segment_tool_uses_remote_backend_when_configured(tmp_path, mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image
    from tools.segment_tool import SegmentTool
    from vision_backends.remote_client import RemoteSegmentResult

    image_path = tmp_path / "source.png"
    Image.new("RGB", (12, 12), color=(255, 255, 255)).save(image_path)
    monkeypatch.setenv("SEGMENT_BACKEND", "remote")
    mocker.patch("tools.segment_tool.resolve_segment_backend", return_value="remote")

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="分割主体",
            )
        },
        "session": SessionState(
            session_id="sess_segment_remote",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    def fake_remote_segment(*, image_path, prompt, output_path):
        Image.fromarray(np.pad(np.ones((8, 8), dtype=np.uint8), 2) * 255).save(output_path)
        return RemoteSegmentResult(
            mask_path=output_path,
            mask_score=0.91,
            selection_metrics={"final_score": 0.88},
            source_stage="remote_sam31",
            text_prompt=prompt,
        )

    remote_mock = mocker.patch("tools.segment_tool.request_sam31_segment", side_effect=fake_remote_segment)

    execution = SegmentTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="subject",
        ),
    )

    artifact = execution.artifacts[0]
    remote_mock.assert_called_once()
    assert artifact.kind == ArtifactKind.MASK
    assert artifact.payload["source_stage"] == "remote_sam31"
    assert artifact.payload["selection_metrics"] == {"final_score": 0.88}
    assert Path(artifact.uri).is_file()


def test_segment_tool_remote_retries_with_shorter_prompt(tmp_path, mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image
    from tools.segment_tool import SegmentTool
    from vision_backends.remote_client import RemoteBackendError, RemoteSegmentResult

    image_path = tmp_path / "source.png"
    Image.new("RGB", (12, 12), color=(255, 255, 255)).save(image_path)
    monkeypatch.setenv("SEGMENT_BACKEND", "remote")
    mocker.patch("tools.segment_tool.resolve_segment_backend", return_value="remote")

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="segment retry",
            )
        },
        "session": SessionState(
            session_id="sess_segment_remote_retry",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    seen_prompts: list[str] = []

    def fake_remote_segment(*, image_path, prompt, output_path):
        seen_prompts.append(prompt)
        if prompt != "person":
            raise RemoteBackendError("remote backend HTTP 400: SAM 3.1 found no acceptable mask candidate")
        Image.fromarray(np.pad(np.ones((8, 8), dtype=np.uint8), 2) * 255).save(output_path)
        return RemoteSegmentResult(
            mask_path=output_path,
            mask_score=0.91,
            selection_metrics={"final_score": 0.88},
            source_stage="remote_sam31",
            text_prompt=prompt,
        )

    mocker.patch("tools.segment_tool.request_sam31_segment", side_effect=fake_remote_segment)

    execution = SegmentTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="Generate a photo of this person wearing the provided top and skirt in the provided background.",
        ),
    )

    artifact = execution.artifacts[0]
    assert seen_prompts == ["person"]
    assert artifact.payload["prompt"] == "person"
    assert artifact.payload["source_stage"] == "remote_sam31"


def test_segment_tool_remote_falls_back_to_full_image_mask_after_prompt_failures(
    tmp_path, mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    from tools.segment_tool import SegmentTool
    from vision_backends.remote_client import RemoteBackendError

    image_path = tmp_path / "source.png"
    Image.new("RGB", (9, 7), color=(255, 255, 255)).save(image_path)
    monkeypatch.setenv("SEGMENT_BACKEND", "remote")
    mocker.patch("tools.segment_tool.resolve_segment_backend", return_value="remote")

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="segment fallback",
            )
        },
        "session": SessionState(
            session_id="sess_segment_remote_fallback",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    mocker.patch(
        "tools.segment_tool.request_sam31_segment",
        side_effect=RemoteBackendError("remote backend HTTP 400: SAM 3.1 found no acceptable mask candidate"),
    )

    execution = SegmentTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="Generate a photo of this person wearing the provided top and skirt in the provided background.",
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.payload["source_stage"] == "fallback_full_image"
    with Image.open(artifact.uri) as mask_image:
        assert mask_image.size == (9, 7)
        assert mask_image.getbbox() == (0, 0, 9, 7)


def test_sam31_segment_server_endpoint_writes_output(tmp_path, mocker) -> None:
    from PIL import Image
    from vision_backends import sam31_segment_server

    source_path = tmp_path / "source.png"
    output_path = tmp_path / "mask.png"
    Image.new("RGB", (12, 12), color=(255, 255, 255)).save(source_path)
    mocker.patch("tools.segment_tool.SegmentTool._is_sam31_text_prompt_backend_available", return_value=True)
    mocker.patch(
        "tools.segment_tool.sam31_predict_text_prompt_candidates",
        return_value=[
            {
                "name": "sam31_text_0",
                "mask": np.pad(np.ones((8, 8), dtype=bool), 2),
                "score": 0.9,
            }
        ],
    )

    response = sam31_segment_server.handle_segment(
        {
            "image_path": str(source_path),
            "prompt": "subject",
            "output_path": str(output_path),
        }
    )

    assert response["mask_path"] == str(output_path)
    assert response["mask_score"] == 0.9
    assert response["source_stage"] == "remote_sam31"
    assert output_path.is_file()


def test_sam31_segment_server_preload_backend_loads_runtime(mocker) -> None:
    from vision_backends import sam31_segment_server

    preload_mock = mocker.patch("vision_backends.sam31_segment_server.preload_text_prompt_runtime")

    sam31_segment_server.preload_backend()

    preload_mock.assert_called_once_with()


def test_sam31_segment_server_health_reports_loaded_state(monkeypatch: pytest.MonkeyPatch, mocker) -> None:
    from vision_backends import sam31_segment_server

    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", "/tmp/sam31.ckpt")
    monkeypatch.setenv("SAM31_PRELOAD_ON_START", "true")
    mocker.patch(
        "vision_backends.sam31_segment_server.sam3_runtime_status",
        return_value={
            "sam3_image_model_loaded": True,
            "sam3_text_processor_loaded": True,
        },
    )

    payload = sam31_segment_server.health()

    assert payload["status"] == "ok"
    assert payload["service"] == "sam31_segment"
    assert payload["sam3_checkpoint_configured"] is True
    assert payload["sam31_preload_on_start"] is True
    assert payload["sam3_image_model_loaded"] is True
    assert payload["sam3_text_processor_loaded"] is True
    assert payload["sam3_ready"] is True



def test_crop_tool_uses_grounding_preview_branch(tmp_path) -> None:
    from PIL import Image
    from tools.crop_tool import CropTool

    image_path = tmp_path / "source.png"
    Image.new("RGBA", (12, 12), color=(255, 255, 255, 255)).save(image_path)

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="裁剪定位区域",
            )
        },
        "session": SessionState(
            session_id="sess_crop_grounding",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
            "art_geometry_001": GeometryArtifact(
                id="art_geometry_001",
                payload={
                    "image_artifact_id": "art_img_001",
                    "grounding_query": "Locate subject",
                    "candidates": [
                        {
                            "label": "subject",
                            "bbox": [2, 3, 8, 9],
                            "score": 0.9,
                            "positive_points": [],
                            "negative_points": [],
                        }
                    ],
                },
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = CropTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=CropArgs(
            image_ref="art_img_001",
            grounding_ref="art_geometry_001",
            padding=1,
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert artifact.payload["crop_mode"] == "grounding_preview"
    assert artifact.payload["bbox"] == [1, 2, 9, 10]
    assert Path(artifact.uri).is_file()


def test_crop_tool_rejects_mismatched_mask_source(tmp_path) -> None:
    from PIL import Image
    from tools.crop_tool import CropTool

    image_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGBA", (10, 10), color=(255, 255, 255, 255)).save(image_path)
    Image.new("L", (10, 10), color=255).save(mask_path)

    state = {
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="local_edit", instruction="裁剪")
        },
        "session": SessionState(
            session_id="sess_crop_bad_mask",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
            artifact_index=ArtifactIndex(by_type={}),
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(id="art_img_001", uri=str(image_path), payload={"role": "input"}),
            "art_mask_001": MaskArtifact(id="art_mask_001", uri=str(mask_path), payload={"image_ref": "art_img_other"}),
        },
        "operations": [],
        "task_act_records": [],
    }

    with pytest.raises(ValueError, match="does not belong"):
        CropTool().run(
            state,
            task_id="task_001",
            loop_index=1,
            args=CropArgs(image_ref="art_img_001", mask_ref="art_mask_001"),
        )


def test_execute_agent_accepts_grounding_and_collage_strategy(mocker) -> None:
    from PIL import Image
    from tools.collage_tool import CollageLayoutItem, CollageLayoutResult
    from tools.grounding_tool import GroundingTool

    import tempfile

    tmpdir = tempfile.TemporaryDirectory()
    img1 = Path(tmpdir.name) / "1.png"
    img2 = Path(tmpdir.name) / "2.png"
    img3 = Path(tmpdir.name) / "3.png"
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img1)
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img2)
    Image.new("RGBA", (8, 8), color=(255, 255, 255, 255)).save(img3)

    state = {
        "input": {
            "instruction_text": "先定位人物，再整理参考图，最后编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="完成图像编辑",
                input_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
            )
        },
        "session": SessionState(
            session_id="sess_exec_new_tools",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002", "art_img_input_003"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(img1),
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri=str(img2),
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_003": ImageArtifact(
                id="art_img_input_003",
                uri=str(img3),
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch("runtime.input_selector.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "runtime.input_selector.invoke_llm",
        return_value=mocker.Mock(content="Need base image and two references."),
    )
    mocker.patch(
        "runtime.input_selector.invoke_structured_llm",
        return_value=TaskInputSelectionOutput(
            selected_artifact_ids=["art_img_input_001", "art_img_input_002", "art_img_input_003"],
        ),
    )

    invoke_structured = mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        side_effect=[
            ExecuteLLMOutput(reasoning="Need coarse region first.", selected_tools=["grounding"]),
            ExecuteLLMOutput(reasoning="Need one combined reference image.", selected_tools=["collage"]),
            ExecuteLLMOutput(reasoning="Now edit with the prepared references.", selected_tools=["edit"]),
        ],
    )
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=32,
            canvas_height=16,
            background="transparent",
            items=[
                CollageLayoutItem(
                    artifact_id="art_img_input_002",
                    x=0,
                    y=0,
                    width=8,
                    height=8,
                    z_index=0,
                ),
                CollageLayoutItem(
                    artifact_id="art_img_input_003",
                    x=12,
                    y=0,
                    width=8,
                    height=8,
                    z_index=1,
                ),
                CollageLayoutItem(
                    artifact_id="art_img_input_001",
                    x=24,
                    y=0,
                    width=8,
                    height=8,
                    z_index=2,
                ),
            ],
        ),
    )

    def fake_grounding_run(state, *, task_id, loop_index, args):
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.GROUNDING
        fake.invocation.args = args.model_dump()
        fake.invocation.output_refs = ["art_geometry_001"]
        fake.artifacts = [
            GeometryArtifact(
                id="art_geometry_001",
                payload={
                    "image_artifact_id": "art_img_input_001",
                    "grounding_query": args.grounding_query,
                    "candidates": [
                        {
                            "label": "subject",
                            "bbox": [1, 1, 6, 6],
                            "score": 0.95,
                            "positive_points": [],
                            "negative_points": [],
                        }
                    ],
                },
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(GroundingTool, "run", side_effect=fake_grounding_run)

    def fake_edit_run(state, *, task_id, loop_index, args):
        fake = mocker.Mock()
        fake.invocation = ToolInvocationRecord(
            id="op_edit_001",
            task_id=task_id,
            loop_index=loop_index,
            tool_name=ToolName.EDIT,
            args=args.model_dump(),
            status="succeeded",
            output_refs=["art_image_edit_001"],
        )
        fake.artifacts = [
            ImageArtifact(
                id="art_image_edit_001",
                uri=str(img1),
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch("tools.edit_tool.EditTool.run", side_effect=fake_edit_run)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        side_effect=[
            ObserveLLMOutput(outcome="continue", observation="grounding is useful", artifact_summaries=[]),
            ObserveLLMOutput(outcome="continue", observation="collage is useful", artifact_summaries=[]),
            ObserveLLMOutput(outcome="success", observation="candidate is ready", artifact_summaries=[]),
        ],
    )

    result = ExecuteAgent().run(state)
    tmpdir.cleanup()

    tool_names = [op.tool_name for op in result["operations"]]
    assert ToolName.GROUNDING in tool_names
    assert ToolName.COLLAGE in tool_names
    assert result["operations"][-1].tool_name == ToolName.EDIT
    assert invoke_structured.call_count == 3


def test_execute_agent_crop_accepts_grounding_without_mask(mocker) -> None:
    state = {
        "artifacts": {},
        "session": SessionState(
            session_id="sess_crop_from_grounding",
            phase=SessionPhase.EXECUTING,
            task_states={},
        ),
    }

    captured = {}

    def fake_crop_run(state, *, task_id, loop_index, args):
        captured["args"] = args
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.CROP
        fake.invocation.args = args.model_dump()
        fake.invocation.output_refs = ["art_image_crop_001"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_crop_001",
                uri="examples/fig1.jpg",
                payload={"role": "cropped_preview"},
                scope="task",
            )
        ]
        return fake

    mocker.patch("tools.registry.CropTool.run", side_effect=fake_crop_run)

    ExecuteAgent()._run_tool_step(
        state=state,
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.CROP,
        runtime_ctx={
            "initial_base_image_ref": "art_img_input_001",
            "mask_ref": None,
            "grounding_ref": "art_geometry_001",
        },
    )

    assert captured["args"].mask_ref is None
    assert captured["args"].grounding_ref == "art_geometry_001"


def test_execute_agent_segment_passes_grounding_ref(mocker) -> None:
    from tools.segment_tool import SegmentTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="segment the target object",
            )
        },
        "artifacts": {},
        "session": SessionState(
            session_id="sess_segment_from_grounding",
            phase=SessionPhase.EXECUTING,
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=[],
                )
            },
        ),
    }

    captured = {}

    def fake_segment_run(state, *, task_id, loop_index, args):
        captured["args"] = args
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.SEGMENT
        fake.invocation.args = args.model_dump()
        fake.invocation.output_refs = ["art_mask_001"]
        fake.artifacts = [
            MaskArtifact(
                id="art_mask_001",
                uri="generated/segment/task_001_001_mask.png",
                payload={"image_ref": "art_img_input_001", "grounding_ref": "art_geometry_001", "target": "primary_edit_region", "positive_points": [], "negative_points": [], "mask_score": 0.9},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(SegmentTool, "run", side_effect=fake_segment_run)

    ExecuteAgent()._run_tool_step(
        state=state,
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.SEGMENT,
        runtime_ctx={
            "base_image_ref": "art_img_input_001",
            "grounding_ref": "art_geometry_001",
        },
    )

    assert captured["args"].image_ref == "art_img_input_001"
    assert captured["args"].prompt


def test_edit_tool_prefers_instruction_artifact_human_text() -> None:
    from tools.edit_tool import EditTool

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="原始任务指令",
            )
        },
        "session": SessionState(
            session_id="sess_instruction_override",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_inst_001"],
                )
            },
        ),
        "artifacts": {
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={
                    "instruction_text": "将 art_img_input_004 作为背景，将人物放入其中。 Relevant artifacts: art_img_input_004"
                },
            ),
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri="store://images/input.png",
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    execution = EditTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=EditArgs(
            instruction="将人物放到背景里",
            image_refs=["art_img_001"],
        ),
    )

    assert execution.invocation.args["instruction"] == "将人物放到背景里"


def test_local_edit_flow_materializes_task_inputs_and_intermediate_task_artifacts() -> None:
    graph = build_runtime_graph(stop_after_plan=True)
    state = graph.invoke(
        {
            "input": {
                "session_id": "local-edit-chain",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "只修改上衣区域。",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["tasks"]["task_001"].type = "local_edit"
    state["tasks"]["task_001"].input_artifact_ids = ["art_img_input_001"]
    state["session"].current_task_id = "task_001"
    state["session"].task_states["task_001"].status = TaskStatus.RUNNING

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.INSTRUCTION
        for artifact_id in task_state.task_artifact_ids
    )
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.MASK
        for artifact_id in task_state.task_artifact_ids
    )
    assert any(
        result["artifacts"][artifact_id].kind == ArtifactKind.UNDERSTANDING
        for artifact_id in task_state.task_artifact_ids
    )
    assert [loop.selected_tools for loop in result["task_loops"]] == [
        ["segment", "crop", "understand", "edit"],
    ]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert result["session"].task_states["task_001"].latest_execute_checkpoint == "passed"
    assert len(result["task_act_records"]) == 4


def test_evaluator_agent_uses_structured_evaluation_verdict() -> None:
    graph = build_runtime_graph()
    state = graph.invoke(
        {
            "input": {
                "session_id": "eval_llm",
                "image_uri": "store://images/input.png",
                "instruction_text": "change the shirt color",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )
    state["session"].phase = SessionPhase.EVALUATING
    state["session"].current_task_id = "task_001"
    state["session"].final_result_id = None
    candidate_id = state["session"].task_states["task_001"].final_artifact_id or state["session"].final_result_id
    assert candidate_id is not None
    state["session"].task_states["task_001"].latest_artifact_ids = [candidate_id]
    state["session"].task_states["task_001"].latest_execute_checkpoint = "passed"
    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.PASS
    assert result["session"].phase == SessionPhase.DONE
    assert result["session"].current_task_id is None


def test_execute_agent_exhausted_budget_goes_directly_to_replan() -> None:
    state = {
        "input": {
            "instruction_text": "先处理局部，再编辑",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="local_edit",
                instruction="只修改上衣",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_fail",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 2,
    }

    result = ExecuteAgent().run(state)

    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].latest_execute_checkpoint == "failed"
    assert result["decision"].route == DecisionRoute.REPLAN
    assert result["decision"].replan is not None


def test_execute_agent_prompt_reconstruct_success_is_not_evaluable_candidate(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "先改写提示，再编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_prompt_not_candidate",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 1,
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="rewrite prompt first",
            selected_tools=["prompt_reconstruct"],
        ),
    )
    fake_execution = mocker.Mock()
    fake_execution.invocation = ToolInvocationRecord(
        id="op_prompt_reconstruct_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.PROMPT_RECONSTRUCT,
        args={"input_artifact_ids": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_instruction_001"],
    )
    fake_execution.artifacts = [
        InstructionArtifact(
            id="art_instruction_001",
            payload={"instruction_text": "更清晰的编辑提示"},
            created_by=ToolName.PROMPT_RECONSTRUCT.value,
            scope="task",
        )
    ]
    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", return_value=fake_execution)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="prompt is ready",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert task_state.latest_execute_checkpoint == "failed"
    assert task_state.latest_artifact_ids == []
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["decision"].route == DecisionRoute.REPLAN
    assert "art_instruction_001" in task_state.task_artifact_ids


def test_execute_agent_prompt_reconstruct_then_edit_uses_only_image_candidate(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "先改写提示，再编辑",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_prompt_then_edit",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 2,
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    strategies = [
        ExecuteLLMOutput(reasoning="rewrite prompt first", selected_tools=["prompt_reconstruct"]),
        ExecuteLLMOutput(reasoning="now edit", selected_tools=["edit"]),
    ]
    mocker.patch("agents.execute_agent.invoke_structured_llm", side_effect=strategies)
    executions = []
    prompt_execution = mocker.Mock()
    prompt_execution.invocation = ToolInvocationRecord(
        id="op_prompt_reconstruct_001",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.PROMPT_RECONSTRUCT,
        args={"input_artifact_ids": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_instruction_001"],
    )
    prompt_execution.artifacts = [
        InstructionArtifact(
            id="art_instruction_001",
            payload={"instruction_text": "更清晰的编辑提示"},
            created_by=ToolName.PROMPT_RECONSTRUCT.value,
            scope="task",
        )
    ]
    edit_execution = mocker.Mock()
    edit_execution.invocation = ToolInvocationRecord(
        id="op_edit_002",
        task_id="task_001",
        loop_index=1,
        tool_name=ToolName.EDIT,
        args={"instruction": "更清晰的编辑提示", "image_refs": ["art_img_input_001"]},
        status="succeeded",
        output_refs=["art_image_candidate_001"],
    )
    edit_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            created_by=ToolName.EDIT.value,
            scope="task",
        )
    ]
    executions.extend([prompt_execution, edit_execution])
    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", side_effect=executions)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="step done",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert task_state.latest_execute_checkpoint == "passed"
    assert task_state.latest_artifact_ids == ["art_image_candidate_001"]
    assert result["session"].phase == SessionPhase.EVALUATING
    assert [record.tool_name for record in result["task_act_records"][-2:]] == [
        ToolName.PROMPT_RECONSTRUCT.value,
        ToolName.EDIT.value,
    ]


def test_execute_agent_tool_failure_goes_directly_to_replan(mocker) -> None:
    image_path = Path("examples/fig1.jpg")
    state = {
        "input": {
            "instruction_text": "保持图片内容不变",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="global_edit",
                instruction="保持图片内容不变",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_execute_tool_failure",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(image_path),
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
        "max_execute_acts": 3,
    }

    mocker.patch("tools.edit_tool.EditTool.run", side_effect=RuntimeError("backend unavailable"))

    result = ExecuteAgent().run(state)

    task_state = result["session"].task_states["task_001"]
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
    assert task_state.latest_execute_checkpoint == "failed"
    assert task_state.latest_execution_outcome == ExecutionOutcome.FAILURE
    assert result["operations"][-1].status == "failed"
    assert result["operations"][-1].error == {
        "type": "RuntimeError",
        "message": "backend unavailable",
    }
    assert result["decision"].route == DecisionRoute.REPLAN
    assert "execute_tool_failed" in result["decision"].issues
    assert "backend unavailable" in result["decision"].summary


def test_evaluator_agent_failed_candidate_can_continue_same_task() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "continue_execute",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                acceptance_criteria=["subject appears in background"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_continue",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_image_candidate_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    evaluator_checkpoint_count=0,
                    task_artifact_ids=["art_inst_001", "art_image_candidate_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={"instruction_text": "把人物放到背景里"},
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
            ),
        },
        "operations": [],
        "task_act_records": [],
        "max_evaluator_checkpoints": 3,
    }

    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.CONTINUE_EXECUTE
    assert result["decision"].task_retry is not None
    assert result["session"].phase == SessionPhase.EXECUTING
    assert result["session"].current_task_id == "task_001"
    assert result["session"].task_states["task_001"].latest_evaluate_checkpoint == "failed"
    assert "art_img_input_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert "art_image_candidate_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert "art_inst_001" in result["session"].task_states["task_001"].retry_input_artifact_ids
    assert result["session"].task_states["task_001"].retry_context_text is not None
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == []


def test_evaluator_agent_invalid_candidate_replans_without_tool_call() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "pass",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                acceptance_criteria=["subject appears in background"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_invalid_candidate",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_instruction_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    task_artifact_ids=["art_instruction_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_instruction_001": InstructionArtifact(
                id="art_instruction_001",
                payload={"instruction_text": "更清晰的编辑提示"},
                created_by=ToolName.PROMPT_RECONSTRUCT.value,
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    result = EvaluatorAgent().run(state)

    assert result["operations"] == []
    assert result["decision"].route == DecisionRoute.REPLAN
    assert "invalid_evaluation_candidate" in result["decision"].issues
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None
    assert result["session"].task_states["task_001"].status == TaskStatus.REPLANNED



def test_evaluator_agent_budget_exhausted_upgrades_to_replan() -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "desired_decision_route": "continue_execute",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
                acceptance_criteria=["subject appears in background"],
            )
        },
        "session": SessionState(
            session_id="sess_eval_budget",
            phase=SessionPhase.EVALUATING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    latest_artifact_ids=["art_image_candidate_001"],
                    latest_execution_outcome=ExecutionOutcome.SUCCESS,
                    latest_execute_checkpoint="passed",
                    evaluator_checkpoint_count=3,
                    task_artifact_ids=["art_image_candidate_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
            ),
        },
        "operations": [],
        "task_act_records": [],
        "max_evaluator_checkpoints": 3,
    }

    result = EvaluatorAgent().run(state)

    assert result["decision"].route == DecisionRoute.REPLAN
    assert result["decision"].replan is not None
    assert result["session"].phase == SessionPhase.PLANNING
    assert result["session"].current_task_id is None


def test_prepare_task_inputs_prefers_retry_input_artifact_ids() -> None:
    state = {
        "input": {"use_llm": False},
        "session": SessionState(
            session_id="sess_retry_inputs",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=["art_img_input_001", "art_image_candidate_001", "art_inst_001"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正",
            )
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://images/candidate.png",
                payload={"role": "candidate_image"},
            ),
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={"instruction_text": "继续修正当前结果"},
            ),
        },
        "operations": [],
    }

    selection = TaskInputSelectionOutput.model_validate(
        {"selected_artifact_ids": ["placeholder"]}
    )
    from runtime.input_selector import prepare_task_inputs

    result = prepare_task_inputs(state, "task_001")

    assert result.selected_artifact_ids == ["art_img_input_001", "art_image_candidate_001", "art_inst_001"]
    assert state["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_image_candidate_001",
        "art_inst_001",
    ]
    assert state["session"].task_states["task_001"].input_selection_reasoning == "resolved from evaluator retry context"


def test_execute_agent_llm_prompt_includes_retry_context(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "继续修图",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正",
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "session": SessionState(
            session_id="sess_retry_prompt",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    retry_input_artifact_ids=["art_img_input_001", "art_image_candidate_001"],
                    retry_context_text="Evaluator feedback:\n- Reason: 边界不干净\n- Fix focuses:\n  1. 修边界",
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://images/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    captured = {}

    def fake_structured(*, user_prompt, system_prompt, output_schema):
        captured["user_prompt"] = user_prompt
        return ExecuteLLMOutput(
            reasoning="retry with edit",
            selected_tools=["edit"],
        )

    mocker.patch("agents.execute_agent.invoke_structured_llm", side_effect=fake_structured)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent().run(state)

    assert "Retry context:" in captured["user_prompt"]
    assert "边界不干净" in captured["user_prompt"]
    assert result["session"].task_states["task_001"].resolved_input_artifact_ids == [
        "art_img_input_001",
        "art_image_candidate_001",
    ]


def test_execute_agent_observe_injects_artifact_summary(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "use_llm": False,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                input_artifact_ids=["art_img_input_001", "art_img_input_002"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_summary",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/person.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="store://images/background.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.tool_name = ToolName.EDIT
    fake_execution.invocation.args = {
        "instruction": "把人物放到背景里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.invocation.output_refs = ["art_image_candidate_001"]
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_candidate_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", return_value=fake_execution)
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="This candidate is ready for evaluator checkpoint.",
            artifact_summaries=[
                ObserveArtifactSummary(
                    artifact_id="art_image_candidate_001",
                    summary="人物已放入背景的候选图，可交给 evaluator 评估。",
                )
            ],
        ),
    )

    result = ExecuteAgent().run(state)

    artifact = result["artifacts"]["art_image_candidate_001"]
    assert artifact.summary == "人物已放入背景的候选图，可交给 evaluator 评估。"
    assert result["task_act_records"][-1].observation_text == "This candidate is ready for evaluator checkpoint."


def test_understand_tool_uses_multimodal_llm(mocker, tmp_path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    state = {
        "input": {"instruction_text": "理解图片"},
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri=str(image_path),
                payload={"role": "input"},
                scope="session",
            )
        },
        "operations": [],
        "session": SessionState(
            session_id="sess_understand",
            phase=SessionPhase.UNDERSTANDING,
            task_states={},
        ),
    }
    mocker.patch(
        "tools.understand_tool.invoke_multimodal_llm",
        return_value=mocker.Mock(content="图片中是一个站立的人物。"),
    )

    result = build_default_tool_registry().get(ToolName.UNDERSTAND).run(
        state,
        task_id="bootstrap",
        loop_index=0,
        args=UnderstandArgs(image_ref="art_img_input_001", question="图里有什么"),
    )

    assert result.artifacts[0].payload["summary"] == "图片中是一个站立的人物。"


def test_evaluate_tool_uses_multimodal_llm(mocker, tmp_path) -> None:
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    state = {
        "artifacts": {
            "art_image_001": ImageArtifact(
                id="art_image_001",
                uri=str(image_path),
                payload={"role": "candidate_image"},
                scope="task",
            ),
            "art_instruction_task_001_001": InstructionArtifact(
                id="art_instruction_task_001_001",
                payload={"instruction_text": "检查是否满足要求"},
                scope="task",
            ),
        },
        "operations": [],
        "session": SessionState(
            session_id="sess_evaluate",
            phase=SessionPhase.EVALUATING,
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.WAITING_EVALUATION,
                    task_artifact_ids=["art_instruction_task_001_001"],
                )
            },
        ),
    }
    mocker.patch(
        "tools.evaluate_tool.invoke_structured_multimodal_llm",
        return_value=EvaluateLLMOutput(
            is_satisfied=True,
            scores=EvaluationScores(
                instruction_success=4,
                reference_consistency=4,
                overediting=4,
                naturalness=4,
                artifacts=4,
            ),
            reason="候选图满足大部分要求，人物已进入背景。",
            issues=[],
            new_rewritten_prompt=None,
        ),
    )

    result = build_default_tool_registry().get(ToolName.EVALUATE).run(
        state,
        task_id="task_001",
        loop_index=1,
        args=EvaluateArgs(candidate_refs=["art_image_001"], checks=["人物在背景里"]),
    )

    assert result.artifacts[0].payload["reason"] == "候选图满足大部分要求，人物已进入背景。"
    assert result.artifacts[0].payload["verdict"] == "pass"
    assert result.artifacts[0].payload["scores"]["weighted_score"] == 4


def test_execute_agent_observe_uses_multimodal_when_image_artifact_exists(mocker, tmp_path) -> None:
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\x0b\xe7\x02\x9d"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    state = {
        "input": {"instruction_text": "把人物放到背景里", "use_llm": True},
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                acceptance_criteria=["人物位于背景中"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_mm",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "artifacts": {},
    }
    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.args = {
        "instruction": "将人物主体放到背景图里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_001",
            uri=str(image_path),
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]
    mocker.patch(
        "agents.execute_agent.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )
    multimodal_call = mocker.patch(
        "agents.execute_agent.invoke_structured_multimodal_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="candidate looks good",
            artifact_summaries=[],
        ),
    )

    result = ExecuteAgent()._observe_with_llm(
        state=state,
        task=state["tasks"]["task_001"],
        task_id="task_001",
        selected_tool=ToolName.EDIT,
        execution=fake_execution,
    )

    assert result.outcome == "success"
    multimodal_call.assert_called_once()


def test_execute_agent_observe_rejects_non_local_image_uri(mocker) -> None:
    state = {
        "input": {"instruction_text": "把人物放到背景里", "use_llm": True},
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                acceptance_criteria=["人物位于背景中"],
            )
        },
        "session": SessionState(
            session_id="sess_observe_bad_uri",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={"task_001": TaskState(task_id="task_001", status=TaskStatus.RUNNING)},
        ),
        "artifacts": {},
    }
    fake_execution = mocker.Mock()
    fake_execution.invocation = mocker.Mock()
    fake_execution.invocation.args = {
        "instruction": "将人物主体放到背景图里",
        "image_refs": ["art_img_input_001", "art_img_input_002"],
    }
    fake_execution.artifacts = [
        ImageArtifact(
            id="art_image_001",
            uri="store://generated/task_001/candidate.png",
            payload={"role": "candidate_image"},
            scope="task",
        )
    ]
    mocker.patch(
        "agents.execute_agent.load_llm_config",
        return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"),
    )

    with pytest.raises(ValueError, match="not a local file path"):
        ExecuteAgent()._observe_with_llm(
            state=state,
            task=state["tasks"]["task_001"],
            task_id="task_001",
            selected_tool=ToolName.EDIT,
            execution=fake_execution,
        )


def test_execute_agent_uses_llm_selected_base_image_artifact_id(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "把人物放到背景里",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="将人物主体放到背景图里",
                input_artifact_ids=["art_img_input_001", "art_img_input_002"],
            )
        },
        "session": SessionState(
            session_id="sess_base_select",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_img_input_002"],
                    task_artifact_ids=["art_under_001", "art_under_002"],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_img_input_002"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/person.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="store://images/background.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_under_001": UnderstandingArtifact(
                id="art_under_001",
                payload={"image_ref": "art_img_input_001", "summary": "人物主体图"},
                scope="task",
            ),
            "art_under_002": UnderstandingArtifact(
                id="art_under_002",
                payload={"image_ref": "art_img_input_002", "summary": "拍照背景图"},
                scope="task",
            ),
        },
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="use the background as base and the person as reference",
            selected_tools=["edit"],
            base_image_artifact_id="art_img_input_002",
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    captured = {}

    def fake_select_and_run_tool(self, *, state, task, task_id, loop_index, selected_tool, resolved_inputs, base_image_ref):
        captured["base_image_ref"] = base_image_ref
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.EDIT
        fake.invocation.args = {
            "instruction": "把人物放到背景里",
            "image_refs": ["art_img_input_002", "art_img_input_001"],
        }
        fake.invocation.output_refs = ["art_image_candidate_001"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", fake_select_and_run_tool)

    ExecuteAgent().run(state)

    assert captured["base_image_ref"] == "art_img_input_002"


def test_execute_agent_retry_candidate_has_priority_as_base_image(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "继续修正失败结果",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="继续修正当前结果",
            )
        },
        "session": SessionState(
            session_id="sess_retry_base",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    resolved_input_artifact_ids=["art_img_input_001", "art_image_candidate_001"],
                    task_artifact_ids=[],
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_image_candidate_001"]}),
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/original.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_image_candidate_001": ImageArtifact(
                id="art_image_candidate_001",
                uri="store://generated/task_001/candidate.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_001",
            route=DecisionRoute.CONTINUE_EXECUTE,
            task_id="task_001",
            summary="继续修正",
            task_retry=TaskRetryAdvice(
                reason="继续沿失败结果修正",
                base_candidate_artifact_id="art_image_candidate_001",
                fix_focuses=["修边界"],
            ),
        ),
        "operations": [],
        "task_act_records": [],
        "task_loops": [],
    }

    mocker.patch("agents.execute_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.execute_agent.invoke_structured_llm",
        return_value=ExecuteLLMOutput(
            reasoning="even if another image exists, continue from failed candidate",
            selected_tools=["edit"],
            base_image_artifact_id="art_img_input_001",
        ),
    )
    mocker.patch.object(
        ExecuteAgent,
        "_observe_with_llm",
        return_value=ObserveLLMOutput(
            outcome="success",
            observation="ready for evaluator",
            artifact_summaries=[],
        ),
    )

    captured = {}

    def fake_select_and_run_tool(self, *, state, task, task_id, loop_index, selected_tool, resolved_inputs, base_image_ref):
        captured["base_image_ref"] = base_image_ref
        fake = mocker.Mock()
        fake.invocation = mocker.Mock()
        fake.invocation.tool_name = ToolName.EDIT
        fake.invocation.args = {
            "instruction": "继续修正当前结果",
            "image_refs": ["art_image_candidate_001", "art_img_input_001"],
        }
        fake.invocation.output_refs = ["art_image_candidate_002"]
        fake.artifacts = [
            ImageArtifact(
                id="art_image_candidate_002",
                uri="store://generated/task_001/candidate_retry.png",
                payload={"role": "candidate_image"},
                scope="task",
            )
        ]
        return fake

    mocker.patch.object(ExecuteAgent, "_select_and_run_tool", fake_select_and_run_tool)

    ExecuteAgent().run(state)

    assert captured["base_image_ref"] == "art_image_candidate_001"


def test_execute_agent_build_edit_image_refs_prefers_base_and_caps_at_three() -> None:
    state = {
        "session": SessionState(
            session_id="sess_edit_refs",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_collage_001", "art_crop_001"],
                )
            },
        ),
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="examples/fig1.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_002": ImageArtifact(
                id="art_img_input_002",
                uri="examples/fig2.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_img_input_003": ImageArtifact(
                id="art_img_input_003",
                uri="examples/fig3.jpg",
                payload={"role": "input"},
                scope="session",
            ),
            "art_collage_001": ImageArtifact(
                id="art_collage_001",
                uri="examples/fig4.jpg",
                payload={"role": "collage_reference"},
                scope="task",
            ),
            "art_crop_001": ImageArtifact(
                id="art_crop_001",
                uri="examples/fig1.jpg",
                payload={"role": "cropped_preview"},
                scope="task",
            ),
        },
    }

    image_refs = ExecuteAgent()._build_edit_image_refs(
        state=state,
        task_id="task_001",
        runtime_ctx={
            "base_image_ref": "art_img_input_001",
            "initial_base_image_ref": "art_img_input_001",
            "reference_refs": ["art_img_input_002", "art_img_input_003"],
            "grounding_ref": None,
            "mask_ref": None,
            "crop_ref": "art_crop_001",
        },
    )

    assert image_refs == ["art_img_input_001", "art_collage_001", "art_crop_001"]


def test_edit_tool_generates_local_candidate_image_with_unified_args(tmp_path, mocker) -> None:
    from PIL import Image
    from tools.edit_tool import EditTool

    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(source_path)

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
            )
        },
        "session": SessionState(
            session_id="sess_edit_tool",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_inst_001"],
                )
            },
        ),
        "artifacts": {
            "art_inst_001": InstructionArtifact(
                id="art_inst_001",
                payload={"instruction_text": "把人物放到背景里"},
            ),
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(source_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    mocker.patch(
        "tools.edit_tool.edit_images",
        return_value=Image.new("RGB", (16, 16), color=(0, 128, 255)),
    )
    mocker.patch(
        "tools.edit_tool.backend_config_snapshot",
        return_value={"model_path": "mock-firered"},
    )

    execution = EditTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=EditArgs(
            instruction="把人物放到背景里",
            image_refs=["art_img_001"],
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.payload["role"] == "candidate_image"
    assert artifact.payload["primary_image_ref"] == "art_img_001"
    assert artifact.payload["auxiliary_image_refs"] == []
    assert artifact.payload["backend_name"] == "firered"
    assert artifact.payload["source"] == "edit_output"
    assert artifact.source_ids == ["art_img_001"]
    assert Path(artifact.uri).is_file()
    assert execution.invocation.args == {
        "instruction": "把人物放到背景里",
        "image_refs": ["art_img_001"],
    }


def test_edit_tool_uses_remote_backend_when_configured(tmp_path, mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image
    from tools.edit_tool import EditTool
    from vision_backends.remote_client import RemoteEditResult

    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(source_path)
    monkeypatch.setenv("EDIT_BACKEND", "remote")

    state = {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="把人物放到背景里",
            )
        },
        "session": SessionState(
            session_id="sess_edit_remote",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=[],
                )
            },
        ),
        "artifacts": {
            "art_img_001": ImageArtifact(
                id="art_img_001",
                uri=str(source_path),
                payload={"role": "input"},
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    def fake_remote_edit(*, image_paths, instruction, output_path):
        Image.new("RGB", (16, 16), color=(0, 128, 255)).save(output_path)
        return RemoteEditResult(
            output_path=output_path,
            backend_config={"backend": "remote-firered"},
        )

    remote_mock = mocker.patch("tools.edit_tool.request_firered_edit", side_effect=fake_remote_edit)

    execution = EditTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=EditArgs(
            instruction="把人物放到背景里",
            image_refs=["art_img_001"],
        ),
    )

    artifact = execution.artifacts[0]
    remote_mock.assert_called_once()
    assert artifact.payload["backend_name"] == "firered_remote"
    assert artifact.payload["backend_config_snapshot"] == {"backend": "remote-firered"}
    assert Path(artifact.uri).is_file()


def test_firered_edit_server_endpoint_writes_output(tmp_path, mocker) -> None:
    from PIL import Image
    from vision_backends import firered_edit_server

    source_path = tmp_path / "source.png"
    output_path = tmp_path / "out.png"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(source_path)
    mocker.patch("vision_backends.firered_edit_server.edit_images", return_value=Image.new("RGB", (16, 16), color=(0, 128, 255)))
    mocker.patch("vision_backends.firered_edit_server.backend_config_snapshot", return_value={"backend": "mock"})

    response = firered_edit_server.handle_edit(
        {
            "image_paths": [str(source_path)],
            "instruction": "edit",
            "output_path": str(output_path),
        }
    )

    assert response == {"output_path": str(output_path), "backend_config": {"backend": "mock"}}
    assert output_path.is_file()


def test_firered_edit_server_preload_backend_loads_pipeline(mocker) -> None:
    from vision_backends import firered_edit_server

    load_mock = mocker.patch("vision_backends.firered_edit_server.load_pipeline")

    firered_edit_server.preload_backend()

    load_mock.assert_called_once_with()


def test_firered_edit_server_health_reports_loaded_state(monkeypatch: pytest.MonkeyPatch, mocker) -> None:
    from vision_backends import firered_edit_server

    monkeypatch.setenv("FIRERED_PRELOAD_ON_START", "true")
    mocker.patch("vision_backends.firered_edit_server.is_pipeline_loaded", return_value=True)

    payload = firered_edit_server.health()

    assert payload["status"] == "ok"
    assert payload["service"] == "firered_edit"
    assert payload["firered_cached"] is True
    assert payload["firered_preload_on_start"] is True


def test_remote_client_uses_service_specific_base_urls(monkeypatch: pytest.MonkeyPatch, mocker) -> None:
    from vision_backends import remote_client

    monkeypatch.setenv("FIRERED_EDIT_BACKEND_BASE_URL", "http://edit.local")
    monkeypatch.setenv("SAM31_SEGMENT_BACKEND_BASE_URL", "http://sam.local")
    monkeypatch.setenv("VISION_BACKEND_BASE_URL", "http://shared.local")
    captured_urls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            if captured_urls[-1].endswith("/v1/edit/firered"):
                return b'{"output_path":"generated/edit/out.png","backend_config":{}}'
            return b'{"mask_path":"generated/segment/mask.png","mask_score":0.5,"selection_metrics":{}}'

    def fake_urlopen(req, timeout):
        captured_urls.append(req.full_url)
        return FakeResponse()

    mocker.patch("vision_backends.remote_client.request.urlopen", side_effect=fake_urlopen)

    remote_client.request_firered_edit(image_paths=["in.png"], instruction="edit")
    remote_client.request_sam31_segment(image_path="in.png", prompt="subject")

    assert captured_urls == [
        "http://edit.local/v1/edit/firered",
        "http://sam.local/v1/segment/sam31",
    ]


def test_remote_client_defaults_to_remote_split_services(monkeypatch: pytest.MonkeyPatch) -> None:
    from vision_backends.remote_client import remote_base_url, resolve_edit_backend, resolve_segment_backend

    monkeypatch.delenv("EDIT_BACKEND", raising=False)
    monkeypatch.delenv("SEGMENT_BACKEND", raising=False)
    monkeypatch.delenv("FIRERED_EDIT_BACKEND_BASE_URL", raising=False)
    monkeypatch.delenv("SAM31_SEGMENT_BACKEND_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_BACKEND_BASE_URL", raising=False)

    assert resolve_edit_backend() == "remote"
    assert resolve_segment_backend() == "remote"
    assert remote_base_url("edit") == "http://127.0.0.1:8765"
    assert remote_base_url("segment") == "http://127.0.0.1:8766"


def test_plan_agent_replan_creates_new_plan_version_and_marks_old_tasks() -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_replan",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            latest_decision_id="dec_001",
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
                "task_003": TaskState(task_id="task_003", status=TaskStatus.PENDING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002", "task_003"],
                input_artifact_ids=["art_img_input_001"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
            "task_003": Task(id="task_003", plan_id="plan_001", type="future", instruction="未来 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                summary="原始输入人物图",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                summary="已保留的人物主体候选图",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_001",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["plans"]["plan_001"].task_ids == ["task_001", "task_002", "task_003"]
    assert result["session"].task_states["task_001"].status == TaskStatus.PASSED
    assert result["session"].task_states["task_002"].status == TaskStatus.REPLANNED
    assert result["session"].task_states["task_003"].status == TaskStatus.ABANDONED
    assert "task_001" in result["plans"]["plan_002"].task_ids
    assert any(task_id.startswith("task_00") and task_id not in {"task_001", "task_002", "task_003"} for task_id in result["plans"]["plan_002"].task_ids)
    assert "art_keep_001" in result["plans"]["plan_002"].input_artifact_ids
    new_task_ids = [
        task_id
        for task_id in result["plans"]["plan_002"].task_ids
        if task_id != "task_001"
    ]
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_selects_first_runnable_new_task() -> None:
    state = {
        "input": {
            "instruction_text": "重排路线",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_replan_runnable",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="done", instruction="旧已完成"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="旧失败"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "decision": Decision(
            id="dec_002",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="重排顺序",
            replan=ReplanRequest(
                mode=ReplanMode.REROUTE_PLAN,
                reason="后续路线需要重排",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=[],
            ),
        ),
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["session"].current_task_id is not None
    assert result["session"].task_states[result["session"].current_task_id].status == TaskStatus.RUNNING


def test_plan_agent_initial_planning_uses_incrementing_plan_id() -> None:
    state = {
        "input": {
            "instruction_text": "新建计划",
            "use_llm": False,
        },
        "session": SessionState(
            session_id="sess_initial_increment",
            phase=SessionPhase.PLANNING,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001"]),
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="old", instruction="旧 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "task_act_records": [],
        "task_loops": [],
        "operations": [],
    }

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"


def test_plan_agent_invalid_llm_output_falls_back_to_replan_template(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_invalid_replan_llm",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(
                id="plan_001",
                instruction="旧计划",
                task_ids=["task_001", "task_002"],
            )
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_003",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction='[{"id":"task_999","type":"bad"}]',
            tasks=[
                PlanTaskSpec(
                    id="task_001",
                    type="bad",
                    instruction="bad",
                    input_artifact_ids=[],
                    depends_on=[],
                    acceptance_criteria=["bad"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["session"].current_plan_id == "plan_002"
    assert result["plans"]["plan_002"].instruction.startswith("Replan mode:")
    new_task_ids = [task_id for task_id in result["plans"]["plan_002"].task_ids if task_id != "task_001"]
    assert len(new_task_ids) == 2
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_rejects_illegal_llm_input_artifact_id(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_replan_illegal_input",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001", "task_002"])
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                payload={"role": "input"},
                scope="session",
            ),
        },
        "decision": Decision(
            id="dec_illegal_input",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要重做",
            replan=ReplanRequest(
                mode=ReplanMode.REROUTE_PLAN,
                reason="输入绑定不应由 planner 决定",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=[],
            ),
        ),
    }

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch(
        "agents.plan_agent.invoke_structured_llm",
        return_value=PlanLLMOutput(
            plan_instruction="补全后续任务",
            tasks=[
                PlanTaskSpec(
                    id="task_003",
                    type="bad_input_binding",
                    instruction="错误绑定不存在的输入",
                    input_artifact_ids=["art_future_999"],
                    depends_on=["task_001"],
                    acceptance_criteria=["完成"],
                )
            ],
        ),
    )

    result = PlanAgent().run(state)

    assert result["plans"]["plan_002"].instruction.startswith("Replan mode:")
    new_task_ids = [task_id for task_id in result["plans"]["plan_002"].task_ids if task_id != "task_001"]
    assert len(new_task_ids) == 2
    assert all(result["tasks"][task_id].input_artifact_ids == [] for task_id in new_task_ids)


def test_plan_agent_replan_prompt_includes_available_artifact_summaries(mocker) -> None:
    state = {
        "input": {
            "instruction_text": "重新规划当前路线",
            "use_llm": True,
        },
        "session": SessionState(
            session_id="sess_replan_artifact_catalog",
            phase=SessionPhase.PLANNING,
            current_plan_id="plan_001",
            current_task_id=None,
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_img_input_001", "art_keep_001"]}),
            task_states={
                "task_001": TaskState(task_id="task_001", status=TaskStatus.PASSED),
                "task_002": TaskState(task_id="task_002", status=TaskStatus.RUNNING),
            },
        ),
        "plans": {
            "plan_001": Plan(id="plan_001", instruction="旧计划", task_ids=["task_001", "task_002"])
        },
        "tasks": {
            "task_001": Task(id="task_001", plan_id="plan_001", type="keep", instruction="保留 task"),
            "task_002": Task(id="task_002", plan_id="plan_001", type="failed", instruction="失败 task"),
        },
        "artifacts": {
            "art_img_input_001": ImageArtifact(
                id="art_img_input_001",
                uri="store://images/input.png",
                summary="原始输入人物图",
                payload={"role": "input"},
                scope="session",
            ),
            "art_keep_001": ImageArtifact(
                id="art_keep_001",
                uri="store://images/keep.png",
                summary="已保留的人物主体候选图",
                payload={"role": "candidate_image"},
                scope="task",
            ),
        },
        "decision": Decision(
            id="dec_004",
            route=DecisionRoute.REPLAN,
            task_id="task_002",
            plan_id="plan_001",
            summary="需要拆分 task",
            replan=ReplanRequest(
                mode=ReplanMode.SPLIT_TASK,
                reason="当前 task 太粗",
                preserve_task_ids=["task_001"],
                preserve_artifact_ids=["art_keep_001"],
            ),
        ),
    }

    captured = {}

    def fake_structured(*, user_prompt, system_prompt, output_schema):
        captured["user_prompt"] = user_prompt
        return PlanLLMOutput(
            plan_instruction="补全后续任务",
            tasks=[
                PlanTaskSpec(
                    id="task_003",
                    type="prepare_refined_inputs",
                    instruction="准备后续输入",
                    input_artifact_ids=[],
                    depends_on=["task_001"],
                    acceptance_criteria=["输入准备完成"],
                )
            ],
        )

    mocker.patch("agents.plan_agent.load_llm_config", return_value=mocker.Mock(api_key="k", base_url="u", model_name="m"))
    mocker.patch("agents.plan_agent.invoke_structured_llm", side_effect=fake_structured)

    result = PlanAgent().run(state)

    assert "Available artifacts for planning:" in captured["user_prompt"]
    assert "art_keep_001" in captured["user_prompt"]
    assert "已保留的人物主体候选图" in captured["user_prompt"]
    assert result["plans"]["plan_002"].task_ids == ["task_001", "task_003"]
    assert result["tasks"]["task_003"].input_artifact_ids == []


def test_runtime_run_logger_writes_jsonl(tmp_path, monkeypatch, capsys) -> None:
    import json

    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LOG_ENABLED", "true")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "true")

    graph = build_runtime_graph(stop_after_plan=True)
    result = graph.invoke(
        {
            "input": {
                "session_id": "log-smoke",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "记录一次日志",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    captured = capsys.readouterr()
    assert "[agent:" in captured.out
    assert "run_start" in captured.out
    assert result["run_id"]
    assert result["run_log_uri"] is not None

    log_path = Path(result["run_log_uri"])
    assert log_path.is_file()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["event"] == "run_start"
    assert any(record["event"] == "node_start" and record["payload"]["node"] == "plan" for record in records)
    assert any(record["event"] == "plan_created" for record in records)
    assert all(record["run_id"] == result["run_id"] for record in records)


def test_runtime_run_logger_can_disable_file_and_console(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LOG_ENABLED", "false")
    monkeypatch.setenv("AGENT_LOG_CONSOLE", "false")

    result = build_runtime_graph(stop_after_plan=True).invoke(
        {
            "input": {
                "session_id": "log-disabled",
                "image_uri": "examples/fig1.jpg",
                "image_uris": ["examples/fig1.jpg"],
                "instruction_text": "不要写日志",
                "desired_decision_route": "pass",
                "use_llm": False,
            }
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert result["run_id"]
    assert result["run_log_uri"] is None
    assert list(tmp_path.iterdir()) == []


def test_prompt_module_builds_plan_prompt() -> None:
    from runtime.prompts import PLAN_SYSTEM_PROMPT, build_plan_user_prompt

    prompt = build_plan_user_prompt(
        instruction_text="把人物放到背景里",
        replan_context="Replan mode: split_task\n",
        retained_prefix_plan="Retained prefix task ids: ['task_001']\n",
        available_artifacts="Available artifacts for planning:\n- art_image_001\n",
        image_artifact_ids=["art_img_input_001"],
        understanding_summaries=[{"summary": "人物参考图"}],
    )

    assert "planning agent" in PLAN_SYSTEM_PROMPT
    assert "3-4 tasks" in PLAN_SYSTEM_PROMPT
    assert "Avoid over-splitting" in PLAN_SYSTEM_PROMPT
    assert "Instruction: 把人物放到背景里" in prompt
    assert "Available artifacts for planning" in prompt
    assert "1-2 edit attempts" in prompt
    assert "Do not invent future artifact ids" in prompt


def test_prompt_module_builds_evaluate_prompt_with_rubric() -> None:
    from runtime.prompts import EVALUATE_SCORE_RUBRIC, build_evaluate_user_prompt

    prompt = build_evaluate_user_prompt(
        reference_text="Image 1 is reference. Image 2 is candidate.",
        input_refs=["art_img_input_001"],
        candidate_ref="art_image_001",
        instruction="保持人物身份",
        checks=["身份一致"],
    )

    assert EVALUATE_SCORE_RUBRIC in prompt
    assert "minor imperfections" in EVALUATE_SCORE_RUBRIC.lower()
    assert "Candidate ref: art_image_001" in prompt
    assert "Set is_satisfied=true" in prompt
    assert "severe route failure" in prompt


def test_prompt_module_includes_execute_convergence_guidance() -> None:
    from runtime.prompts import (
        EXECUTE_OBSERVE_SYSTEM_PROMPT,
        EXECUTE_STRATEGY_SYSTEM_PROMPT,
        build_execute_observe_user_prompt,
        build_execute_strategy_user_prompt,
    )

    task = Task(
        id="task_prompt",
        plan_id="plan_prompt",
        type="reference_edit",
        instruction="把人物放到背景里",
    )
    strategy_prompt = build_execute_strategy_user_prompt(
        task=task,
        user_instruction="生成最终图",
        resolved_ids=["art_img_input_001"],
        resolved_image_summaries="- art_img_input_001: 人物",
        retry_context=None,
        active_instruction="把人物放到背景里",
        task_artifact_context="(none)",
        latest_candidate_refs=[],
    )
    observe_prompt = build_execute_observe_user_prompt(
        task=task,
        active_instruction="把人物放到背景里",
        retry_context=None,
        selected_tool="edit",
        tool_args={"image_refs": ["art_img_input_001"]},
        source_lines=[],
        new_artifact_lines=["- art_image_001 | kind=image"],
    )

    assert "Target 1-2 edit attempts per task" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "Prefer edit" in EXECUTE_STRATEGY_SYSTEM_PROMPT
    assert "let evaluator decide" in EXECUTE_OBSERVE_SYSTEM_PROMPT
    assert "1-2 edit attempts" in strategy_prompt
    assert "prefer success and let evaluator decide" in observe_prompt
