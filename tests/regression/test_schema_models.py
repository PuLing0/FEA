from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


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


def test_evaluate_llm_output_accepts_pass_with_issues() -> None:
    output = EvaluateLLMOutput(
        is_satisfied=False,
        verdict="pass_with_issues",
        scores=_evaluation_scores(4, naturalness=3),
        reason="核心目标已完成，但仍有轻微视觉问题。",
        issues=["minor visual roughness"],
    )

    assert output.verdict == "pass_with_issues"


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
