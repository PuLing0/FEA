from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _evaluation_scores,
    _make_instruction_resolution_state,
    _run_tool,
)


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


def test_tool_registry_contains_prompt_reconstruct() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.PROMPT_RECONSTRUCT) is not None


def test_tool_registry_contains_grounding_and_collage() -> None:
    registry = build_default_tool_registry()
    assert registry.get(ToolName.GROUNDING) is not None
    assert registry.get(ToolName.COLLAGE) is not None


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
            execution = _run_tool(
                GroundingTool(),
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

    result = _run_tool(
        GroundingTool(),
        state,
        task_id="task_001",
        loop_index=1,
        args=GroundingArgs(
            image_ref="art_img_001",
            grounding_query="Locate the person torso",
            top_k=1,
        ),
    )

    _assert_tool_failed(result, error_type="ValueError", message="not a local file path")


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

    execution = _run_tool(
        CollageTool(),
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

    execution = _run_tool(
        CropTool(),
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


def test_crop_tool_uses_unique_output_paths_within_same_loop(tmp_path) -> None:
    from PIL import Image
    from tools.crop_tool import CropTool

    image_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGBA", (10, 10), color=(255, 0, 0, 255)).save(image_path)
    Image.new("L", (10, 10), color=255).save(mask_path)

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
            session_id="sess_crop_unique",
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

    first = _run_tool(
        CropTool(),
        state,
        task_id="task_001",
        loop_index=1,
        args=CropArgs(image_ref="art_img_001", mask_ref="art_mask_001"),
    )
    for artifact in first.artifacts:
        state["artifacts"][artifact.id] = artifact
    second = _run_tool(
        CropTool(),
        state,
        task_id="task_001",
        loop_index=1,
        args=CropArgs(image_ref="art_img_001", mask_ref="art_mask_001"),
    )

    first_uri = first.artifacts[0].uri
    second_uri = second.artifacts[0].uri
    assert first_uri != second_uri
    assert first.artifacts[0].id in first_uri
    assert second.artifacts[0].id in second_uri
    assert Path(first_uri).is_file()
    assert Path(second_uri).is_file()


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

    execution = _run_tool(
        SegmentTool(),
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

    execution = _run_tool(
        SegmentTool(),
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

    execution = _run_tool(
        SegmentTool(),
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

    execution = _run_tool(
        SegmentTool(),
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

    execution = _run_tool(
        CropTool(),
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

    result = _run_tool(
        CropTool(),
        state,
        task_id="task_001",
        loop_index=1,
        args=CropArgs(image_ref="art_img_001", mask_ref="art_mask_001"),
    )

    _assert_tool_failed(result, error_type="ValueError", message="does not belong")


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

    execution = _run_tool(
        EditTool(),
        state,
        task_id="task_001",
        loop_index=1,
        args=EditArgs(
            instruction="将人物放到背景里",
            image_refs=["art_img_001"],
        ),
    )

    assert execution.invocation.args["instruction"] == "将人物放到背景里"


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

    registry = build_default_tool_registry()
    result = ToolRunner(registry).run(
        state,
        ToolName.UNDERSTAND,
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

    registry = build_default_tool_registry()
    result = ToolRunner(registry).run(
        state,
        ToolName.EVALUATE,
        task_id="task_001",
        loop_index=1,
        args=EvaluateArgs(candidate_refs=["art_image_001"], checks=["人物在背景里"]),
    )

    assert result.artifacts[0].payload["reason"] == "候选图满足大部分要求，人物已进入背景。"
    assert result.artifacts[0].payload["verdict"] == "pass"
    assert result.artifacts[0].payload["scores"]["weighted_score"] == 4


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

    execution = _run_tool(
        EditTool(),
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

    execution = _run_tool(
        EditTool(),
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
