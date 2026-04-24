from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from schema import (
    ArtifactIndex,
    GroundingPoint,
    ImageArtifact,
    SegmentArgs,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
)
from tools.segment_tool import SegmentTool
from vision_backends import sam3_point_backend
from vision_backends.config import settings


DEFAULT_SAM31_CHECKPOINT = Path(
    "/mnt/sda/sijuzheng/models/facebook/sam3.1/sam3.1_multiplex.pt"
)
DEFAULT_EXAMPLE_IMAGE = Path("examples/fig1.jpg")


def _require_cuda_and_assets() -> tuple[Path, Path]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real SAM 3.1 integration test.")

    checkpoint_path = Path(
        os.getenv("SAM3_CHECKPOINT_PATH", str(DEFAULT_SAM31_CHECKPOINT))
    )
    if not checkpoint_path.is_file():
        pytest.skip(f"SAM3 checkpoint not found: {checkpoint_path}")

    image_path = Path(os.getenv("SAM31_TEST_IMAGE", str(DEFAULT_EXAMPLE_IMAGE)))
    if not image_path.is_file():
        pytest.skip(f"SAM3 test image not found: {image_path}")

    return checkpoint_path, image_path


def _configure_real_sam31_backend(checkpoint_path: Path) -> None:
    settings.sam3_checkpoint_path = str(checkpoint_path)
    settings.sam3_model_version = "sam3.1"
    settings.sam3_load_from_hf = False
    settings.sam3_device = "cuda"
    settings.sam3_compile = False
    os.environ["SAM3_REPO_ROOT"] = "/mnt/sda/sijuzheng/project/sam3"
    sam3_point_backend._load_model.cache_clear()
    sam3_point_backend._load_runtime_modules.cache_clear()


def test_sam31_predict_candidates_with_local_checkpoint() -> None:
    checkpoint_path, image_path = _require_cuda_and_assets()
    _configure_real_sam31_backend(checkpoint_path)

    image_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    positive_points = [GroundingPoint(x=545, y=1180)]
    negative_points = [GroundingPoint(x=120, y=780)]

    candidates = sam3_point_backend.predict_candidates(
        image_array=image_array,
        positive_points=positive_points,
        negative_points=negative_points,
    )

    assert candidates
    assert any(float(candidate["score"]) >= 0.0 for candidate in candidates)
    assert any(
        np.asarray(candidate["mask"], dtype=bool)[1180, 545]
        for candidate in candidates
    )


def test_segment_tool_runs_with_real_sam31() -> None:
    checkpoint_path, image_path = _require_cuda_and_assets()
    _configure_real_sam31_backend(checkpoint_path)

    state = {
        "tasks": {
            "task_sam31": Task(
                id="task_sam31",
                plan_id="plan_sam31",
                type="local_edit",
                instruction="segment the woman wearing white clothes",
            )
        },
        "session": SessionState(
            session_id="sess_sam31",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_sam31",
            current_task_id="task_sam31",
            task_states={
                "task_sam31": TaskState(
                    task_id="task_sam31",
                    status=TaskStatus.RUNNING,
                )
            },
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

    execution = SegmentTool().run(
        state,
        task_id="task_sam31",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="woman in white blouse",
            backend_name="sam31",
        ),
    )

    artifact = execution.artifacts[0]
    assert artifact.payload["image_ref"] == "art_img_001"
    assert artifact.payload["prompt"] == "woman in white blouse"
    assert float(artifact.payload["mask_score"]) >= 0.0
    assert artifact.payload["source_stage"] == "sam_text_only"
    assert Path(artifact.uri).is_file()

    with Image.open(artifact.uri) as mask_image:
        assert mask_image.mode == "L"
        assert mask_image.getbbox() is not None
        bbox = mask_image.getbbox()
        assert bbox is not None
        left, top, right, bottom = bbox
        assert right > left
        assert bottom > top
