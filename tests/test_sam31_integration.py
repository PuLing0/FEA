from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from schema import (
    ArtifactIndex,
    ImageArtifact,
    SegmentArgs,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
)
from runtime.tool_runner import ToolRunner
from tools.segment_tool import SegmentTool
from tools.registry import ToolRegistry
from vision_backends import sam3_point_backend
from vision_backends.config import settings


DEFAULT_SAM31_CHECKPOINT = Path(
    "/mnt/sda/sijuzheng/models/facebook/sam3.1/sam3.1_multiplex.pt"
)
DEFAULT_SAM3_REPO_ROOT = Path("/mnt/sda/sijuzheng/project/sam3")
DEFAULT_EXAMPLE_IMAGE = Path("examples/fig1.jpg")


def _require_real_sam31_assets() -> tuple[Path, Path, Path]:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real SAM 3.1 tool test.")

    checkpoint_path = Path(
        os.getenv("SAM3_CHECKPOINT_PATH", str(DEFAULT_SAM31_CHECKPOINT))
    )
    image_path = Path(os.getenv("SAM31_TEST_IMAGE", str(DEFAULT_EXAMPLE_IMAGE)))
    repo_root = Path(os.getenv("SAM3_REPO_ROOT", str(DEFAULT_SAM3_REPO_ROOT)))

    if not checkpoint_path.is_file():
        pytest.skip(f"SAM3 checkpoint not found: {checkpoint_path}")
    if not image_path.is_file():
        pytest.skip(f"SAM3 test image not found: {image_path}")
    if not (repo_root / "sam3" / "__init__.py").is_file():
        pytest.skip(f"SAM3 repo is invalid: {repo_root}")

    return checkpoint_path, image_path, repo_root


def _configure_real_sam31_backend(checkpoint_path: Path, repo_root: Path) -> None:
    os.environ["SAM3_REPO_ROOT"] = str(repo_root)
    os.environ["SAM3_CHECKPOINT_PATH"] = str(checkpoint_path)
    settings.sam3_checkpoint_path = str(checkpoint_path)
    settings.sam3_model_version = "sam3.1"
    settings.sam3_load_from_hf = False
    settings.sam3_device = "cuda"
    settings.sam3_compile = False
    sam3_point_backend._load_image_model.cache_clear()
    sam3_point_backend._load_text_processor.cache_clear()
    sam3_point_backend._load_runtime_modules.cache_clear()


def _build_segment_state(image_path: Path):
    return {
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


def test_segment_tool_runs_with_real_sam31_text_prompt() -> None:
    if os.getenv("RUN_REAL_VISION_TESTS") != "1":
        pytest.skip("set RUN_REAL_VISION_TESTS=1 to run real vision backend integration tests")

    checkpoint_path, image_path, repo_root = _require_real_sam31_assets()
    _configure_real_sam31_backend(checkpoint_path, repo_root)

    tool = SegmentTool()
    execution = ToolRunner(ToolRegistry({tool.name: tool})).run(
        _build_segment_state(image_path),
        tool.name,
        task_id="task_sam31",
        loop_index=1,
        args=SegmentArgs(
            image_ref="art_img_001",
            prompt="woman in white blouse",
            backend_name="sam31",
        ),
    )

    assert execution.invocation.status == "succeeded"
    assert execution.invocation.args == {
        "image_ref": "art_img_001",
        "prompt": "woman in white blouse",
        "backend_name": "sam31",
    }

    artifact = execution.artifacts[0]
    assert artifact.payload["image_ref"] == "art_img_001"
    assert artifact.payload["prompt"] == "woman in white blouse"
    assert artifact.payload["source_stage"] == "sam_text_only"
    assert float(artifact.payload["mask_score"]) >= 0.0
    assert Path(artifact.uri).is_file()

    with Image.open(artifact.uri) as mask_image:
        assert mask_image.mode == "L"
        bbox = mask_image.getbbox()
        assert bbox is not None
        left, top, right, bottom = bbox
        assert right > left
        assert bottom > top
