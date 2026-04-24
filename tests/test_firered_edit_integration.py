from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from schema import (
    ImageArtifact,
    InstructionArtifact,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
)
from schema.tools import EditArgs
from tools.edit_tool import EditTool
from vision_backends.firered_edit_backend import (
    DEFAULT_LORA_PATH,
    DEFAULT_LORA_WEIGHT_NAME,
    DEFAULT_MODEL_PATH,
    load_pipeline,
)


DEFAULT_EXAMPLE_IMAGE = Path("examples/fig1.jpg")
OUTPUT_HEIGHT = 1024
OUTPUT_WIDTH = 464
EDIT_INSTRUCTION = (
    "Edit the provided image into a realistic full-body photo of the same person. "
    "Preserve the identity, face, clothing style, and photographic realism. "
    "Extend the framing so the entire body is visible from head to feet. "
    "Do not add extra people."
)


def _require_real_firered_assets() -> Path:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real FireRed edit tool test.")

    model_path = Path(os.getenv("FIRERED_MODEL_PATH", DEFAULT_MODEL_PATH))
    lora_path = Path(os.getenv("FIRERED_LORA_PATH", DEFAULT_LORA_PATH))
    lora_weight_name = os.getenv("FIRERED_LORA_WEIGHT_NAME", DEFAULT_LORA_WEIGHT_NAME)
    image_path = Path(os.getenv("FIRERED_EDIT_TEST_IMAGE", str(DEFAULT_EXAMPLE_IMAGE)))

    if not model_path.is_dir():
        pytest.skip(f"FireRed model path not found: {model_path}")
    if not (model_path / "model_index.json").is_file():
        pytest.skip(f"FireRed model_index.json not found under: {model_path}")
    if not lora_path.is_dir():
        pytest.skip(f"FireRed LoRA path not found: {lora_path}")
    if not (lora_path / lora_weight_name).is_file():
        pytest.skip(f"FireRed LoRA weight not found: {lora_path / lora_weight_name}")
    if not image_path.is_file():
        pytest.skip(f"FireRed edit test image not found: {image_path}")
    return image_path


def _build_edit_state(image_path: Path):
    return {
        "tasks": {
            "task_firered_edit": Task(
                id="task_firered_edit",
                plan_id="plan_firered_edit",
                type="global_edit",
                instruction=EDIT_INSTRUCTION,
                acceptance_criteria=["full body is visible", "identity is preserved"],
            )
        },
        "session": SessionState(
            session_id="sess_firered_edit",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_firered_edit",
            current_task_id="task_firered_edit",
            task_states={
                "task_firered_edit": TaskState(
                    task_id="task_firered_edit",
                    status=TaskStatus.RUNNING,
                    task_artifact_ids=["art_inst_firered_edit"],
                )
            },
        ),
        "artifacts": {
            "art_inst_firered_edit": InstructionArtifact(
                id="art_inst_firered_edit",
                payload={"instruction_text": EDIT_INSTRUCTION},
            ),
            "art_img_firered_input": ImageArtifact(
                id="art_img_firered_input",
                uri=str(image_path),
                payload={"role": "input"},
                created_by="user",
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
    }


def test_edit_tool_runs_with_real_firered_backend(monkeypatch) -> None:
    if os.getenv("RUN_REAL_VISION_TESTS") != "1":
        pytest.skip("set RUN_REAL_VISION_TESTS=1 to run real vision backend integration tests")

    image_path = _require_real_firered_assets()
    monkeypatch.setenv("FIRERED_HEIGHT", str(OUTPUT_HEIGHT))
    monkeypatch.setenv("FIRERED_WIDTH", str(OUTPUT_WIDTH))
    monkeypatch.setenv("FIRERED_NUM_INFERENCE_STEPS", "8")
    monkeypatch.setenv("FIRERED_FUSE_LORA", "false")
    load_pipeline.cache_clear()

    execution = EditTool().run(
        _build_edit_state(image_path),
        task_id="task_firered_edit",
        loop_index=1,
        args=EditArgs(
            instruction=EDIT_INSTRUCTION,
            image_refs=["art_img_firered_input"],
        ),
    )

    assert execution.invocation.status == "succeeded"
    assert execution.invocation.args == {
        "instruction": EDIT_INSTRUCTION,
        "image_refs": ["art_img_firered_input"],
    }

    artifact = execution.artifacts[0]
    assert artifact.payload["role"] == "candidate_image"
    assert artifact.payload["backend_name"] == "firered"
    assert artifact.payload["primary_image_ref"] == "art_img_firered_input"
    assert artifact.payload["auxiliary_image_refs"] == []
    assert artifact.payload["source"] == "edit_output"
    assert artifact.source_ids == ["art_img_firered_input"]
    assert Path(artifact.uri).is_file()

    snapshot = artifact.payload["backend_config_snapshot"]
    assert snapshot["height"] == OUTPUT_HEIGHT
    assert snapshot["width"] == OUTPUT_WIDTH
    assert snapshot["num_inference_steps"] == 8
    assert snapshot["fuse_lora"] is False

    with Image.open(artifact.uri) as output_image:
        assert output_image.mode == "RGB"
        assert output_image.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
