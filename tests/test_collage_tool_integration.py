from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from schema import (
    ArtifactIndex,
    ArtifactKind,
    CollageArgs,
    ImageArtifact,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
    ToolName,
)
from tools.collage_tool import CollageTool


def test_collage_tool_real_llm_returns_layout_and_renders_image() -> None:
    """Real integration test: call the configured multimodal LLM for collage layout."""

    if os.getenv("RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("set RUN_REAL_LLM_TESTS=1 to run real LLM integration tests")

    image_a = Path("examples/fig1.jpg")
    image_b = Path("examples/fig2.jpg")
    assert image_a.is_file()
    assert image_b.is_file()

    state = {
        "input": {
            "instruction_text": "把多张参考图整理成一张清晰的编辑参考板。",
            "use_llm": True,
        },
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="创建一张包含人物身份和服装参考的 collage reference。",
            )
        },
        "session": SessionState(
            session_id="sess_collage_real_llm",
            phase=SessionPhase.EXECUTING,
            current_plan_id="plan_001",
            current_task_id="task_001",
            task_states={
                "task_001": TaskState(
                    task_id="task_001",
                    status=TaskStatus.RUNNING,
                )
            },
            artifact_index=ArtifactIndex(by_type={ArtifactKind.IMAGE: ["art_face_001", "art_cloth_001"]}),
        ),
        "artifacts": {
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_a),
                summary="人物身份、脸部和整体外观参考图。",
                payload={"role": "identity_reference"},
                scope="session",
            ),
            "art_cloth_001": ImageArtifact(
                id="art_cloth_001",
                uri=str(image_b),
                summary="服装、纹理、颜色和穿搭风格参考图。",
                payload={"role": "clothing_reference"},
                scope="session",
            ),
        },
        "operations": [],
        "task_act_records": [],
    }

    result = CollageTool().run(
        state,
        task_id="task_001",
        loop_index=1,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="左侧突出人物身份参考，右侧突出服装参考，形成清晰的编辑参考板。",
        ),
    )

    artifact = result.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert artifact.created_by == ToolName.COLLAGE.value
    assert artifact.payload["role"] == "collage_reference"
    assert artifact.payload["block_artifact_ids"] == ["art_face_001", "art_cloth_001"]
    assert len(artifact.payload["layers"]) == 2
    assert artifact.source_ids == ["art_face_001", "art_cloth_001"]
    assert artifact.summary is None

    output_path = Path(artifact.uri)
    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.mode == "RGBA"
        assert image.width == artifact.payload["canvas"]["width"]
        assert image.height == artifact.payload["canvas"]["height"]
        assert image.width > 0
        assert image.height > 0

    print(f"collage_output={output_path}")
    print(f"layout={artifact.payload}")
