from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from schema import (
    ArtifactIndex,
    ArtifactKind,
    CollageArgs,
    GeometryArtifact,
    ImageArtifact,
    SessionPhase,
    SessionState,
    Task,
    TaskState,
    TaskStatus,
    ToolName,
)
from runtime.tool_runner import ToolRunner
from tools.collage_tool import CollageLayoutItem, CollageLayoutResult, CollageTool
from tools.registry import ToolRegistry


def _make_state(*, artifacts: dict[str, object]) -> dict:
    return {
        "tasks": {
            "task_001": Task(
                id="task_001",
                plan_id="plan_001",
                type="reference_edit",
                instruction="整理多张参考图，生成一张清晰参考板。",
            )
        },
        "session": SessionState(
            session_id="sess_collage_tool",
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
        "artifacts": artifacts,
        "operations": [],
        "task_act_records": [],
    }


def _write_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    Image.new("RGBA", size, color=color).save(path)


def _run_collage_tool(state: dict, *, args: CollageArgs, loop_index: int = 1):
    tool = CollageTool()
    return ToolRunner(ToolRegistry({tool.name: tool})).run(
        state,
        ToolName.COLLAGE,
        task_id="task_001",
        loop_index=loop_index,
        args=args,
    )


def _assert_failed(result, *, error_type: str, message: str) -> None:
    assert result.invocation.status == "failed"
    assert result.invocation.error is not None
    assert result.invocation.error["type"] == error_type
    assert message in result.invocation.error["message"]


def _layout_for_two_images() -> CollageLayoutResult:
    return CollageLayoutResult(
        canvas_width=96,
        canvas_height=48,
        background="transparent",
        items=[
            CollageLayoutItem(
                artifact_id="art_face_001",
                x=0,
                y=0,
                width=32,
                height=32,
                z_index=0,
            ),
            CollageLayoutItem(
                artifact_id="art_cloth_001",
                x=48,
                y=8,
                width=32,
                height=32,
                z_index=1,
            ),
        ],
    )


def test_collage_tool_renders_local_image_and_records_layout(mocker, tmp_path) -> None:
    face_path = tmp_path / "face.png"
    cloth_path = tmp_path / "cloth.png"
    _write_image(face_path, size=(24, 36), color=(255, 0, 0, 255))
    _write_image(cloth_path, size=(36, 24), color=(0, 255, 0, 255))
    llm_call = mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=_layout_for_two_images(),
    )
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(face_path),
                summary="face identity reference",
                payload={"role": "identity_reference"},
            ),
            "art_cloth_001": ImageArtifact(
                id="art_cloth_001",
                uri=str(cloth_path),
                payload={"role": "clothing_reference", "description": "green shirt"},
            ),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="identity on the left, clothing on the right",
        ),
    )

    artifact = result.artifacts[0]
    assert artifact.kind == ArtifactKind.IMAGE
    assert artifact.created_by == ToolName.COLLAGE.value
    assert artifact.summary is None
    assert artifact.source_ids == ["art_face_001", "art_cloth_001"]
    assert artifact.payload["role"] == "collage_reference"
    assert artifact.payload["layout_goal"] == "identity on the left, clothing on the right"
    assert artifact.payload["block_artifact_ids"] == ["art_face_001", "art_cloth_001"]
    assert artifact.payload["canvas"] == {
        "width": 96,
        "height": 48,
        "background": "transparent",
    }
    assert len(artifact.payload["layers"]) == 2
    assert "previous_collage_ref" not in artifact.payload
    assert result.invocation.tool_name == ToolName.COLLAGE
    assert result.invocation.output_refs == [artifact.id]
    assert result.invocation.result_payload["collage_artifact_id"] == artifact.id
    assert Path(artifact.uri).is_file()
    with Image.open(artifact.uri) as output_image:
        assert output_image.mode == "RGBA"
        assert output_image.size == (96, 48)
    llm_call.assert_called_once()
    assert "face identity reference" in llm_call.call_args.kwargs["user_prompt"]
    assert "green shirt" in llm_call.call_args.kwargs["user_prompt"]
    assert llm_call.call_args.kwargs["image_paths"] == [str(face_path), str(cloth_path)]


def test_collage_tool_rejects_unknown_artifact(tmp_path) -> None:
    image_path = tmp_path / "face.png"
    _write_image(image_path, size=(16, 16), color=(255, 0, 0, 255))
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_path),
                payload={"role": "input"},
            )
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_missing_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="Unknown image artifact ref")


def test_collage_tool_rejects_non_image_artifact(tmp_path) -> None:
    image_path = tmp_path / "face.png"
    _write_image(image_path, size=(16, 16), color=(255, 0, 0, 255))
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
            "art_geo_001": GeometryArtifact(
                id="art_geo_001",
                payload={"candidates": []},
            ),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_geo_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="Artifact is not an image")


def test_collage_tool_rejects_non_local_image_uri(tmp_path) -> None:
    image_path = tmp_path / "face.png"
    _write_image(image_path, size=(16, 16), color=(255, 0, 0, 255))
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
            "art_cloth_001": ImageArtifact(
                id="art_cloth_001",
                uri="store://images/cloth.png",
                payload={"role": "input"},
            ),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="not a local file path")


def test_collage_tool_rejects_missing_image_file(tmp_path) -> None:
    image_path = tmp_path / "face.png"
    _write_image(image_path, size=(16, 16), color=(255, 0, 0, 255))
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(
                id="art_face_001",
                uri=str(image_path),
                payload={"role": "input"},
            ),
            "art_cloth_001": ImageArtifact(
                id="art_cloth_001",
                uri=str(tmp_path / "missing.png"),
                payload={"role": "input"},
            ),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="FileNotFoundError", message="Image path does not exist")


def test_collage_tool_rejects_layout_that_omits_input(mocker, tmp_path) -> None:
    face_path = tmp_path / "face.png"
    cloth_path = tmp_path / "cloth.png"
    _write_image(face_path, size=(16, 16), color=(255, 0, 0, 255))
    _write_image(cloth_path, size=(16, 16), color=(0, 255, 0, 255))
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=32,
            canvas_height=16,
            items=[
                CollageLayoutItem(
                    artifact_id="art_face_001",
                    x=0,
                    y=0,
                    width=16,
                    height=16,
                )
            ],
        ),
    )
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(id="art_face_001", uri=str(face_path)),
            "art_cloth_001": ImageArtifact(id="art_cloth_001", uri=str(cloth_path)),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="omitted required artifact ids")


def test_collage_tool_rejects_layout_with_unknown_artifact(mocker, tmp_path) -> None:
    face_path = tmp_path / "face.png"
    cloth_path = tmp_path / "cloth.png"
    _write_image(face_path, size=(16, 16), color=(255, 0, 0, 255))
    _write_image(cloth_path, size=(16, 16), color=(0, 255, 0, 255))
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=48,
            canvas_height=16,
            items=[
                CollageLayoutItem(
                    artifact_id="art_face_001",
                    x=0,
                    y=0,
                    width=16,
                    height=16,
                ),
                CollageLayoutItem(
                    artifact_id="art_unknown_001",
                    x=24,
                    y=0,
                    width=16,
                    height=16,
                ),
            ],
        ),
    )
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(id="art_face_001", uri=str(face_path)),
            "art_cloth_001": ImageArtifact(id="art_cloth_001", uri=str(cloth_path)),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="referenced unknown artifact ids")


def test_collage_tool_auto_expands_canvas_for_layout_outside_canvas(mocker, tmp_path) -> None:
    face_path = tmp_path / "face.png"
    cloth_path = tmp_path / "cloth.png"
    _write_image(face_path, size=(16, 16), color=(255, 0, 0, 255))
    _write_image(cloth_path, size=(16, 16), color=(0, 255, 0, 255))
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=32,
            canvas_height=16,
            items=[
                CollageLayoutItem(
                    artifact_id="art_face_001",
                    x=0,
                    y=0,
                    width=16,
                    height=16,
                ),
                CollageLayoutItem(
                    artifact_id="art_cloth_001",
                    x=24,
                    y=0,
                    width=16,
                    height=16,
                ),
            ],
        ),
    )
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(id="art_face_001", uri=str(face_path)),
            "art_cloth_001": ImageArtifact(id="art_cloth_001", uri=str(cloth_path)),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    artifact = result.artifacts[0]
    assert artifact.payload["canvas"] == {
        "width": 40,
        "height": 16,
        "background": "transparent",
    }
    assert artifact.payload["canvas_auto_expanded"] is True
    assert artifact.payload["planned_canvas"] == {"width": 32, "height": 16}
    with Image.open(artifact.uri) as output_image:
        assert output_image.size == (40, 16)


def test_collage_tool_rejects_layout_that_exceeds_max_canvas_after_auto_expand(mocker, tmp_path) -> None:
    face_path = tmp_path / "face.png"
    cloth_path = tmp_path / "cloth.png"
    _write_image(face_path, size=(16, 16), color=(255, 0, 0, 255))
    _write_image(cloth_path, size=(16, 16), color=(0, 255, 0, 255))
    mocker.patch(
        "tools.collage_tool.invoke_structured_multimodal_llm",
        return_value=CollageLayoutResult(
            canvas_width=4096,
            canvas_height=32,
            items=[
                CollageLayoutItem(
                    artifact_id="art_face_001",
                    x=0,
                    y=0,
                    width=16,
                    height=16,
                ),
                CollageLayoutItem(
                    artifact_id="art_cloth_001",
                    x=4090,
                    y=0,
                    width=16,
                    height=16,
                ),
            ],
        ),
    )
    state = _make_state(
        artifacts={
            "art_face_001": ImageArtifact(id="art_face_001", uri=str(face_path)),
            "art_cloth_001": ImageArtifact(id="art_cloth_001", uri=str(cloth_path)),
        }
    )

    result = _run_collage_tool(
        state,
        args=CollageArgs(
            block_artifact_ids=["art_face_001", "art_cloth_001"],
            layout_goal="make a reference board",
        ),
    )

    _assert_failed(result, error_type="ValueError", message="requires a larger canvas than allowed")
