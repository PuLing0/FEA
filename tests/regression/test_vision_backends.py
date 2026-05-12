from __future__ import annotations

from tests.regression.common import *
from tests.regression.common import (
    _assert_tool_failed,
    _make_instruction_resolution_state,
    _run_tool,
)


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
