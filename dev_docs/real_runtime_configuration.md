# Real Runtime Configuration

This project can run lightweight unit tests without external services, but real
vision execution needs explicit environment configuration.

## FireRed CUDA edit backend

Use `FIRERED_CUDA_VISIBLE_DEVICES` before Python imports `torch` or `diffusers`.
For the validated 4-GPU layout:

```bash
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4
```

With that order, physical GPUs are remapped inside the process as:

- logical `cuda:0` -> physical GPU `5`
- logical `cuda:1` -> physical GPU `6`
- logical `cuda:2` -> physical GPU `3`
- logical `cuda:3` -> physical GPU `4`

The manual FireRed layout places the transformer halves on logical `cuda:0` and
`cuda:1`, the text encoder on logical `cuda:2` and `cuda:3`, and the VAE on the
last logical GPU. Keep the first two visible GPUs as the cards with the most
available memory.

`FIRERED_DISABLE_LORA` controls whether the configured/default FireRed LoRA is
loaded. The FireRed service startup script defaults `FIRERED_FUSE_LORA=true` and leaves
`FIRERED_HEIGHT`/`FIRERED_WIDTH` unset unless you explicitly override them:

- `FIRERED_DISABLE_LORA=true` runs the base model only.
- `FIRERED_DISABLE_LORA=false` or unset loads `FIRERED_LORA_PATH` and
  `FIRERED_LORA_WEIGHT_NAME` when configured.

Real FireRed pytest coverage is opt-in:

```bash
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
RUN_REAL_VISION_TESTS=1 \
./.venv/bin/pytest tests/test_firered_edit_integration.py -q
```

The profiling helper records per-stage CUDA memory and optional inference output:

```bash
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
./.venv/bin/python tests/scripts/profile_firered_memory.py \
  --image examples/fig1.jpg \
  --height 768 \
  --width 352 \
  --steps 8 \
  --output generated/profile_firered_memory.json
```

## Remote vision backend mode

Heavy vision models can run in a separate Python process. The agent then sends
JSON requests with shared local file paths and receives output file paths, like
the LLM layer sends requests to a remote model provider.

Start each model service independently on the GPU host:

```bash
./scripts/start_firered_edit_server.sh
```

```bash
SAM3_CHECKPOINT_PATH=/path/to/sam3.1/checkpoint.pt \
./scripts/start_sam31_segment_server.sh
```

Both service scripts now preload their models on startup by default, so boot can
take time and GPU memory should be occupied immediately after the process is
ready.

Run the agent against the default remote services. With default ports, backend mode and backend URL variables are optional:

```bash
VISION_BACKEND_TIMEOUT_SECONDS=900 \
./.venv/bin/python src/agent.py \
  --images examples/fig1.jpg examples/fig2.jpg examples/fig3.jpg examples/fig4.jpg \
  --instruction "Generate a photo of this person wearing the provided top and skirt in the provided background."
```

Remote mode is the runtime default; local mode is still available for development and tests:

- `EDIT_BACKEND=remote` is the default; set `EDIT_BACKEND=local` to run FireRed in-process.
- `SEGMENT_BACKEND=remote` is the default; set `SEGMENT_BACKEND=local` to run SAM3.1 in-process.
- `FIRERED_EDIT_BACKEND_BASE_URL` points edit requests to the FireRed service and defaults to `http://127.0.0.1:8765`.
- `SAM31_SEGMENT_BACKEND_BASE_URL` points segment requests to the SAM service and defaults to `http://127.0.0.1:8766`.
- `VISION_BACKEND_BASE_URL` remains a shared fallback when both services use the same host/port.
- `FIRERED_PRELOAD_ON_START=true|false` controls FireRed eager preload and defaults to `true`.
- `SAM31_PRELOAD_ON_START=true|false` controls SAM3.1 eager preload and defaults to `true`.
- `GET /health` checks service availability plus loaded-state flags such as `firered_cached` and `sam3_ready`.

Remote mode assumes the agent and service share the same filesystem paths for
input and generated images. Use multipart upload or object storage only if the
service runs on a separate machine without a shared mount.

## LLM configuration

Real agent runs load `.env` through `python-dotenv`. Configure:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_NAME=...
LLM_TEMPERATURE=0.0
```

Do not commit `.env` or secrets. If these values are missing, the runtime falls
back to rule-based behavior for normal graph runs; the real smoke script fails
fast because it is intended to verify the real LLM path.

## Agent run outputs

Agent graph runs create one run-scoped output directory and mirror concise progress to the console by default. User-facing runs default to `generated/runs/`, pytest-generated outputs default to `generated/tests/pytest/`, and backend service fallback files default to `generated/services/`. Each run directory contains `logs/run.jsonl`, `logs/messages.jsonl`, generated artifacts under `artifacts/<tool>/`, and per-tool JSON snapshots under `tool_results/<task_id>/`.

Useful switches:

- `AGENT_LOG_ENABLED=false` disables JSONL file output.
- `AGENT_LOG_CONSOLE=false` disables console mirroring.
- `AGENT_OUTPUT_DIR=generated/runs` changes the root for user-facing runtime runs.
- `AGENT_TEST_OUTPUT_DIR=generated/tests` changes where pytest-created run-scoped outputs are written.
- `AGENT_SERVICE_OUTPUT_DIR=generated/services` changes where backend service fallback outputs are written.
- `AGENT_LOG_LEVEL=debug` is reserved for verbose/debug filtering.

The smoke summary prints `run_id`, `output_dir`, and `run_log_uri`. Each JSONL record includes the session phase, current plan/task, event name, and structured payload for operations, artifacts, checkpoints, and decisions. Secrets such as `LLM_API_KEY` are not logged.

## Real end-to-end smoke

Run the full graph with real LLM planning/evaluation and the real FireRed edit
backend:

```bash
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
RUN_REAL_AGENT_SMOKE=1 \
./.venv/bin/python tests/scripts/run_real_agent_smoke.py
```

Optional overrides:

- `REAL_AGENT_SMOKE_IMAGES` is a comma-separated image list; by default the script uses `examples/fig1.jpg` through `examples/fig4.jpg`.
- `REAL_AGENT_SMOKE_IMAGE` points to a single input image when `REAL_AGENT_SMOKE_IMAGES` is not set.
- `REAL_AGENT_SMOKE_INSTRUCTION` overrides the edit instruction.
- `AGENT_MAX_EXECUTE_ACTS` sets the default execute-loop tool-call budget for the runtime.
- `REAL_AGENT_SMOKE_MAX_EXECUTE_ACTS` overrides that budget for the smoke script only; if unset, the smoke script falls back to `AGENT_MAX_EXECUTE_ACTS`.
- `REAL_AGENT_SMOKE_STOP_AFTER_FIRST_EDIT` defaults to `true`; the smoke exits after the first real FireRed candidate instead of waiting for all evaluator retries. Set it to `false` for a full terminal graph run.
- `REAL_AGENT_SMOKE_MAX_EVALUATOR_CHECKPOINTS` limits evaluator retries.

## SAM3.1 segment backend

Real SAM3.1 segmentation is also gated by `RUN_REAL_VISION_TESTS=1`:

```bash
RUN_REAL_VISION_TESTS=1 \
./.venv/bin/pytest tests/test_sam31_integration.py -q
```

Relevant environment variables:

- `SAM3_REPO_ROOT` points to the official SAM3 checkout.
- `SAM3_CHECKPOINT_PATH` points to the local SAM3.1 checkpoint.
- `SAM3_DEVICE` defaults to `cpu`; set it to `cuda` for GPU execution.
- `SAM3_CUDA_VISIBLE_DEVICES` can restrict visible GPUs for SAM3. Set it before
  any code imports `torch`.
