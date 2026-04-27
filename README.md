# FEA: Figure Edit Agent

FEA is a LangGraph-based image editing agent. It plans a figure-editing task, selects visual tools, runs image edit/segmentation backends, evaluates intermediate candidates, and writes structured artifacts and logs for inspection.

The project defaults to **remote backend mode**: heavy vision models run as separate HTTP services, while the agent only sends requests and receives output paths, similar to how the LLM layer calls a remote model provider. Local in-process model loading remains available for development by setting `EDIT_BACKEND=local` or `SEGMENT_BACKEND=local`.

## Project Layout

- `src/agent.py` — public agent entrypoint and CLI runner.
- `src/runtime/` — LangGraph runtime, scheduling, prompts, logging, and input selection.
- `src/agents/` — planner, executor, and evaluator agents.
- `src/tools/` — tool adapters for edit, segment, crop, collage, grounding, understanding, evaluation, and prompt reconstruction.
- `src/vision_backends/` — FireRed, SAM3.1, and remote backend service/client code.
- `src/llm/` — OpenAI-compatible LLM client layer.
- `tests/` — runtime, schema, and backend regression tests.
- `dev_docs/real_runtime_configuration.md` — detailed real-model runtime notes.
- `examples/` — sample images for local experiments.

## Setup

Use Python 3.11+ and `uv`.

```bash
cd /mnt/sda/sijuzheng/project/FEA
uv sync
```

Or use the existing virtual environment when present:

```bash
cd /mnt/sda/sijuzheng/project/FEA
./.venv/bin/pytest tests/test_schema.py -q
```

Expected lightweight test result:

```text
99 passed
```

## LLM Configuration

Real agent runs load `.env` via `python-dotenv`. Configure an OpenAI-compatible endpoint:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_NAME=...
LLM_TEMPERATURE=0.0
```

Do not commit `.env` or secrets.

## Running the Agent with Local Backends

Local mode keeps vision model loading inside the agent process. This is useful for development and fallback testing. Because remote mode is the default, set local backend flags explicitly when using this path.

```bash
cd /mnt/sda/sijuzheng/project/FEA

EDIT_BACKEND=local \
SEGMENT_BACKEND=local \
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
AGENT_LOG_ENABLED=true \
AGENT_LOG_CONSOLE=true \
AGENT_LOG_DIR=generated/agent_logs \
./.venv/bin/python src/agent.py \
  --images examples/fig1.jpg examples/fig2.jpg examples/fig3.jpg examples/fig4.jpg \
  --instruction "Generate a photo of this person wearing the provided top and skirt in the provided background."
```

For a quick smoke run that stops after the first edit candidate:

```bash
cd /mnt/sda/sijuzheng/project/FEA

EDIT_BACKEND=local \
SEGMENT_BACKEND=local \
FIRERED_CUDA_VISIBLE_DEVICES=5,6,3,4 \
./.venv/bin/python src/agent.py \
  --stop-after-first-edit \
  --images examples/fig1.jpg examples/fig2.jpg examples/fig3.jpg examples/fig4.jpg \
  --instruction "Generate a photo of this person wearing the provided top and skirt in the provided background."
```

## Remote Vision Backend Mode

Remote backend mode isolates heavy models from the agent process. The agent sends JSON requests containing shared filesystem paths; each service writes outputs to disk and returns the output path.

This is useful because:

- FireRed and SAM can be deployed independently.
- The agent process no longer needs to load all vision models.
- Model services can stay warm across multiple agent runs.
- Crashes or restarts in one backend do not necessarily kill the agent process or other backends.

Remote mode assumes the agent and backend service share the same filesystem paths, for example the same machine or a shared mount.

### Start Both Vision Services

Open terminal 1. This starts both FireRed edit and SAM3.1 segment with one command:

```bash
cd /mnt/sda/sijuzheng/project/FEA

./scripts/start_remote_vision_backends.sh
```

The script:

- loads `.env` automatically
- starts FireRed on `127.0.0.1:8765`
- starts SAM3.1 on `127.0.0.1:8766`
- preloads both models on startup and occupies VRAM immediately
- writes logs under `generated/service_logs/`
- stops both services when you press `Ctrl+C`

If you prefer to start them separately, use the commands below.

### Start FireRed Edit Service

Open terminal 1:

```bash
cd /mnt/sda/sijuzheng/project/FEA

./scripts/start_firered_edit_server.sh
```

Service endpoint:

```text
POST /v1/edit/firered
GET  /health
```

Check health:

```bash
curl http://127.0.0.1:8765/health
```

The FireRed service now preloads the pipeline during startup. The first boot may take time, and
`firered_cached: true` means the model is already resident in GPU memory.

### Start SAM3.1 Segment Service

Open terminal 2 if segmentation/local editing is needed:

```bash
cd /mnt/sda/sijuzheng/project/FEA

SAM3_CHECKPOINT_PATH=/path/to/sam3.1/checkpoint.pt \
./scripts/start_sam31_segment_server.sh
```

Service endpoint:

```text
POST /v1/segment/sam31
GET  /health
```

Check health:

```bash
curl http://127.0.0.1:8766/health
```

The SAM3.1 service now preloads the text-prompt runtime during startup. `sam3_ready: true` means
the model and processor are already loaded.

### Run Agent Against Remote Backends

Open terminal 3. With the default ports, no backend mode or backend URL variables are required:

```bash
cd /mnt/sda/sijuzheng/project/FEA

VISION_BACKEND_TIMEOUT_SECONDS=900 \
AGENT_LOG_ENABLED=true \
AGENT_LOG_CONSOLE=true \
AGENT_LOG_DIR=generated/agent_logs \
./.venv/bin/python src/agent.py \
  --images examples/fig1.jpg examples/fig2.jpg examples/fig3.jpg examples/fig4.jpg \
  --instruction "Generate a photo of this person wearing the provided top and skirt in the provided background."
```

If only FireRed edit is needed, start only the FireRed service and run:

```bash
cd /mnt/sda/sijuzheng/project/FEA

VISION_BACKEND_TIMEOUT_SECONDS=900 \
./.venv/bin/python src/agent.py \
  --stop-after-first-edit \
  --images examples/fig1.jpg examples/fig2.jpg examples/fig3.jpg examples/fig4.jpg \
  --instruction "Generate a photo of this person wearing the provided top and skirt in the provided background."
```

## Backend Environment Variables

### Shared Remote Backend

- `EDIT_BACKEND=local|remote` — default is `remote`.
- `SEGMENT_BACKEND=local|remote` — default is `remote`.
- `FIRERED_EDIT_BACKEND_BASE_URL` — FireRed edit service URL, default `http://127.0.0.1:8765`.
- `SAM31_SEGMENT_BACKEND_BASE_URL` — SAM3.1 segment service URL, default `http://127.0.0.1:8766`.
- `VISION_BACKEND_BASE_URL` — fallback shared service URL.
- `VISION_BACKEND_TIMEOUT_SECONDS` — HTTP request timeout, default `600`.

### FireRed

- `FIRERED_CUDA_VISIBLE_DEVICES` — GPU mapping, set before importing torch.
- `FIRERED_DISABLE_LORA=true|false` — disable or enable LoRA loading.
- `FIRERED_FUSE_LORA=true|false` — fuse LoRA weights in the FireRed service; the startup script defaults to `true`.
- `FIRERED_PRELOAD_ON_START=true|false` — preload FireRed during service startup; default `true`.
- `FIRERED_HEIGHT`, `FIRERED_WIDTH` — optional output size overrides; unset by default.
- `FIRERED_NUM_INFERENCE_STEPS` — diffusion steps.
- `FIRERED_ENABLE_ATTENTION_SLICING` — default enabled.
- `FIRERED_ENABLE_TORCH_COMPILE` — default enabled.
- `FIRERED_ENABLE_WARMUP` — default enabled.

### SAM3.1

- `SAM3_CHECKPOINT_PATH` — local SAM3.1 checkpoint path.
- `SAM3_DEVICE` — `cpu` or `cuda`.
- `SAM3_CUDA_VISIBLE_DEVICES` — optional GPU visibility restriction.
- `SAM31_PRELOAD_ON_START=true|false` — preload the SAM3.1 text-prompt runtime during service startup; default `true`.

## Logs and Outputs

Generated outputs are written under `generated/`, including:

- `generated/edit/` — edited image candidates.
- `generated/segment/` — mask outputs.
- `generated/agent_logs/` — JSONL agent run logs.

The CLI prints a JSON summary containing `run_id`, `run_log_uri`, `final_artifact`, `operations`, and evaluator decision information.

## Development Commands

```bash
cd /mnt/sda/sijuzheng/project/FEA
./.venv/bin/pytest tests/test_schema.py -q
./.venv/bin/python -m py_compile src/agent.py
./.venv/bin/python -m py_compile src/vision_backends/firered_edit_server.py src/vision_backends/sam31_segment_server.py
```

For LangGraph development:

```bash
uv run langgraph dev
```

## Notes

- Remote backend mode currently uses shared file paths, not multipart uploads or object storage.
- Use separate terminals/process managers for FireRed and SAM if both are needed.
- Stop services with `Ctrl+C`; FireRed service unloads its cached pipeline on shutdown.
- For more detailed real-runtime notes, see `dev_docs/real_runtime_configuration.md`.
