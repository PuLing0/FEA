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
loaded:

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
- `REAL_AGENT_SMOKE_MAX_EXECUTE_ACTS` limits execute-loop tool calls; the default is `1` for a fast real edit smoke.
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
