# Experimental MLX-LM reward training

**WARNING: generated Python executes with your user privileges. A subprocess is NOT a sandbox.**
Both built-in tasks refuse reward/evaluation execution unless explicitly enabled with
`--allow-unsafe-code` (Python API: `allow_unsafe_code=True`). Do not enable it on an
untrusted machine/account or where credentials, private files, or networks are accessible.
This project does not implement a sandbox. Custom task modules are trusted Python imports
and are responsible for declaring `executes_generated_code = True` and enforcing their own guards.

This is a **GRPO-style experimental objective**, not full GRPO. It normalizes rewards within
a sample group and optimizes summed sequence log probabilities with a reference log-probability
difference penalty. It has no old-policy importance ratios, clipped surrogate, or full token-wise
KL estimator. The reference difference is not guaranteed to be a nonnegative KL divergence.

Model support is limited to **MLX-LM compatible models** with supported LoRA attention projection
keys and differentiable operations; arbitrary Hugging Face models/architectures are not supported.
Quantized MoE VJP limitations remain model-dependent. The default model identifier is inherited
from the notebooks, not a compatibility guarantee. No model download or full training run was
performed for this migration.

## Model families

The CLI and shared Python helpers accept these presets, or any explicit MLX-LM-compatible
repository ID or local checkpoint directory:

| Preset | Checkpoint |
|---|---|
| `gemma` | `mlx-community/gemma-3-1b-it-4bit` |
| `qwen` | `mlx-community/Qwen3-4B-4bit` |
| `gpt-oss` (default) | `mlx-community/gpt-oss-20b-MXFP4-Q4` |

```sh
python3 -m rl_framework.train game_2048.RL2048 gemma --allow-unsafe-code
python3 -m rl_framework.train mbpp_unittests.MBPPUnitTests qwen --allow-unsafe-code
python3 -m rl_framework.train game_2048.RL2048 /path/to/mlx-checkpoint --allow-unsafe-code
```

```python
from rl_framework.core.models import setup_model, sample_group

model, tokenizer = setup_model("gemma", lora_rank=8, lora_layers=8)
completions, token_ids = sample_group(model, tokenizer, "Write a Python identity function.")
```

The checkpoint's tokenizer supplies its chat template and stop tokens. Pass raw prompts;
chat-template special tokens are encoded without adding another BOS, and likelihood computation
uses the same prompt IDs as generation. Completion IDs come directly from sampling, including
a sampled stop token, rather than being reconstructed from decoded text. Base checkpoints
without a chat template use plain tokenizer encoding. Qwen reasoning behavior follows the
checkpoint's template defaults.

LoRA defaults target attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`, leaving quantized MoE
experts frozen. Use `--lora-rank`, `--lora-layers` (`-1` for all), and `--lora-keys` (paths such
as `self_attn.q_proj`) to configure adapters. Models with fewer layers use all available layers;
missing targets fail before training. Family names are preset shortcuts, not an architecture
whitelist: supported variants depend on the installed MLX-LM version. Multimodal inputs are
outside this text-only task interface.

Offline regression checks use actual, locally initialized Gemma, Gemma 2, Gemma 3 text,
Qwen 2 (also used by Qwen 2.5), and Qwen 3 implementations, with both floating-point and 4-bit
weights and real LoRA/Adam updates. Small Gemma 3 and Qwen 3 checkpoints additionally exercise
local loading, tokenization, generation, and likelihood agreement. These checks do not download
or establish training quality for the full preset checkpoints.

## Install and run

### Browser workspace with Socket.IO

```sh
python3 -m pip install -e '.[web,test]'
cd frontend
npm ci
npm run build
cd ..
python3 -m rl_framework.web.server
```

Open **http://localhost:8765**. The Python server serves the built React/TypeScript frontend
and its Socket.IO connection from the same origin. No Node server is needed after building.
The `rl-web` console command is equivalent once installed. Change the port with `--port`;
the server binds to loopback only.

Choose a task, model preset or custom MLX checkpoint, and training settings. The workspace has:

- Live completion text streamed directly from the model, with sample selection and auto-scroll.
- Per-step mean reward and loss charts, generation speed, phase, progress, and elapsed time.
- Worker stdout/stderr, traceback and loading output, run settings, and evaluation results.
- Start/stop controls, reconnect snapshots, persistent run history, and JSONL event downloads.
- **Try demo stream**, a clearly labeled simulation that exercises the real worker and Socket.IO
  transport without a model download or generated-code execution. Demo metrics are simulated.

Training/evaluation retain the existing generated-code execution opt-in. The browser requires
the checkbox before those actions. The 2048 board demo and MBPP dataset preview use their
existing task actions; dataset preview can download MBPP data. Web runs support the two built-in
tasks; custom Python task imports remain available through the CLI.

Each run starts a separate Python process; the web event loop never executes MLX training.
One worker runs at a time to avoid competing for model memory. Closing the browser leaves the
worker running; reconnect to view its current snapshot or press Stop. Stop sends SIGTERM to the
worker process group and escalates to SIGKILL after four seconds, including reward children
that remain in that group. This is process lifecycle management, not a security sandbox.

Run data stays in `artifacts/web-runs/<run-id>/` (override with `--run-dir`): `events.jsonl`
contains the full event stream, `run.json` the latest saved snapshot, and `artifacts/` task
outputs. Browser snapshots retain the latest 12 completions, 300 log chunks, and 2,000 metrics;
the downloadable event file retains the complete run. Snapshots save on phase changes, metrics,
evaluation, and terminal status. After a server restart, previously active runs are marked
interrupted; this UI does not resume training or restore adapters. This is a trusted local
workspace, with same-origin and local-host checks, not a remote multi-user service.

For frontend development, keep the Python backend running and use `cd frontend && npm run dev`.
Vite proxies `/api` and `/socket.io` to port 8765. Rebuild for changes to appear in the Python-served
production UI. Frontend development/builds require Node 20.19+ or 22.12+; `npm test` uses Node 24's
built-in TypeScript support. Web tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_web.py`
after installing the web and test extras.

Socket.IO commands are `start_run` (settings object), `watch_run` (`{run_id}`), and `stop_run`
(`{run_id}`). Each acknowledges with `{ok, error?}`. The server sends `run_list` summaries,
`run_snapshot` for the selected run, and `run_event` updates with `run_id`, monotonically increasing
`seq`, UTC `timestamp`, and `type`. Event types include `phase`, `generation_start`, `token`,
`generation_end`, `metric`, `evaluation`, `log`, and `status`. Snapshots plus sequence checks
prevent duplicated text when clients reconnect. The CLI remains usable independently; event
instrumentation is inactive unless a sink is registered.

### Command line

On a supported Apple Silicon macOS installation with MLX:

```sh
python3 -m pip install -e '.[test]'
python3 -m rl_framework.train mbpp_unittests.MBPPUnitTests --action preview
python3 -m rl_framework.train game_2048.RL2048 --action demo --output-dir artifacts
# These commands load models and execute generated Python; run only after reviewing the warning:
python3 -m rl_framework.train mbpp_unittests.MBPPUnitTests --allow-unsafe-code
python3 -m rl_framework.train game_2048.RL2048 --allow-unsafe-code
python3 -m rl_framework.train game_2048.RL2048 --action evaluate --phase baseline --allow-unsafe-code
python3 -m rl_framework.train game_2048.RL2048 --action evaluate --phase final --allow-unsafe-code
```

Preview loads MBPP data but no model. Demo loads neither a model nor dataset and prints an ASCII
board plus a standalone `demo.html`. Training automatically evaluates the current model before
and after updates in the same process. Evaluation-only loads the specified initial model; it does
not restore a previous run's trained adapters (checkpoint persistence was not in the notebooks).
Use the Python task evaluation API with an in-memory trained model when needed.

| Task | Steps | Group | Learning rate | Completion tokens | Temperature |
|---|---:|---:|---:|---:|---:|
| MBPP | 200 | 6 | 1e-5 | 2048 | .9 |
| 2048 | 200 | 4 | 5e-6 | 2048 | .9 |

All training settings can be overridden: `--steps`, `--group`, `--lr`, `--max_tokens` (or
`--max-tokens`), `--temp`. An optional second positional argument overrides the model ID.
Custom tasks use `package.module:TaskClass` or a fully qualified dotted class path; no fixed
list of task choices is imposed. They must implement `make_prompt(tokenizer)` returning raw
text and `reward(completion)` returning `(float, info)`. Optional class `defaults` overrides
CLI defaults; optional `evaluate(model, tokenizer, phase=..., output_dir=...)` enables automatic
before/after inspection. `--action preview` and `demo` require corresponding task functionality.

## Restored behavior and intentional corrections

- MBPP: `train[:200]`, held-out `test[:50]`, counts and first 10 task ID/text/test-list rows.
  Evaluation generates one sample per held-out row at 512 tokens / temperature .2, runs the
  entire test list together, and reports `passed`, actual `evaluated`, and `pass_at_1`.
  The denominator is the number actually evaluated, not the requested maximum; empty evaluation
  reports zero evaluated and a 0.0 rate. Reward retains per-test fractional credit.
- Prompts are raw at the task boundary. `core.models` applies the chat template once for
  generation and once separately for likelihood computation; MBPP no longer double-templates.
- 2048: original merges, spawn mechanics, seed behavior, ASCII renderer, color HTML renderer,
  extraction, and reusable `run_strategy` are available in `tasks/game_2048.py`. Demo uses seed 42
  for both board and move RNG (the notebook's move RNG was unseeded).
- Baseline inspection generates 3 strategies at 2048 tokens / .7, reporting rewards, metadata,
  raw first 300 characters, and extracted first 300 characters. Final inspection generates 3
  at 512 / .3, reports extracted first 400 characters, and runs each extractable strategy on seed
  99 for up to 500 moves. Final ASCII boards are returned and `final-1.html` through
  `final-3.html` are written under `--output-dir` for successful compiled strategies.
- Rewards evaluate seeds 1, 2, 3 for up to 400 moves. Invalid moves and ordinary strategy
  exceptions stop that game and retain its partial score, matching the notebook. No function:
  -1; compile/name failure: -.5; base reward .5 plus capped score and tile bonuses, capped at 2.
- MBPP temporary scripts are cleaned on success, failure, and timeout. 2048 compilation and
  execution run in a fresh child interpreter with a parent-enforced 5-second default timeout
  (Python API `RL2048(timeout=...)`). An infinite loop cannot block the parent indefinitely;
  timeout/worker failure returns 0 and a reason. The deadline includes interpreter startup and
  all games, not just a single strategy call. Temporary request/result files are cleaned.
- Timeouts kill the immediate child, **not arbitrary descendants** created by malicious code.
  There are no filesystem, network, memory, output-size, or privilege restrictions. MBPP captures
  subprocess output in memory. Process creation and hostile cleanup behavior are not bounded
  security guarantees. `run_strategy` accepts trusted callables and has no wall-clock timeout;
  generated source must go through `RL2048.execute` rather than direct `exec` or `_worker`.
- `core/grpo.py` uses `mlx.utils.tree_flatten`'s list API and `tree_unflatten`.
  The redundant top-level notebook compatibility wrapper has been removed; use the
  framework task interfaces above for raw prompts and guarded task execution.

## Tests and verification limits

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

The environment has a broken globally installed `langsmith.pytest_plugin` entry point, so plugin
autoload is disabled for these tests. Tests use synthetic datasets and mocked generation, real
Python subprocess execution/timeouts, real game mechanics, a tiny
locally initialized MLX model with a real Adam optimizer step. No downloads are needed.
The model-family tests also cover real local checkpoint loading and generation plus quantized
LoRA gradients on small instances of the architectures listed above. They do not establish
full-checkpoint generation quality, training convergence, arbitrary quantization/MoE compatibility,
dataset download availability, or a security sandbox.
