# AutoEnvAgentRL

[中文](README.md) | [English](README-en.md)

**Generate interactive tasks from a knowledge graph, build executable sandboxes, and use them for Agentic RL training.**

AutoEnvAgentRL helps agents learn within environments with concrete business constraints: clarify requirements with users, call tools, use earlier results in subsequent actions, and deliver verifiable outcomes. Tasks, business data, user scripts, tools, and rewards share a contract. A common runtime and multiple validation stages keep task generation and sandbox execution consistent.

**Everything is done on your macbook/studio**

Entry point: `scripts/run_pipeline.sh` → `python -m env_factory.task_pipeline`. Task generation and sandbox construction use Code Agent only (Codex, Claude, OpenCode), with `gpt-6.1-sol` as the default author and builder model. Legacy staged generation, fixed prototypes, and the experiment loop have been removed.

Realism is unified: mechanisms are declared and verified in business contracts. Eligible multistep slots receive normal/change/recovery requirements in a 50/30/20 mix; singleton runs sample those probabilities. Per-mechanism results are written to `realism_summary.json`.

## Purpose

- **Generate tasks:** A Code Agent designs tasks, business data, tools, reference trajectories, and rewards from Scene graph paths. Routes include direct response, single-step, and multi-step agentic tasks.
- **Build environments:** Compile task contracts into sandboxes with persistent state, tool endpoints, a user simulator, and a reward endpoint. The Code Agent supplies additional business implementations.
- **Validate quality:** Run structural checks, independent semantic review, success and failure trajectories, counterfactual tests, and executable Docker validation to qualify training environments.
- **Train agents:** Run asynchronous PPO / GRPO on Apple Silicon with MLX, parallel rollouts, batched decoding, limited reuse of older-policy trajectories, and quantization-aware tuning (QAT).

```text
Scene knowledge graph
    ↓
Code Agent task generation → Preflight checks and semantic review
    ↓
Sandbox construction → Contract, reward, and executable Docker validation
    ↓
Docker sandbox services ← HTTP → Parallel rollout workers
                                      ↓
                         Complete trajectories and terminal rewards
                                      ↓
                              Actor: PPO / GRPO updates
```

Passing validation establishes eligibility under the current training contract. It does not establish that training improves model capabilities.

## 1. Installation and configuration

### Requirements

| Purpose | Requirements |
| --- | --- |
| Base environment | Python ≥ 3.14, uv |
| Graph and task generation | Accessible Neo4j instance, model services, an installed and authenticated Code Agent CLI |
| Sandbox builds and default training services | Docker Engine; Docker Desktop can be used on macOS |
| Local RL training | Apple Silicon, Metal, and the optional `rl` dependencies |

From the repository root:

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your Neo4j connection and model service configuration. Do not commit credentials. Main configuration roles:

| Configuration | Purpose |
| --- | --- |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Scene graph database |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | General model service and fallback values for role-specific settings |
| `ROLLOUT_LLM_*` | Agent model for standalone diagnostics; unused by the construction pipeline |
| `SANDBOX_LLM_*` | Sandbox user simulator and model-based evaluator |
| `GRAPH_SEEDS_FILE` | Graph seed file, one term per line |
| `WIKIPEDIA_DUMP_DB` | Optional local Wikipedia index; online sources are used when unset |

Task authoring, sandbox construction, and independent review use the selected **Code Agent CLI's authentication and model configuration**. `--code-agent-model` does not change the standalone diagnostic or user-simulator model. Select the RL policy model separately with `train_rl.sh --model`.

### Prepare the knowledge graph

Skip this step if a Scene graph is already available. Otherwise, configure seed terms and Neo4j, then run:

```bash
./scripts/build_graph.sh --help
./scripts/build_graph.sh
```

For a local Wikipedia index, see `scripts/download_wikipedia_dump.sh` and `scripts/index_wikipedia_dump.sh`. Graph construction writes incrementally to Neo4j.

## 2. Generate tasks and build sandboxes

```bash
# Show options; preview the command without calling models, Neo4j, or Docker
./scripts/run_pipeline.sh --help
./scripts/run_pipeline.sh --dry-run

# Generate and build one task by default
./scripts/run_pipeline.sh

# Generate and build five tasks in a specified output directory
./scripts/run_pipeline.sh --count 5 --output output/my_tasks

# Recovery tasks: stateful multistep generation with exercised events and faults

# Set the content language and Code Agent model
./scripts/run_pipeline.sh --code-agent-model gpt-6.1-sol --language en
```

### Common options

| Option | Default | Description |
| --- | --- | --- |
| `--count` | `1` | Number of new tasks to generate; acceptance is not guaranteed |
| `--output` | `output` | Root directory for tasks, sandboxes, and logs |
| `--code-agent` | `codex` | Supports `codex`, `claude`, and `opencode` |
| `--code-agent-model` | `gpt-6.1-sol` | Task authoring model; required explicitly for non-codex CLIs |
| `--sandbox-model` | `gpt-6.1-sol` (Codex) | Sandbox construction and independent review; other CLIs inherit the authoring model |
| `--language` | `zh-CN` | Generated content language, such as `en` or `ja`; does not translate platform logs |
| `--code-agent-timeout` | `600` | Cumulative author/repair subprocess budget in seconds; excludes parent validation. Each independent review has 180 seconds. |

See the [quality improvement plan](docs/plans/astra-quality-improvement.md) (Chinese) for priorities and acceptance criteria.

The default target mix is 20% direct response, 30% single-step agentic, and 50% multi-step agentic. Actual counts depend on allocation for small runs. There is no hard five-minute limit per sample.

To switch CLIs, install and authenticate the chosen tool, then provide a model identifier available to your account:

```bash
./scripts/run_pipeline.sh --code-agent claude --code-agent-model YOUR_MODEL
./scripts/run_pipeline.sh --code-agent opencode --code-agent-model PROVIDER/MODEL

```

To generate tasks only, or specifically generate multi-step tasks:

```bash
./scripts/generate_task.sh --count 5
./scripts/generate_task.sh --training-category multi_step_agentic --count 5
```

### Artifacts and logs

```text
output/
├── artifacts/
│   └── artifact-N/
│       ├── task/          # Task contracts, business data, user scripts, generation evidence
│       └── sandbox/       # Sandbox code, build logs, validation reports, status.json
├── logs/pipeline-*        # Separate console log for each main-pipeline invocation
├── generation.log        # Main-pipeline task-generation log
└── pipeline_events.jsonl  # Stages, timings, results, and run IDs
```

Task IDs increase automatically. Failed tasks retain diagnostic evidence while other qualified tasks proceed to construction.

Training requires **`training_ready == true`** in both `status.json` and `pipeline_result.json`, plus matching artifact hashes. Build `success` or a high quality score alone is insufficient. `training_ready` permits training; it does not certify real-policy behavior. Construction skips real rollouts and live reward calibration, records these checks as `not_run`, and uses the executable rubric for the full 10-point score. Actual rollouts and environment/reward monitoring run during RL training.

Main-pipeline exit codes: `0` passes the selected validation, `1` fails qualification, and `2` indicates a configuration or infrastructure failure.

## 3. Run RL training

The local training implementation lives in `src/rl/` and uses MLX/Metal on Apple Silicon. Prepare a qualified sandbox and a compatible MLX policy model:

```bash
uv sync --extra rl
./scripts/train_rl.sh --help

./scripts/train_rl.sh \
  --sandbox output/artifacts/artifact-1/sandbox \
  --model /absolute/path/to/mlx-model \
  --output output/rl_runs/grpo-new \
  --algorithm grpo --tuning qat \
  --epochs 2 --batch-size 1 --mini-batch-size 1 --rollout-group 4
```

Use `--algorithm ppo` for PPO with critic training; GRPO does not use a critic. Use `--tuning lora` for LoRA or `--tuning full` for full parameter training on floating-point weights with sufficient memory. Use `--tuning qat --qat-scope full` for full parameter QAT and standalone quantized export; this still requires memory for all FP32 masters and Adam states. Original Qwen3 supports `--thinking-mode thinking` / `no-thinking`. Supply a multi-sandbox dataset through `--tasks PATH`; see the [training guide](docs/agent_rl.md) for its format. Use a new output directory for a new run, or `--resume` with a compatible configuration to continue an existing run.

See the [step-by-step RL training guide (Chinese)](docs/agent_rl.md#使用指南从准备到导出) for setup, multiple sandboxes, thinking modes, full/QLoRA/QAT training, TensorBoard, recovery, and HF export.

Optimization options include configurable QLoRA targets, activation recomputation, chunked logits/prefill, packed QAT inference copies, independent checkpoint/publication intervals, and memory profiling. `scripts/convert_to_hf.sh` exports HF / PEFT models; see the [optimization and CUDA guide](docs/agent_rl.md#第一第二阶段优化配置).

Training writes TensorBoard metrics and one live rollout text trace per group by default.
Run `./scripts/train_dashboard.sh --logdir output/rl_runs/grpo-new/tensorboard` in another terminal
and open `http://127.0.0.1:6006`. Use `--rollout-trace-samples N` to change trace sampling,
or `--no-tensorboard` to disable event files while retaining append-only JSONL logs.
Held-out evaluation runs every 100 optimizer steps at batch boundaries. Zero-variance groups
are retried then explicitly skipped by default. Old artifacts are pruned while protecting
checkpoint, replay and worker references. Live metrics use JSONL; JSON exports are written on exit.

| Option | Meaning |
| --- | --- |
| `--epochs` | Full passes through the training sandbox dataset |
| `--batch-size` | Sandbox visits processed per batch; skipped visits do not count as updates |
| `--mini-batch-size` | Sandboxes per gradient update, retaining each complete rollout group |
| `--rollout-group` | Trajectories sampled per sandbox visit |
| `--rollout-workers` | Independent sampling processes |
| `--rollout-concurrency` | Concurrent trajectories per sampling process |

**Reward timing:** After an episode ends, the RL framework calls `/v1/reward` to obtain the combined process and outcome score. Intermediate actions do not fetch rewards. PPO assigns the terminal reward to the last action; GRPO uses terminal scores for within-group comparison. Older sandboxes must be rebuilt to support the terminal-scoring protocol.

The default **Docker Engine + manager** backend asynchronously prewarms sandboxes, serves rollouts over HTTP, and manages container reuse and cleanup. Kubernetes is not required for a single machine. See the [sandbox service guide](docs/agent_rl.md#单机-docker-engine--沙箱管理器默认) for standalone prewarming and remote services. For local development, explicitly select `--sandbox-backend local`.

## 4. Quality checks and further reading

Task and sandbox quality scores use a 0–10 scale. A hard-gate failure sets the effective score to zero; raw scores and failure evidence remain available for diagnosis.

```bash
uv run python examples/score_tasks.py output/artifacts/artifact-1/task/task.json --min-score 8 \
  --report output/task_quality_report.json
uv run python scripts/sandbox/score_sandbox_offline.py output/artifacts/artifact-1/sandbox
```

The detailed guides below are primarily in Chinese.

| Document | Contents |
| --- | --- |
| [Code Agent authoring](docs/code_agent_authoring.md) | Business design, tools, interactions, and reward contracts |
| [Task generation pipeline](docs/task_generation_pipeline.md) | Generation stages and artifacts |
| [Runtime integrity](docs/runtime_integrity.md) | Shared runtime, validation, and evidence boundaries |
| [Agent RL](docs/agent_rl.md) | Training options, asynchronous scheduling, QAT, Docker, and recovery |
| [Code structure](docs/code_structure.md) | Entry points and module boundaries |
| [Pipeline](docs/task_generation_pipeline.md) | Code Agent generation, construction and validation |
| [Training material certification](docs/production_readiness.md) | Production material certification, a separate scope from local RL training |

### Realistic business sandboxes

The shared runtime supports permissions and state transitions, conditional user
responses, virtual time and external events, bounded fault injection, idempotent
retries, and compensating operations. Code Agent authors these mechanics through
`business_lifecycle`; generation validation and generated sandboxes share the implementation.

The [procurement and cancellation example](examples/realism/README.md) includes
local and Docker HTTP exercises for competing inventory reservations, rejected
split orders, committed writes with lost responses, payment failures, refunds and
reservation release. Passing this example does not grant model-authored tasks
`training_ready` status.

`--generation-hops N` sets the maximum graph hop count (0..20, default 2). Each generation attempt samples uniformly from 0 through N; zero hops means one node. The main pipeline always generates new tasks; `--task-ids` has been removed.
