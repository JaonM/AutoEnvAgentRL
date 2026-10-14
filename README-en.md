<div align="center">

# AutoEnvAgentRL

**Automatically generate certified business sandboxes and train tool-using conversational agents with SFT and reinforcement learning.**

[中文](README.md) | [English](README-en.md)

![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20MLX-black)
![Docker](https://img.shields.io/badge/sandbox-Docker-2496ED)
![Tests](https://img.shields.io/badge/tests-pytest-green)

</div>

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Training an agent that gets real business work done (customer service, bookings, after-sales) takes many **interactive, automatically gradable** environments: the agent talks with a user over several turns, follows a policy, changes data through tools, and ends with an outcome that can be checked automatically. Building such environments by hand is expensive, and letting a language model generate them directly rarely yields a reward you can trust.

AutoEnvAgentRL turns this into an automated production line:

1. A code agent writes an **executable Python sandbox** (data, tools, a reference solution) and, in a separate session, an **independent oracle** for each business scenario.
2. The two implementations must **agree exactly** on the original world, every approval flip and perturbed worlds before the sandbox is certified.
3. Certified sandboxes are served over HTTP with a language-model user, ready for **SFT demonstration collection and reinforcement learning**.
4. Training runs on Apple Silicon: generation, certification and training all fit on one Mac.

## Features

- **Trustworthy reward**: environment and oracle are written independently and accepted only after differential certification, negative checks, branch and perturbation audits; the reward comes from the oracle, never from author-written code.
- **Deterministic grading, realistic dialogue**: the simulated user is a language model that only decides *how* to say things; whether a write is approved or which option is chosen comes from structured facts the platform enforces, so the reward stays deterministic.
- **Rich user scripts**: several goals, fallbacks, persistent unreasonable demands, misremembered identifiers, vague requests, undecided users, questions before agreeing, changes of mind and out-of-scope requests combine into long, varied dialogues.
- **Safety constraint**: any write the user did not agree to zeroes the episode; both a "proposal + consent token" protocol and plain-dialogue confirmation are supported.
- **Grading what was said**: the agent tells the user the result like a human agent would; the platform extracts the answer from the dialogue and verifies every number and identifier was actually said.
- **Hardened runtime**: generated code passes static sandbox rules; every sandbox is packed into a read-only, capability-free, non-root Docker image and re-verified offline.
- **Training designed for transfer**: tool names, consent protocol, tool-result format and system-prompt layout are randomized per episode, so the policy learns to read its instructions instead of memorizing one platform.
- **Local training**: asynchronous PPO / GRPO on MLX with parallel rollouts, batched decoding, LoRA / QAT / full fine-tuning and resumable checkpoints.

## How it works

```text
Business domains (sampled from a knowledge graph, or a list)
        │
        ▼
Request generation: a code agent writes rules, data, user facts and script → a model writes the
user's opening and the customer-facing policy manual
        │
        ▼
Sandbox build: answer schema → environment (tools + reference) ∥ independent oracle →
differential certification → arbitration → audits → every gate
        │
        ▼
Delivery: signed bundle → hardened Docker image verified offline → policy gate (a strong model
actually solves it in free dialogue)
        │
        ▼
tasks.json ──► demonstration collection → SFT cold start ──► reinforcement learning (GRPO / PPO)
```

Each certified sandbox (bundle) contains:

| File | Content |
| --- | --- |
| `request.json` | user request, business rules, customer-facing manual, tables, the user's private facts and script |
| `env.py` | tools and the reference solution (may import only `env_factory.runtime`) |
| `oracle.py` | the independently computed expected outcome (final state + answer) |
| `answer_schema.json` | answer format and graded fields |
| `gates.json`, `manifest.json` | certification report and file hashes signed for the platform identity |

## Quick start

### Requirements

| Purpose | Requirement |
| --- | --- |
| Base | macOS (Apple Silicon), Python ≥ 3.14, [uv](https://github.com/astral-sh/uv) |
| Sandbox generation and build | [Codex CLI](https://github.com/openai/codex), installed and logged in |
| User simulation, openings, judging | an OpenAI-compatible chat model service (DeepSeek by default) |
| Hardened images and training environments | Docker Desktop / Docker Engine |
| Optional | Neo4j (sampling business domains from a knowledge graph) |

### Install

```bash
git clone <repo-url> && cd EnvFactory
uv sync                 # generation and certification
uv sync --extra rl      # local training (MLX)
cp .env.example .env    # fill in LLM_API_KEY and friends
```

### Generate your first sandboxes

```bash
cat > domains.json <<'EOF'
[{"name": "Airline flight changes", "detail": "changes, refunds, baggage"},
 {"name": "Online retail returns", "detail": "returns, exchanges, refunds"}]
EOF

./scripts/run_pipeline.sh --domains-file domains.json --count 4 --parallel 2 --output output/run_demo
```

`output/run_demo/tasks.json` lists the sandboxes ready for training and `report.md` summarizes every task and why any were excluded. Running again with the same `--output` resumes where it stopped.

### Try a sandbox

```bash
# Serve it over HTTP; in natural mode a language model plays the user
.venv/bin/python -m env_factory serve output/run_demo/build/gen-000/bundle --port 8090 --user-mode natural

# Let a strong model play 4 free-dialogue episodes (policy gate)
.venv/bin/python -m env_factory policy output/run_demo/build/gen-000/bundle --k 4 --user-mode natural
```

### Train

```bash
./scripts/train_rl.sh --tasks output/run_demo/tasks.json --model models/Qwen3-0.6B \
  --output output/train-demo --tuning lora --rollout-group 8 --rollout-concurrency 8 --max-steps 30
```

SFT cold start, demonstration collection and behaviour checks: [docs/rl-training.md](docs/rl-training.md).

## Usage

### End-to-end production line

`./scripts/run_pipeline.sh` is the main entry point (equivalent to `python -m env_factory run`):

| Option | Meaning |
| --- | --- |
| `--count N` | number of tasks to generate |
| `--domains-file FILE` / `--split NAME` | use a domain list, or sample a knowledge-graph split (needs Neo4j) |
| `--mix SPEC` | task shapes, e.g. `direct=0.1,simple=0.25,ordinary=0.25,complex=0.4` |
| `--english-share X` | share of English tasks |
| `--parallel N` | concurrent generations / builds |
| `--policy-k N` | policy-gate episodes per sandbox (0 skips) |
| `--no-docker` | skip hardened images (not recommended) |

Full reference: `./scripts/run_pipeline.sh --help`.

### Command line

| Command | Purpose |
| --- | --- |
| `python -m env_factory run` | run or resume the whole production line |
| `python -m env_factory generate` | build requests for new business domains |
| `python -m env_factory domains` | sample domains from the knowledge graph × business grid |
| `python -m env_factory build` / `batch` | build one / many sandboxes |
| `python -m env_factory verify BUNDLE [--resign]` | check a bundle's signature; re-certify after platform changes |
| `python -m env_factory docker BUNDLE` | build the hardened image and verify it offline |
| `python -m env_factory serve BUNDLE` | serve a sandbox over HTTP |
| `python -m env_factory policy BUNDLE` | policy gate: an independent model plays k episodes |
| `python -m env_factory findings` / `clarify` | list open business-rule questions / record the owner's answers |

### HTTP API

| Route | Meaning |
| --- | --- |
| `POST /reset` | start an episode; returns what the agent may see |
| `POST /step` | one action: call a tool, talk to the user, wait, record an answer |
| `POST /reward` | end and grade |
| `POST /release` | release the episode |
| `GET /health` | health check |

## Project structure

```text
src/
  env_factory/          main pipeline: sandbox generation, certification and serving
    runtime.py          the contract generated code runs against (write, BusinessError, Episode)
    checks.py           self-checks run inside code-agent sessions
    core/               task model and grading: spec, sandbox, worlds, answers, reward, rubric, extract
    serving/            HTTP service, the agent's view, simulated user
    certify/            gates, signing, Docker, policy gate, tool fuzzing, policy sufficiency
    authoring/          the build of one sandbox and its prompts
    production/         end-to-end line: domains, request generation, openings, manuals, domain worlds
  env_factory_v3/       compatibility alias of the former package name (older sandboxes still import it)
  rl/                   training: rollout, algorithms, MLX backend, checkpoints, observability, export
scripts/                production-line and training entry points
experiments/            SFT demonstration collection, selection, training, interpolation, behaviour checks
tests/                  unit and integration tests
docs/                   architecture, domain-world design, training guide
```

## Configuration

`.env` (see `.env.example`):

| Variable | Meaning |
| --- | --- |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | default model service; every language-model role falls back to it |
| `USER_SIM_*` | simulated user (optional override) |
| `JUDGE_*` | answer extraction, rubric judge, simulator audit (optional override) |
| `OPENING_*` | opening and manual writers (optional override) |
| `DOMAIN_*` | knowledge-graph domain proposals (optional override) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | optional: knowledge graph |
| `SANDBOX_SERVICE_TOKEN` | optional: bearer token required by `serve` |
| `SIMULATOR_AUDIT=off` | optional: disable the simulator audit of failed episodes |

Per-episode protocol randomization during training is controlled by `ENVFACTORY_*` variables; see [docs/rl-training.md](docs/rl-training.md).

## Testing

```bash
.venv/bin/python -m pytest -q
```

Tests never call external model services; model-dependent parts use deterministic stand-ins.

## Documentation

| Document | Content |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | production line, gates, grading, the free-dialogue user, HTTP protocol |
| [docs/domain-world-design.md](docs/domain-world-design.md) | domain worlds: one certified environment hosting many tasks |
| [docs/rl-training.md](docs/rl-training.md) | SFT cold start and reinforcement-learning guide (Chinese) |

## Contributing

Issues and pull requests are welcome. Before submitting:

1. run `.venv/bin/python -m pytest -q` and make sure it passes;
2. changes to `runtime.py`, `core/`, `serving/` or `certify/` (except `policy_gate`) change the platform identity: existing sandboxes need `verify --resign`; say so in the PR;
3. never commit model weights, credentials or anything under `output/`.

## License

No open-source license has been chosen yet.
