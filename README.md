<div align="center">

# AutoEnvAgentRL

**自动生成可认证的业务沙箱，用 SFT + 强化学习训练会使用工具的对话 Agent。**

[中文](README.md) | [English](README-en.md)

![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20MLX-black)
![Docker](https://img.shields.io/badge/sandbox-Docker-2496ED)
![Tests](https://img.shields.io/badge/tests-pytest-green)

</div>

---

## 目录

- [简介](#简介)
- [特性](#特性)
- [工作原理](#工作原理)
- [快速开始](#快速开始)
- [使用](#使用)
- [项目结构](#项目结构)
- [配置](#配置)
- [测试](#测试)
- [文档](#文档)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介

训练一个能在真实业务里办事的 Agent（客服、预订、售后……），需要大量**可交互、可判分**的环境：Agent 要和用户多轮对话、按政策办事、调用工具改动数据，最后得到一个能自动判定对错的结果。手工搭建这样的环境成本很高，让大模型直接生成又很难保证判分可信。

AutoEnvAgentRL 把这件事做成一条自动生产线：

1. 由代码 Agent 为每个业务场景编写**可执行的 Python 沙箱**（数据、工具、参考解）和一个**独立的验证器（verifier）**；
2. 两份独立实现必须在原始世界、审批翻转和扰动世界上**完全一致**，才算通过认证；
3. 认证后的沙箱以 HTTP 服务对外，配有大模型扮演的用户，可直接用于 **SFT 示范采集和强化学习**；
4. 训练框架在 Apple Silicon 上运行，一台 Mac 即可完成生成、认证和训练。

## 特性

- **可信的奖励**：环境与 verifier 由不同会话独立编写，经差分认证、反例检查、分支与扰动审计后才接受；奖励由 verifier 计算，作者从不编写奖励代码。
- **确定的判分，真实的对话**：模拟用户由大模型扮演，只负责"怎么说"；是否同意写操作、选择哪个方案等决定都来自结构化事实，由平台执行，因此奖励完全确定。
- **丰富的用户剧本**：多诉求、备选方案、坚持不合理要求、记错编号、意图含糊、拿不定主意、追问后再决定、中途改主意、超出范围等，组合出多轮、多样的对话。
- **安全约束**：未经用户同意的写操作使整局奖励为 0；支持"提议 + 授权凭证"和"纯对话确认"两种同意协议。
- **按说话判分**：Agent 像真人客服一样把结果说给用户，平台从对话中抽取答案，并校验每个数字和编号确实被说出。
- **加固的运行环境**：生成代码通过静态沙箱规则检查；每个沙箱打包成只读、无权限、非 root 的 Docker 镜像，并在断网条件下复验全部认证项。
- **为迁移而设计的训练**：每局随机化工具名、同意协议、工具返回格式和 system prompt 版式，让策略学会读说明再行动。
- **本地训练**：基于 MLX 的异步 PPO / GRPO，支持并行 rollout、批量解码、LoRA / QAT / 全参数微调、断点续训。

## 工作原理

```text
业务领域（知识图谱采样或领域清单）
        │
        ▼
请求生成：代码 Agent 写业务规则、数据、用户事实与剧本 → 大模型写用户开场白与对外政策手册
        │
        ▼
沙箱构建：答案格式 → 环境（工具 + 参考解）∥ 独立 verifier → 差分认证 → 仲裁 → 审计 → 全部门禁
        │
        ▼
交付：签名的沙箱包 → 加固 Docker 镜像离线复验 → policy gate（强模型在自由对话中实际求解）
        │
        ▼
tasks.json ──► 示范采集 → SFT 冷启动 ──► 强化学习（GRPO / PPO）
```

每个认证沙箱（bundle）包含：

| 文件 | 内容 |
| --- | --- |
| `request.json` | 用户请求、业务规则、对外手册、数据表、用户私人事实与剧本 |
| `env.py` | 工具实现与参考解（只能导入 `env_factory.runtime`） |
| `verifier.py` | 独立计算的期望结果（最终状态 + 答案） |
| `answer_schema.json` | 答案格式与判分字段 |
| `gates.json`、`manifest.json` | 认证报告与按平台身份签名的文件哈希 |

## 快速开始

### 环境要求

| 用途 | 要求 |
| --- | --- |
| 基础 | macOS（Apple Silicon）、Python ≥ 3.14、[uv](https://github.com/astral-sh/uv) |
| 沙箱生成与构建 | 已安装并登录的 [Codex CLI](https://github.com/openai/codex) |
| 用户模拟、开场白、评审 | OpenAI 兼容的对话模型服务（默认 DeepSeek） |
| 加固镜像与训练环境 | Docker Desktop / Docker Engine |
| 可选 | Neo4j（从知识图谱采样业务领域） |

### 安装

```bash
git clone <repo-url> && cd EnvFactory
uv sync                 # 生成与认证
uv sync --extra rl      # 本地训练（MLX）
cp .env.example .env    # 填写 LLM_API_KEY 等
```

### 生成第一批沙箱

```bash
# 准备一个领域清单
cat > domains.json <<'EOF'
[{"name": "Airline flight changes", "detail": "改签、退票、行李"},
 {"name": "Online retail returns", "detail": "退货、换货、退款"}]
EOF

./scripts/run_pipeline.sh --domains-file domains.json --count 4 --parallel 2 --output output/run_demo
```

运行结束后，`output/run_demo/tasks.json` 列出可直接训练的沙箱，`report.md` 汇总每个任务的结果与淘汰原因。同一个 `--output` 再次运行会从中断处续跑。

### 试玩一个沙箱

```bash
# 以 HTTP 服务方式启动，自由对话模式由大模型扮演用户
.venv/bin/python -m env_factory serve output/run_demo/build/gen-000/bundle --port 8090 --user-mode natural

# 让一个强模型在自由对话中实际玩 4 局（policy gate）
.venv/bin/python -m env_factory policy output/run_demo/build/gen-000/bundle --k 4 --user-mode natural
```

### 训练

```bash
./scripts/train_rl.sh --tasks output/run_demo/tasks.json --model models/Qwen3-0.6B \
  --output output/train-demo --tuning lora --rollout-group 8 --rollout-concurrency 8 --max-steps 30
```

SFT 冷启动、示范采集与行为检查见 [docs/rl-training.md](docs/rl-training.md)。

## 使用

### 端到端生产线

`./scripts/run_pipeline.sh` 是主入口（等价于 `python -m env_factory run`），常用参数：

| 参数 | 说明 |
| --- | --- |
| `--count N` | 生成的任务数 |
| `--domains-file FILE` / `--split NAME` | 使用领域清单，或从知识图谱划分采样（需要 Neo4j） |
| `--mix SPEC` | 任务形态比例，如 `direct=0.1,simple=0.25,ordinary=0.25,complex=0.4` |
| `--english-share X` | 英文任务占比 |
| `--parallel N` | 并发生成 / 构建数 |
| `--policy-k N` | policy gate 每个沙箱的对局数（0 跳过） |
| `--no-docker` | 跳过加固镜像（不推荐） |

完整说明：`./scripts/run_pipeline.sh --help`。

### 命令行

| 命令 | 作用 |
| --- | --- |
| `python -m env_factory run` | 运行或续跑整条生产线 |
| `python -m env_factory generate` | 为新的业务领域生成构建请求 |
| `python -m env_factory domains` | 从知识图谱 × 业务网格采样领域 |
| `python -m env_factory build` / `batch` | 构建一个 / 一批沙箱 |
| `python -m env_factory verify BUNDLE [--resign]` | 校验沙箱签名；平台代码变化后重新认证 |
| `python -m env_factory docker BUNDLE` | 构建加固镜像并离线复验 |
| `python -m env_factory serve BUNDLE` | 以 HTTP 服务运行沙箱 |
| `python -m env_factory policy BUNDLE` | policy gate：独立模型实际求解 k 局 |
| `python -m env_factory findings` / `clarify` | 查看业务规则中的待澄清问题 / 写入业务方答复 |

### HTTP 接口

| 路由 | 说明 |
| --- | --- |
| `POST /reset` | 开始一局，返回 Agent 可见的说明 |
| `POST /step` | 执行一个动作：调用工具、对用户说话、等待、记录答案 |
| `POST /reward` | 结束并判分 |
| `POST /release` | 释放 episode |
| `GET /health` | 健康检查 |

## 项目结构

```text
src/
  env_factory/          主链路：沙箱生成、认证与服务
    runtime.py          生成代码的运行契约（write、BusinessError、Episode）
    checks.py           代码 Agent 会话中运行的自检
    core/               任务模型与判分：spec、sandbox、worlds、answers、reward、rubric、extract
    serving/            HTTP 服务、Agent 视角、模拟用户
    certify/            认证门禁、签名、Docker、policy gate、工具模糊测试、手册充分性
    authoring/          单个沙箱的构建流程与提示
    production/         端到端生产线：领域、请求生成、开场白、对外手册、领域世界
  env_factory_v3/       旧包名的兼容别名（早期沙箱中的生成代码仍可导入）
  rl/                   训练框架：rollout、算法、MLX 后端、检查点、观测、导出
scripts/                生产线与训练入口
experiments/            SFT 示范采集、筛选、训练、插值、行为检查等工具
tests/                  单元与集成测试
docs/                   架构、领域世界设计、训练指南
```

## 配置

`.env`（参考 `.env.example`）：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 默认模型服务，所有大模型角色都会回退到它 |
| `USER_SIM_*` | 模拟用户（可选覆盖） |
| `JUDGE_*` | 答案抽取、主观评审、模拟用户审计（可选覆盖） |
| `OPENING_*` | 开场白与对外手册撰写（可选覆盖） |
| `DOMAIN_*` | 知识图谱领域提议（可选覆盖） |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | 可选：知识图谱 |
| `SANDBOX_SERVICE_TOKEN` | 可选：`serve` 要求的访问令牌 |
| `SIMULATOR_AUDIT=off` | 可选：关闭失败对局的模拟用户审计 |

训练时 episode 的协议随机化通过 `ENVFACTORY_*` 环境变量控制，见 [docs/rl-training.md](docs/rl-training.md)。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试不调用外部模型服务；涉及模型的部分使用确定性的替身。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 生产线、门禁、判分、自由对话用户、HTTP 协议的详细说明 |
| [docs/domain-world-design.md](docs/domain-world-design.md) | 领域世界：一个认证环境承载多个任务 |
| [docs/rl-training.md](docs/rl-training.md) | SFT 冷启动与强化学习训练指南 |

## 贡献

欢迎提交 Issue 和 Pull Request。提交前请：

1. 运行 `.venv/bin/python -m pytest -q` 并保证通过；
2. 修改 `runtime.py`、`core/`、`serving/`、`certify/`（`policy_gate` 除外）会改变平台身份，已有沙箱需要 `verify --resign` 重新认证，请在 PR 中说明；
3. 不要提交模型权重、凭据或 `output/` 下的产物。

## 许可证

尚未指定开源许可证。
