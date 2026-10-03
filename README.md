# AutoEnvAgentRL

[中文](README.md) | [English](README-en.md)

**从知识图谱生成可交互任务，构建可执行沙箱，并用于 Agentic RL 训练。**

**一切都在你的 macbook/studio 上完成。**

项目目标是让 Agent 在有真实业务约束的环境中学习：向用户澄清需求、调用工具、使用前一步的结果完成后续操作，最终交付可验证的业务结果。项目将任务、业务数据、用户剧本、工具和奖励组织为统一契约，通过共享运行时和多层验收减少生成与构建的不一致。

## 项目主旨

- **生成任务**：Code Agent 根据 Scene 图谱路径设计任务、业务数据、工具、参考轨迹与奖励，覆盖直接回答、单步和多步 Agentic 任务。
- **构建环境**：将任务契约编译为沙箱，提供状态持久化、工具接口、用户模拟器与奖励接口；Code Agent 补充业务实现。
- **验证质量**：前置结构检查、独立语义审核、成功与失败轨迹、反事实测试以及 live 验收，筛选可用于训练的环境。
- **训练 Agent**：在 Apple Silicon 上使用 MLX 运行异步 PPO / GRPO，支持并行 rollout、批量解码、有限旧策略数据复用及量化感知微调（QAT）。

```text
Scene 知识图谱
    ↓
Code Agent 生成任务 → 前置检查与语义审核
    ↓
沙箱构建 → 契约、奖励与真实 rollout 验收
    ↓
Docker 沙箱服务 ← HTTP → 并行 rollout workers
                              ↓
                     完整轨迹与终局奖励
                              ↓
                     Actor：PPO / GRPO 更新
```

通过门禁表示满足当前训练接入条件，不代表已经证明训练后的模型能力提升。

## 1. 安装与配置

### 环境要求

| 用途 | 要求 |
| --- | --- |
| 项目基础环境 | Python ≥ 3.14、uv |
| 图谱与任务生成 | 可访问的 Neo4j、模型服务、已安装并认证的 Code Agent CLI |
| 沙箱构建与默认训练服务 | Docker Engine；macOS 可使用 Docker Desktop |
| 本地 RL 训练 | Apple Silicon、Metal，以及 `rl` 可选依赖 |

在项目根目录执行：

```bash
uv sync
cp .env.example .env
```

编辑 `.env`，填写 Neo4j 连接信息和模型服务配置。不要提交实际凭据。主要配置角色如下：

| 配置 | 用途 |
| --- | --- |
| `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE` | Scene 图谱数据库 |
| `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` | 通用模型服务及角色配置的回退值 |
| `ROLLOUT_LLM_*` | live 验收中的 Agent 模型 |
| `SANDBOX_LLM_*` | 沙箱用户模拟器与模型评分器 |
| `GRAPH_SEEDS_FILE` | 图谱种子词文件，每行一个词语 |
| `WIKIPEDIA_DUMP_DB` | 可选的本地 Wikipedia 索引；未配置时使用在线来源 |

任务作者、沙箱构建和独立审查使用所选 **Code Agent CLI 的认证与模型配置**。`--code-agent-model` 不改变 live 验收或用户模拟器模型；RL 训练的策略模型另外通过 `train_rl.sh --model` 指定。

### 准备知识图谱

已有 Scene 图谱时可跳过此步。否则先配置种子词和 Neo4j，再运行：

```bash
./scripts/build_graph.sh --help
./scripts/build_graph.sh
```

本地 Wikipedia 数据索引的准备方法见脚本 `scripts/download_wikipedia_dump.sh`、`scripts/index_wikipedia_dump.sh`。图谱构建会增量写入 Neo4j。

## 2. 生成任务并构建沙箱

```bash
# 查看参数；预览命令，不调用模型、图谱或 Docker
./scripts/run_pipeline.sh --help
./scripts/run_pipeline.sh --dry-run

# 默认生成并构建 1 个任务
./scripts/run_pipeline.sh

# 生成并构建 5 个任务，指定输出目录
./scripts/run_pipeline.sh --count 5 --output output/my_tasks

# 指定生成语种与 Code Agent 模型
./scripts/run_pipeline.sh --code-agent-model gpt-6-luna --language en
```

### 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--count` | `1` | 新生成的任务数量；不保证全部通过验收 |
| `--output` | `output` | 任务、沙箱与日志的输出根目录 |
| `--task-ids` | 未设置 | 重建已有任务，例如 `1` 或 `1,2`；不能与 `--count` 同用 |
| `--code-agent` | `codex` | 支持 `codex`、`claude`、`opencode` |
| `--code-agent-model` | `gpt-6-luna` | 作者、构建与独立审查模型；非 codex 必须显式设置 |
| `--language` | `zh-CN` | 生成内容语种，例如 `en`、`ja`；不翻译平台日志 |
| `--validation` | `live` | `live` 执行真实模型验收；`offline` 不能替代训练资格验证 |
| `--code-agent-timeout` | `600` | 任务生成与修复共享的超时秒数 |

默认目标类别比例为直接回答 20%、单步 Agentic 30%、多步 Agentic 50%；小样本实际数量以分配结果为准。主链路没有单样本 5 分钟硬截止。

切换 CLI 时，先完成对应工具安装与认证，再传入账号可用的模型标识：

```bash
./scripts/run_pipeline.sh --code-agent claude --code-agent-model YOUR_MODEL
./scripts/run_pipeline.sh --code-agent opencode --code-agent-model PROVIDER/MODEL

# 使用同一输出根目录，跳过生成并重建 task-1
./scripts/run_pipeline.sh --output output/my_tasks --task-ids 1
```

只生成任务，或专门生成多步任务：

```bash
./scripts/generate_task.sh --count 5
./scripts/generate_task.sh --training-category multi_step_agentic --count 5
```

### 产物与日志

```text
output/
├── task/task-N/           # 任务契约、业务数据、用户剧本与生成证据
├── sandbox/task-N/        # 沙箱代码、构建日志、验收报告与 status.json
├── logs/pipeline-*        # 每次主链路运行的独立终端日志
├── generation.log        # 主链路任务生成日志
└── pipeline_events.jsonl  # 阶段、耗时、结果与运行 ID
```

新任务编号自动递增。失败任务保留诊断信息，其他合格任务继续构建。

训练接入检查 `status.json` 和 `pipeline_result.json` 的 **`training_ready == true`**，并校验产物哈希；仅有构建 `success` 或高质量分不够。主链路退出码：`0` 通过当前验收，`1` 不合格，`2` 配置或基础设施失败。

## 3. 运行 RL 训练

当前本地训练实现位于 `src/rl/`，使用 Apple Silicon 的 MLX/Metal。准备合格沙箱与兼容的 MLX 策略模型：

```bash
uv sync --extra rl
./scripts/train_rl.sh --help

./scripts/train_rl.sh \
  --sandbox output/sandbox/task-1 \
  --model /absolute/path/to/mlx-model \
  --output output/rl_runs/grpo-new \
  --algorithm grpo --tuning qat \
  --epochs 2 --batch-size 1 --mini-batch-size 1 --rollout-group 4
```

`--algorithm ppo` 启用 PPO 和 critic 训练；GRPO 不使用 critic。`--tuning lora` 可切换为 LoRA；`--tuning full` 对浮点底座进行全参训练（需足够内存）。`--tuning qat --qat-scope full` 启用全参 QAT，并导出完整量化模型；它仍需全量 FP32 master/Adam 内存。原版 Qwen3 可用 `--thinking-mode thinking` / `no-thinking` 控制思考模式。多沙箱数据集通过 `--tasks PATH` 传入，格式见[训练文档](docs/agent_rl.md)。新运行使用新的输出目录，续训使用 `--resume` 并保持运行配置兼容。

按步骤操作见 [RL 训练使用指南](docs/agent_rl.md#使用指南从准备到导出)：环境与模型准备、多沙箱、thinking、全参/QLoRA/QAT、TensorBoard、续训和 HF 导出。

新增全层 QLoRA targets、activation 重算、logits/prefill 分块、packed QAT 推理副本、独立提交/发布频率和实测内存探针。`scripts/convert_to_hf.sh` 可导出标准 HF / PEFT 模型；详见[优化配置](docs/agent_rl.md#第一第二阶段优化配置)和[CUDA 转换](docs/agent_rl.md#转换到-cuda-生态)。

训练默认记录 TensorBoard 曲线及每组一条实时 rollout 文本轨迹。在另一个终端运行
`./scripts/train_dashboard.sh --logdir output/rl_runs/grpo-new/tensorboard`，
浏览器打开 `http://127.0.0.1:6006`。用 `--rollout-trace-samples N` 调整轨迹数量，
`--no-tensorboard` 关闭看板事件；追加式 JSONL 日志仍保留。详见[看板说明](docs/agent_rl.md#实时-tensorboard-看板)。
默认每 100 个 optimizer step 在 batch 边界评估 held-out 任务；零方差组有限重试后跳过。
运行中清理受保护引用之外的旧产物，实时指标使用 JSONL，完整 JSON 在退出时导出。
参数与恢复语义见[长训练说明](docs/agent_rl.md#长训练稳定性性能与周期评估)。

| 参数 | 含义 |
| --- | --- |
| `--epochs` | 遍历训练沙箱数据集的次数 |
| `--batch-size` | 每批按完成顺序处理的沙箱访问数，跳过不计有效更新 |
| `--mini-batch-size` | 每个梯度更新 step 使用的沙箱数，保留各自完整 rollout group |
| `--rollout-group` | 每次访问一个沙箱时采样的轨迹数 |
| `--rollout-workers` | 独立采样进程数 |
| `--rollout-concurrency` | 每个采样进程同时推进的轨迹数 |

**奖励时机**：episode 结束后，由 RL 框架调用 `/v1/reward`，统一获取过程与结果综合分；中间 action 不取分。PPO 将终局奖励放在最后一个 action，GRPO 使用终局分做组内比较。旧沙箱需要重建运行时以支持终局评分协议。

默认使用 **Docker Engine + 管理器**异步预热沙箱，通过 HTTP 执行 rollout，并管理容器复用与回收。无需单机部署 Kubernetes。独立预热与远程服务接入见[沙箱服务说明](docs/agent_rl.md#单机-docker-engine--沙箱管理器默认)；本地开发可显式选择 `--sandbox-backend local`。

## 4. 质量检查与进一步阅读

任务和沙箱质量评分均为 0–10 分制；硬门禁失败时有效分为 0，原始分和失败证据用于诊断。

```bash
uv run python examples/score_tasks.py output/task --min-score 8 \
  --report output/task_quality_report.json
uv run python scripts/sandbox/score_sandbox_offline.py output/sandbox/task-1
```

| 文档 | 内容 |
| --- | --- |
| [Code Agent 编写规范](docs/code_agent_authoring.md) | 业务设计、工具、交互与奖励契约 |
| [任务生成流水线](docs/task_generation_pipeline.md) | 生成阶段与产物结构 |
| [运行时一致性](docs/runtime_integrity.md) | 共享运行时、验收与证据边界 |
| [Agent RL](docs/agent_rl.md) | 训练参数、异步调度、QAT、Docker 与恢复 |
| [代码结构](docs/code_structure.md) | 代码入口与模块边界 |
| [循环实验](docs/loop_experiments.md) | 多轮实验与诊断工具 |
| [训练素材准备认证](docs/production_readiness.md) | production 素材认证；与本地 RL 训练是不同范围 |
