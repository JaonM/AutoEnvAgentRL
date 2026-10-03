# Apple Silicon 异步 Agent RL

实现位于 `src/rl/`。独立 rollout worker 在真实沙箱采样，Actor（策略训练侧）在本机 MLX/Metal 更新策略。每个 rollout worker 默认同时推进两条 rollout。
算法只有 **PPO / GRPO**（默认 GRPO），两者共用行为概率比、裁剪目标与有限轨迹复用。

## 使用指南：从准备到导出

本节按实际操作顺序说明使用方法。所有命令在项目根目录执行；沙箱路径和输出目录需替换为实际值。
训练后端为 Apple Silicon 上的 MLX/Metal；HF/CUDA 转换用于部署产物，不会把训练后端切换为 CUDA。

### 1. 准备环境与模型

```bash
uv sync --extra rl
./scripts/train_rl.sh --help
```

需要 Python ≥ 3.14、Apple Silicon/Metal，以及运行中的 Docker Engine（脚本默认使用 Docker 沙箱）。
用户模拟器与模型评分器仍使用外部模型服务，按项目 `.env.example` 配置 `SANDBOX_LLM_*` 等变量；
训练 policy 本身由 `--model` 指向本地模型。仅运行 RL 不要求重新执行图谱生成流程。
本地开发可显式传 `--sandbox-backend local`，已有独立 HTTP 服务使用 `--sandbox-services`，
详见[沙箱服务管理](#单机-docker-engine--沙箱管理器默认)。

本项目已验证的 0.6B 路径是 `models/Qwen3-0.6B`（官方浮点底座）与
`models/Qwen3-0.6B-4bit`（由它转换的 MLX affine 量化底座）。尚未下载时可执行：

```bash
uv run --extra rl hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/Qwen3-0.6B

.venv/bin/python -m mlx_lm.convert \
  --hf-path models/Qwen3-0.6B --mlx-path models/Qwen3-0.6B-4bit \
  -q --q-bits 4 --q-group-size 64 --dtype float32
```

已有模型不必重复下载/转换。普通全参训练使用浮点底座；QLoRA 和 QAT 可使用 MLX 量化底座。
GGUF 不是此训练框架的输入格式。

### 2. 准备一个或多个合格沙箱

沙箱须通过项目生成与验收流程：`status.json` 和 `pipeline_result.json` 均为
`training_ready=true`，且产物哈希匹配。不要通过手动修改状态文件跳过验收。

- 单沙箱：`--sandbox output/sandbox/task-1`。
- 多沙箱：`--tasks tasks.json`；与 `--sandbox` 二选一。

例如在项目根目录保存 `tasks.json`：

```json
{
  "tasks": [
    {"id": "train-a", "sandbox": "output/sandbox/a", "split": "train"},
    {"id": "train-b", "sandbox": "output/sandbox/b", "split": "train"},
    {"id": "held-out", "sandbox": "output/sandbox/c", "split": "eval"}
  ]
}
```

相对路径基于清单所在目录，任务 ID 和沙箱内容身份必须唯一，至少需要一个 train 任务。
`eval` 沙箱不参与训练采样和 replay；用于独立评估。每轮遍历每个训练沙箱一次，
`weight` 不改变此覆盖语义。没有 eval 沙箱也能运行，但不能将训练集奖励视为独立评估结果。

### 3. 启动 Qwen3-0.6B thinking GRPO

以下起步配置使用全 28 层 QLoRA、低并发和较小物理批量。它定义运行预算，不保证每个沙箱都有有效更新。

```bash
./scripts/train_rl.sh \
  --tasks tasks.json \
  --output output/rl_runs/qwen06-qlora-thinking \
  --model models/Qwen3-0.6B-4bit \
  --algorithm grpo --tuning lora \
  --layers 28 --lora-targets all-linear --rank 8 \
  --thinking-mode thinking --temperature 0.6 \
  --epochs 2 --batch-size 1 --mini-batch-size 1 \
  --rollout-workers 1 --rollout-concurrency 1 --rollout-group 4 \
  --max-steps 6 --max-tokens 1024 --max-context 4096 \
  --micro-batch-size 1 --max-tokens-per-micro-batch 4096 \
  --gradient-checkpointing --logits-chunk-size 32 \
  --prefill-chunk-size 256 --profile-memory
```

`--output` 使用新的运行目录；不要指向模型目录。只训练一个沙箱时替换 `--tasks` 参数即可。
首次运行会做模型/沙箱检查及相应评估，开始写 optimizer 曲线前可能需要等待环境调用。

| 要切换的行为 | 修改方式 |
| --- | --- |
| 不生成思考过程 | `--thinking-mode no-thinking`，使用新的输出目录 |
| PPO + critic | `--algorithm ppo` |
| 只训练最后一层 q/v LoRA | `--layers 1 --lora-targets self_attn.q_proj,self_attn.v_proj` |
| 导出合并模型 | 增加 `--lora-merge-export` |
| 导出再量化合并模型 | 同时增加 `--lora-merge-export --lora-requantize-export` |

thinking 将 `enable_thinking=True` 传入 tokenizer 模板；no-thinking 传入 False；auto 沿用模板默认。
原版 Qwen3-0.6B/4B 支持两种模式，不应把只支持非思考的 Instruct 模型当成等价替代。
默认训练全部新生成的 assistant token，包括思考、工具调用和最终回答；工具结果与 prompt 不参与 loss。
当前使用任务终局奖励，没有额外的思考步骤奖励，也没有独立的 `reasoning_effort` 参数。

`--max-tokens` 是每次 assistant 生成的总预算，思考与回答/工具调用共用；不是单独的 thinking 预算。
要求当前 prompt 长度 + 此预算 ≤ `--max-context`。若思考经常截断，可增大生成预算并同步检查上下文、
物理 batch 和内存预算。0.6B 对工具协议提示敏感；不闭合的思考/工具标签会记录协议错误，不自动补齐。

### 4. 全参 GRPO 与全参 QAT

普通全参 GRPO 使用浮点底座，更新全部语言模型参数：

```bash
./scripts/train_rl.sh \
  --tasks tasks.json --output output/rl_runs/qwen06-full-grpo \
  --model models/Qwen3-0.6B --algorithm grpo --tuning full \
  --thinking-mode thinking --temperature 0.6 \
  --rollout-workers 1 --rollout-concurrency 1 --rollout-group 4 \
  --micro-batch-size 1 --max-tokens-per-micro-batch 4096 \
  --max-tokens 1024 --max-context 4096 \
  --gradient-checkpointing --logits-chunk-size 32 --profile-memory
```

全参 QAT 则使用以下完整命令：

```bash
./scripts/train_rl.sh \
  --tasks tasks.json --output output/rl_runs/qwen06-full-qat-grpo \
  --model models/Qwen3-0.6B-4bit --algorithm grpo \
  --tuning qat --qat-scope full --packed-inference \
  --thinking-mode thinking --temperature 0.6 \
  --rollout-workers 1 --rollout-concurrency 1 --rollout-group 4 \
  --micro-batch-size 1 --max-tokens-per-micro-batch 4096 \
  --max-tokens 1024 --max-context 4096 \
  --gradient-checkpointing --logits-chunk-size 32 --profile-memory
```

| 训练模式 | Actor 更新内容 | 部署产物 |
| --- | --- | --- |
| `lora` | 选定模块的 adapter；底座冻结 | `lora_adapter/`，可选合并模型 |
| `full` | 全部浮点参数 | `full_model/` |
| `qat --qat-scope full` | 全部参数；Linear/Embedding 做权重量化感知训练 | `qat_model/` |
| `qat --qat-scope projections`（默认 scope） | 指定末尾层 q/v 的 master 权重 | `qat_quantized.safetensors` 等 overlay，依赖原底座 |

`--packed-inference` 在 QAT 模式中让 reference/rollout worker 使用 packed 量化副本，
Actor 仍保留 FP32 master、梯度和 Adam 状态。它不量化 KV cache，也不会让普通 full 模式自动使用量化推理副本。

本机 48 GiB，0.6B 普通全参在一个 worker 下持久张量下界约 **14.31 GB**；
全参 QAT + packed inference 约 **10.28 GB**，均未包含 activation、KV 和临时缓冲。
真实 0.6B 全参 QAT GRPO 已验证更新与续训；普通全参的集成验证使用小型 Qwen3 架构，
尚未实测真实 0.6B 普通全参 GRPO。长上下文、多 worker 会增加内存，不能把下界当作实际峰值。

### 5. 理解采样与训练批量

| 参数 | 计量单位 | 使用含义 |
| --- | --- | --- |
| `--epochs` | 数据集轮次 | 每轮重新访问全部 train 沙箱 |
| `--rollout-group` | 每次沙箱访问的轨迹数 | GRPO 至少 2；同组比较奖励 |
| `--rollout-workers` | 进程数 | 每个 worker 有模型副本，增加会占用更多内存 |
| `--rollout-concurrency` | 每 worker 的同时活跃轨迹数 | 不等于 rollout-group，不大于它 |
| `--batch-size` | 沙箱组数 | 收集多少组作为训练 batch |
| `--mini-batch-size` | 沙箱组数 | 每个优化 mini-batch 包含多少完整组，不大于 batch-size |
| `--micro-batch-size` | assistant 片段数 | 物理前向/反传的最大序列条数 |
| `--max-tokens-per-micro-batch` | padding 后的输入 token 数 | 约束物理批量；单条过长时需要显式增大 |

GRPO 同组奖励全相同时没有可用的相对优势。默认有限重采样后跳过，日志会区分全成功、全失败和相同奖励。
因此“完成 epochs”不等于“执行同样数量的 optimizer step”；需同时查看 visited、skipped、updated 和评估结果。

### 6. 实时看板与日志

训练默认自动写 TensorBoard。在另一个终端运行：

```bash
./scripts/train_dashboard.sh \
  --logdir output/rl_runs/qwen06-qlora-thinking/tensorboard
```

浏览器打开 `http://127.0.0.1:6006`。

- Scalars：reward、policy entropy、loss、KL、梯度和资源曲线。
- Text：worker 的 `rollout/trajectory`，查看用户消息、工具调用、工具结果和最终奖励。
- 轨迹按动作更新，不是逐 token 流式展示；默认每组展示 1 条，`--rollout-trace-samples 4` 可展示 4 条。
- entropy 为当前策略在 assistant 训练位置的完整词表熵，仅用于监控，不额外增加 entropy bonus。
- JSONL 指标在训练中追加；对应 JSON 通常在退出时汇总。详细路径见[实时 TensorBoard 看板](#实时-tensorboard-看板)。

### 7. 中断恢复与产物

重跑原训练命令，保持原参数和原 `--output`，追加 `--resume` 即可恢复。
脚本不会仅凭 `--output` 自动补回所有训练参数；不要把省略参数后的默认值当作原配置。
允许增加 `--epochs` 延长训练；模型、沙箱、思考模式、训练范围及其他受保护配置必须一致。
源码和依赖身份也会校验，修改代码后不能直接续训旧运行。

| 路径（相对于运行目录） | 用途 |
| --- | --- |
| `config.json` | 本次参数记录 |
| `checkpoints/latest.json` | 指向最近完整训练检查点，包括模型、optimizer、RNG 等 |
| `snapshots/` | 向 worker 发布的策略版本；QAT packed companion 在 `inference/` |
| `training_report.json` | 完成情况、实际更新、评估、耗时和内存汇总 |
| `evaluation_*.json` | 训练前后及相应部署评估 |
| `metrics.jsonl` / `optimizer_metrics.jsonl` | 增量采样与优化指标 |
| `tensorboard/` / `logs/` | 看板事件与进程事件日志 |
| `group-*.json` | 保留窗口内的完整轨迹组 |
| `lora_adapter/` / `full_model/` / `qat_model/` | 按训练模式生成的部署模型，不替代训练检查点 |

默认 checkpoint 与策略发布间隔都是 1。可分别设置 `--checkpoint-interval-steps`、
`--policy-publish-interval-steps`；降低保存频率能减少 I/O，但崩溃后需要重放更多未提交工作。
检查点在安全边界提交，最后强制保存；未提交的未来版本恢复时隔离到 recovery。
训练正常完成后才生成最终部署导出；只恢复模型权重不能恢复完整训练状态。

### 8. 导出 Hugging Face / CUDA 可加载模型

```bash
uv sync --extra rl --extra cuda-export

# 全参 QAT；普通全参训练将模型路径改为对应运行的 full_model/
./scripts/convert_to_hf.sh \
  --model output/rl_runs/qwen06-full-qat-grpo/qat_model \
  --output output/hf-qwen06-qat --dtype bfloat16

# QLoRA：输出匹配的浮点 base_model/ 和 PEFT adapter/
./scripts/convert_to_hf.sh \
  --model models/Qwen3-0.6B-4bit \
  --adapter output/rl_runs/qwen06-qlora-thinking/lora_adapter \
  --format peft --output output/hf-qwen06-peft

# 在 NVIDIA 机器上验证加载与前向；使用具备相应 CUDA 支持的 PyTorch
python -m rl.verify_cuda --model output/hf-qwen06-peft --device cuda
```

转换输出必须为新目录。默认 BF16；数值对比可用 `--dtype float32`。
转换器输出标准浮点 HF/PEFT 文件，不输出 NF4/AWQ/GPTQ，也不迁移 MLX optimizer 状态。
本地已完成 PyTorch CPU 数值校验，尚未实测 NVIDIA GPU。更多格式、局部 QAT 和 PEFT 加载方式见
[转换到 CUDA 生态](#转换到-cuda-生态)。

### 9. 常见问题排查

| 现象 | 优先检查 |
| --- | --- |
| 沙箱资格或哈希校验失败 | 重新运行沙箱验收；确认未修改已验收产物 |
| 一直没有 optimizer step | 查看 worker 心跳、环境调用、零方差组和 KL 拒绝原因 |
| thinking 被截断或工具协议错误 | 查看原始轨迹、生成预算和模型工具协议提示 |
| 上下文超限 | 累计历史加上每次生成预算是否超过 max-context；默认 finish 会结束并评分 |
| 一条片段超过物理 token 预算 | 调整 max-tokens-per-micro-batch 或降低轨迹长度；不会静默截断训练数据 |
| 内存不足或 swap 增长 | 先降低 worker/concurrency、物理 batch 和长度；QAT 使用 packed inference |
| 批量提速没有体现 | 检查 batch_numeric_fallbacks；数值不一致会回退到 cached 路径 |
| resume configuration / identity changed | 使用原代码、依赖、模型和参数；新实验使用新输出目录 |
| reward 上升但实际能力未改善 | 检查 held-out 评估、奖励定义和完整轨迹，避免只看训练奖励 |

需要测量具体配置时运行 `python -m rl.profile_memory --help`；测量范围和实测结果见
[实测内存探针](#实测内存探针)。完整参数以 `./scripts/train_rl.sh --help` 和以下参考章节为准。

---

## 安装与启动

要求 Apple Silicon、Metal、Docker Engine、项目 Python 环境和合格沙箱；用户模拟器使用沙箱配置的外部模型服务。

```bash
uv sync --extra rl
uv run --extra rl hf download mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --local-dir models/Qwen3-4B-Instruct-2507-4bit

./scripts/train_rl.sh --help
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --output output/rl_runs/ppo-new --algorithm ppo
./scripts/train_rl.sh --tasks output/rl_multitask_manifest.json \
  --output output/rl_runs/grpo-new --algorithm grpo
```

只接受 `status.json` 和 `pipeline_result.json` 均为 `training_ready=true` 且产物哈希匹配的沙箱。
输出目录必须新建；继续已有运行使用 `--resume`。旧版本配置和代码无法通过新版本的严格续训身份检查。

## Rollout 消息与工具调用

沙箱 reset 后通过 `GET /v1/tools` 获取工具定义，随每次采样传给 tokenizer 的
`apply_chat_template(..., tools=tools)`。system 消息只包含任务执行规则；工具格式由模型自带的模板生成。
策略 tokenizer 必须具有 MLX-LM 可识别的原生工具调用格式和解析器。

模型生成的工具调用解析为 `assistant.tool_calls`，工具执行结果使用 `role="tool"`，
并用 `tool_call_id` 对应调用。默认 Qwen 模板将其编码成 `<tool_call>` / `<tool_response>`。
同一轮的多个工具调用按输出顺序执行，每个调用返回独立的 tool 消息。
普通 assistant 文本直接提交给用户模拟器，不再使用 `kind=tool/respond` JSON 外层协议。
初始 observation 和用户模拟器回复仍为 user 消息；无效调用格式返回协议错误，未知工具返回对应的 tool 错误。

单条采样、批量解码、并行环境 RPC、HTTP 环境与快照恢复共用这一协议。
轨迹保留原始采样 token 和行为概率，结构化消息只用于执行动作和构造后续上下文；工具结果不作为训练目标。
新运行的策略身份包含 `chat_protocol=native_tools_v1`，不导入旧 JSON 协议运行的 replay。

## 统一概率比与损失

每条轨迹保存生成时的策略版本、真实逐 token `old_logp`、终局奖励、终止标记；PPO 另外保存行为价值。本次 episode 评分后 bootstrap 为 0。
训练时计算 `ratio = exp(new_logp - old_logp)`，PPO 与 GRPO 共用：

```text
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-epsilon, 1+epsilon) * advantage))
```

- 奖励时机：reset 和中间 action 不读取奖励。episode 自然结束、到达步数上限或上下文预算上限后，由 RL rollout 框架调用 `finish()`，通过 `GET /v1/reward` 获取一次过程与结果综合评分；重复 finish 复用该分数。
- PPO：中间 action 奖励为 0，最后一个 action 的末 token 接收终局评分，利用保存的行为价值计算 GAE/returns；加 clipped value loss。步数和上下文上限均作为本次有限 episode 的评分终点，bootstrap 为 0。
- 训练 reset 使用 `reward_mode=episode_end`：用户模拟器不调用奖励判断结束；必要交互完成后的最终提交结束协议轨迹，正确性由终局奖励判定。旧沙箱运行时必须重建以支持该协议，不能静默退回逐步评分。生成验收仍可单独调用奖励接口。
- GRPO：同组轨迹共享任务和环境 seed，以最终奖励组内标准化得到优势；不训练 Critic。零奖励方差组跳过。
- 两种算法均加固定初始参考策略的 KL 正则。只有 assistant token 参与损失。
- 行为概率始终保留；同一个训练 batch 的 optimization pass 使用同一组优势和价值目标；下一个 dataset epoch 重新采样。Replay 保留完整原组，GRPO 不混合不同组计算优势。
- 默认按动作边界折扣：gamma=1.0、lambda=0.95，动作内 token 不折扣；显式 `--discount-unit token` 可用于对照。
- Actor 指策略训练侧；rollout worker 指采样进程，与 slime 口径一致。
- 日志与梯度使用相同的权重：每个沙箱等权，沙箱内各 rollout 等权，rollout 内 assistant token 均值。

## 有限复用与异步调度

Rollout worker 在组边界加载最新快照，组内行为版本一致。有界队列产生背压，Actor trainer 检查策略滞后。
默认不额外 replay；通过 `--replay-batches-per-batch` 显式增加复用。Replay 从已有奖励/工具观测中训练，不重新调用沙箱。
每个 replay batch 按当前数据集 batch 的沙箱身份取各自有效缓存组，不推进 epoch。任务 weight 不改变数据集每轮恰好一次的覆盖语义。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--rollout-workers` | 1 | 独立采样进程数 |
| `--rollout-concurrency` | 2 | 每个 rollout worker 同时推进的 rollout 数，不超过 rollout-group |
| `--watchdog-interval` | 1 秒 | 独立健康检查线程的检查间隔 |
| `--queue-size` | 2 | 采样队列容量 |
| `--rollout-group` | 4 | 每个沙箱每次访问采样的 rollout 轨迹数；GRPO 至少 2 |
| `--batch-size` | 1 | 每批按完成顺序收集多少个沙箱组；保留 epoch 尾批 |
| `--mini-batch-size` | 1 | 每个 optimizer step 使用多少个沙箱的完整 rollout group |
| `--epochs` | 2 | 训练沙箱数据集完整循环次数，每轮每个训练沙箱恰好访问一次 |
| `--optimization-passes` | 1 | 对已采样 batch 重复优化的次数，独立于数据集 epoch |
| `--max-policy-lag` | 1 | 新采样队列允许的版本落后 |
| `--replay-batches-per-batch` | 0 | 每个新采样 batch 后追加的 replay batch 数 |
| `--replay-capacity` | 32 | 最多保留的轨迹组数 |
| `--replay-max-uses` | 4 | 单组最多 replay 抽样尝试次数，拒绝也计数 |
| `--replay-max-age` | 32 | Replay 允许的最大策略版本年龄 |
| `--target-kl` | 0.05 | 更新前超限则拒绝该步；每步更新后重测，超限则停止该 batch 剩余 mini-batch 和 optimization pass（就绪调度下每步 KL 在当前 mini-batch 上计算） |
| `--replay-min-ess` | 0.01 | 相对有效样本量下限 |
| `--replay-max-log-ratio` | 60 | 最大绝对 log-ratio 上限 |

裁剪目标配合 KL、ESS、年龄和复用次数限制使用；不保证任意陈旧数据都适合训练。
`--max-groups` 和 `--updates` 是可选提前停止预算，默认 0 表示不额外限制；若预算不足以完成 dataset epochs，运行不会声称完成。
拒绝组按配置重采样；零方差默认有限重试后跳过并计入已访问沙箱，不计入有效更新。只有实际优化才发布策略版本。

### 长训练稳定性、性能与周期评估

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--train-progress-timeout` | 3600 秒 | 等待 rollout 时无新增可用结果的 deadline，错误包含等待任务与 worker 心跳；不计 actor 更新/评估时间 |
| `--group-timeout` | 3600 秒 | 单组处理 deadline，持续 token 心跳也不能无限延长；超时消耗 worker 重启预算 |
| `--model-load-timeout` | 600 秒 | 模型/快照加载阶段的心跳超时 |
| `--rollout-timeout` | 1200 秒 | 其他 worker 阶段心跳及单次环境 RPC 超时 |
| `--heartbeat-interval` | 1 秒 | 同阶段心跳写盘间隔，阶段变化立即写入；必须小于 rollout-timeout |
| `--zero-variance-policy` | `retry_skip` | `retry_skip` / `retry_fail` 有限重试后跳过/失败；`skip` / `fail` 立即跳过/失败 |
| `--context-limit-policy` | `finish` | `finish` 结束并真实评分；`fail` 保留超限报错 |
| `--numerical-check-mode` | `periodic` | `periodic` 首次物理形状全检并定期复检；`strict` 每次检查 |
| `--numerical-check-interval` | 20 | 同一已验证形状隔多少次 actor step 再检查；发现偏差后本运行及续训切回严格检查 |
| `--reference-cache-tokens` | 1000000 | 固定 reference 的 LRU 缓存预算，按 prompt + response token 数计；0 关闭 |
| `--group-artifacts-keep` | 256 | 保留最近完整 group / 周期评估结果；checkpoint、replay 和最佳评估引用额外受保护 |
| `--metrics-export-interval` | 0 | 0 仅退出导出完整 JSON；实时读取使用 JSONL 或 TensorBoard |
| `--eval-interval-steps` | 100 | 周期评估步数间隔，0 关闭；只选 manifest 中 split=eval 的任务，无 held-out 集时不启动 |

上下文超限不裁剪历史、不补造 token，`finish_reason=context_limit`。PPO 将最后已执行 action 作为有限 horizon 的评分终点，bootstrap=0。
如果初始 prompt 已超限，没有任何可训练 action，仍记录真实 reward；包含这种 episode 的组整组跳过，避免破坏 GRPO 组内比较。
零方差日志区分 `all_success`、`all_failure`、`equal_reward`；成功定义沿用 reward>=1 且自然终止。
TensorBoard 增加 `rollout/zero_reward_variance`、`rollout/context_limit_rate` 及 visited/skipped/updated 计数。

reference 缓存属于当前 trainer 的固定 reference 实例，键为精确 prompt/response token；策略更新不会使其失效。
缓存不进入 checkpoint，恢复后重新预热。数值校验模式不改写采样得到的行为 logprobs，不改变 PPO/GRPO loss。
`train/reference_cache_hits`、`train/reference_cache_misses` 可用于观察复用；阶段 timings 用于验证实际收益。

checkpoint 对两份指标 JSONL 保存记录数、字节位置与 SHA-256，提交前 fsync。
续训验证已提交前缀，未提交尾部移入 recovery 后从 checkpoint 位置继续；不会重复计算已提交的更新。
TensorBoard 继续使用 restart 事件隐藏未提交步。代码身份校验仍严格，不自动允许旧代码运行跨版本续训。
完整轨迹默认会被回收；需要保留全部训练轨迹时将 `--group-artifacts-keep` 设置为足够大的值。

周期评估在完整 batch 边界触发，因此实际间隔可大于配置值。它使用固定任务和 seed、greedy 解码，并恢复评估前 MLX RNG 状态。
曲线包括 `eval/periodic/reward_mean`、按 split/task 的 reward 和 success rate。
`best_policy.json` 指向已提交评估中的最佳策略权重，带校验和；分数相同不替换。
该文件用于加载最佳策略，不含 optimizer；继续训练使用 `checkpoints/latest.json`。
评估失败会明确使运行失败，可从最近训练 checkpoint 恢复，不伪造评估分数。

例如每 20 个 optimizer step 评估，并采用严格数值校验：

```bash
./scripts/train_rl.sh --tasks output/rl_multitask_manifest.json \
  --output output/rl_runs/grpo-monitored --tuning lora \
  --eval-interval-steps 20 --numerical-check-mode strict \
  --zero-variance-policy retry_skip --context-limit-policy finish
```

### 跨运行导入

```bash
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --output output/rl_runs/replay-new --algorithm ppo \
  --replay-source output/rl_runs/previous-run --replay-batches-per-batch 1
```

PPO 和 GRPO 均支持导入。要求来源基座/tokenizer、微调配置、温度和任务身份匹配，并有完整 schema-v2/v3 行为概率。
PPO 另外要求真实行为价值与 bootstrap。来源文件需保留，读取时检查校验和。
来源策略版本保持不变；跨运行版本不可直接比较，因此导入年龄从本地导入版本起计，概率漂移仍由实际行为概率检查。
导入需要显式设置 `--replay-batches-per-batch` 大于 0；不会将 replay 当成新的 dataset epoch。

## 多任务与独立评估

`--sandbox` 和 `--tasks` 互斥；清单内相对路径相对于清单目录解析。

```json
{"tasks":[
  {"id":"train-a","sandbox":"sandbox/a","split":"train","weight":1},
  {"id":"train-b","sandbox":"sandbox/b","split":"train","weight":1},
  {"id":"held-out","sandbox":"sandbox/c","split":"eval"}
]}
```

任务 ID 和沙箱内容身份必须唯一，eval 任务不会进入采样或 replay。
Rollout worker 在组边界切换任务，并关闭旧环境进程。每个并行 rollout 的模块、数据库和历史在独立进程中隔离。
评估按任务及 train/eval 分别汇总，默认 `--eval-episodes 3`；没有 eval 任务时标记 `independent_eval=false`。

## 训练参数范围与思考模式

| `--tuning` | 更新范围 | 底座要求 |
| --- | --- | --- |
| `qat` + `--qat-scope projections`（默认） | 指定层 q/v projection 的 FP32 master 权重 | 支持 MLX 量化底座 |
| `qat` + `--qat-scope full` | 全部语言模型参数；Linear/Embedding 使用 fake quantization，norm/bias 保持浮点 | 浮点或 MLX affine 4/8-bit 底座 |
| `lora` | 指定层的 LoRA adapter | 支持 MLX 量化底座 |
| `full` | 全部语言模型参数，包括 embedding、attention、MLP、norm | 完整浮点 safetensors 底座，不接受 GGUF 或 packed 量化权重 |

三种模式均支持 PPO / GRPO、策略同步、checkpoint 和断点续训。PPO 另外训练 critic。
`full` 不受 `--layers` / `--rank` / `--bits` 限制。运行结束后，`full_model/` 保存完整 FP32
`model*.safetensors`、模型配置和 tokenizer；其中不包含 PPO critic 或 optimizer。
继续训练使用运行目录的 checkpoint。`rl_generation.json` 记录思考模式和温度，外部推理程序需显式应用这些设置。

当前计算使用 FP32，未实现混合精度、optimizer 分片或 offload。启动前根据 safetensors header
检查内存：持久张量下界为 `参数数 × 4 × (5 + rollout_workers)` 字节，包含 actor、reference、
rollout worker、梯度和两个 Adam moments，尚不包含 activation、KV cache、critic 和临时缓冲。
Qwen3-4B 在一个 worker 下约需 96 GB 起，因此本机 48 GB 不适合该模型的全参训练。
全参 QAT 同样使用 FP32 master、梯度和 Adam 状态，未开启 packed inference 时内存下界也约为 96.5 GB，另有 fake quantization 临时缓冲。开启后的估算见下文优化配置。
检查通过也不代表所有上下文长度都能放入内存；本机 48 GB 可对 4B 使用 LoRA / 局部 QAT。
量化底座 + LoRA 会冻结底座；它与更新全部 master 权重的全参 QAT 是不同的训练配置。

原版 **Qwen/Qwen3-4B** 支持以下 `--thinking-mode`：

- `auto`（默认）：保留 tokenizer 默认行为。
- `thinking`：向 chat template 传入 `enable_thinking=True`。
- `no-thinking`：传入 `enable_thinking=False`，Qwen3 模板预填空的 think 段。

该设置统一用于训练侧、rollout worker 和评估，并进入续训及 replay 身份检查。
显式模式要求 tokenizer 模板支持 `enable_thinking`；不支持的模板会报错。
这是模板控制，不是 `reasoning_effort` 档位。思考 token 属于 assistant 生成，参与 logprob、entropy 和 loss，
也消耗 `--max-tokens` / `--max-context` 预算。Instruct-2507 与原版 Qwen3-4B 是不同模型，不能用前者替代思考模式验证。

```bash
# 原版 Qwen3-4B 的 MLX 4-bit 版本：可分别用 thinking / no-thinking
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --model models/Qwen3-4B-4bit --tuning lora --thinking-mode thinking \
  --max-tokens 2048 --max-context 4096 \
  --output output/rl_runs/qwen3-thinking

# 全参训练：需要完整浮点底座和足够内存
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --model /path/to/floating-model --tuning full \
  --output output/rl_runs/full-new
```

模型适配测试可单独运行（需要本机 Metal）：

```bash
.venv/bin/python -m pytest tests/rl/test_full_training.py -q
RL_TEST_QWEN3_MODEL=models/Qwen3-4B-4bit \
  RL_VALIDATION_OUTPUT=output/validation/qwen3-modes \
  .venv/bin/python -m pytest tests/rl/test_qwen3_modes_integration.py -q
```

第一组使用小型 Qwen3 架构检查全参更新、checkpoint 续训、导出回载及内存检查。
第二组用真实原版 4B 权重检查两种模式的工具调用、单条/批量解码、LoRA PPO/GRPO 更新及 QAT 导出。
训练测试使用受控奖励，不用于证明任务成功率提升；真实环境的学习效果需要独立评估集验证。

2026-10-02 本机验证：原版 `mlx-community/Qwen3-4B-4bit`（权重 revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`，使用官方 Qwen3-4B tokenizer 配置）
通过 17 项模型级用例。两种模式的单条/批量工具调用均完成，最大独立 cached logprob 误差约 `1.91e-5`；
四种 LoRA 模式/算法组合各完成 6 步更新与续训。QAT packed 回载最大 logprob 误差为 thinking `0.00187`、
no-thinking `0.00875`（测试门限 `0.01`）。全参更新仅在小型浮点 Qwen3 架构验证，未运行 4B 全参训练。
本地轨迹、指标和汇总位于 `output/validation/qwen3-4b-20261002/`。

### QAT 与数值一致性

默认 `--tuning qat`：最后 1 层 attention q/v projection 使用 FP32 master、4-bit affine fake quantization 和 STE。
其余基座量化权重冻结；可用 `--layers`、`--bits 8` 调整。实现为权重 QAT，不含 activation QAT。
另支持 `--tuning lora`。

`--tuning qat --qat-scope full` 将所有层的 Linear 和 Embedding 转成 FP32 master + affine fake quantization + STE，
包括 attention q/k/v/o、MLP gate/up/down、embedding 及独立 lm_head；共享输出头通过同一个 embedding master 计算。
norm 和普通 bias 保持浮点并参与训练，量化 scales/biases 从 master 重算。PPO critic 保持浮点。
`--layers` / `--rank` 不限制全参 QAT，`--bits 4` / `8` 设置全部 QAT 权重的目标位宽。
当前为 weight-only QAT，不含 activation QAT，不支持与 `--tuning lora` 同时选择。

```bash
# 需要足够内存；本机 48 GB 会在 Qwen3-4B 全参 QAT 加载前被检查拦截
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --model models/Qwen3-4B-4bit --tuning qat --qat-scope full --bits 4 \
  --thinking-mode thinking --max-tokens 2048 --max-context 4096 \
  --output output/rl_runs/qwen3-full-qat
```

全参 QAT 完成后导出 `qat_model/`：包含全部 packed 权重、训练后的浮点 norm/bias、配置与 tokenizer，
可直接由 `mlx_lm.load` 加载，不依赖原始底座。它不包含 critic/optimizer；续训使用运行目录 checkpoint，
其中保存全部 FP32 master。思考模式和温度记录在 `rl_generation.json`，外部推理需显式应用。
`qat_scope` 进入 rollout、评估、策略身份和严格续训检查，不能在同一运行中切换训练范围。

`tests/rl/test_full_qat.py` 使用小型 Qwen3 架构验证浮点/量化底座、共享/独立输出头、4/8-bit 导出回载、
全部参数解冻、真实参数更新和 PPO/GRPO 续训。48 GB 本机没有进行 4B 全参 QAT 实际训练。

采样使用逐 token KV cache，Actor 评分／训练采用右侧 padding 的完整序列张量批次，浮点参数使用 FP32。
真实采样 log-prob 与独立 cached 重算的误差门限仍为 0.001；行为概率始终保留。
批量路径的 kernel 数值差异单独测量，超过 `--batch-logp-tolerance`（默认 0.01）的形状自动拆小，单条仍超标时显式回退 cached 路径；回退次数和原始最大误差进入报告。
局部 QAT 导出仅含训练层，需要同一基座和 tokenizer；全参 QAT 导出完整 `qat_model/`。部署身份清单保存文件哈希。
验收使用全新模型实例加载 packed 层，核对概率误差并执行真实评估。

## 恢复、产物与性能

### 实时 TensorBoard 看板

`uv sync --extra rl` 会安装 TensorBoard 和进程监控依赖，训练默认写入事件文件。
启动训练后，在另一个终端运行：

```bash
./scripts/train_dashboard.sh --logdir output/rl_runs/grpo-new/tensorboard
# 浏览器打开 http://127.0.0.1:6006
```

不指定 `--logdir` 时展示 `output/rl_runs` 下所有带事件文件的运行；可用 `--port 6007` 换端口。
事件默认每 5 秒刷新到磁盘，看板每 5 秒重新读取，因此页面可能落后数秒。
事件写入使用 TensorBoard 自带的异步 writer，不依赖 PyTorch 或 TensorFlow 训练运行时。

| TensorBoard run | 面板与内容 | 横轴 |
| --- | --- | --- |
| `train` | loss、policy/value loss、policy entropy、KL、clip fraction、梯度范数、学习率、优化耗时和训练目标 token/s | optimizer step |
| `rollout_groups` | reward 均值/标准差/范围及直方图、成功/终止/截断率、长度、工具 HTTP 错误率、策略滞后和队列等待 | 主进程已消费的采样组数，含拒绝组 |
| `workers/worker-N-会话ID` | 实时轨迹文本、每轮生成量、工具/协议错误、动作延迟、最终奖励、worker 阶段和进程资源 | 当前进程会话事件序号 |
| `runtime/actor-会话ID` | 配置、checkpoint 进度、累计阶段耗时、接受/拒绝原因、错误栈、运行报告和资源 | 当前主进程会话事件序号 |

在 **Text** 面板选择 worker 下的 `rollout/trajectory`，可以看到 task、policy version、
rollout index、action step、用户消息、原生工具调用与工具结果。
轨迹在 reset、动作生成、环境返回、finish 时逐步写入；工具尚未返回时也可先查看已生成的调用。
工具返回后的反馈与最终 reward 分开记录，中间步骤不会触发额外奖励计算。
这是按动作更新的文本轨迹，不提供逐 token 流式聊天。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--tensorboard` / `--no-tensorboard` | 开启 | 控制 TensorBoard 事件；关闭后仍写 JSONL |
| `--log-flush-seconds` | 5 | 事件刷新及进程资源采样的最小间隔 |
| `--rollout-trace-samples` | 1 | 每个采样组展示前 N 条轨迹；0 关闭轨迹正文 |
| `--rollout-trace-max-chars` | 16000 | 每条轨迹展示的消息正文长度上限，超长时保留首尾 |

例如展示组内所有 4 条轨迹：

```bash
./scripts/train_rl.sh --sandbox output/sandbox/task-1 --output output/rl_runs/grpo-new \
  --rollout-group 4 --rollout-trace-samples 4 --log-flush-seconds 2
```

`train/policy_entropy` 是更新前当前策略在训练轨迹生成位置的全词表熵（nats），
使用与采样和训练打分一致的 temperature。先按 assistant loss mask 对每个片段取平均，
再按训练样本权重聚合；排除 prompt、padding 和被 mask 的 token。PPO、GRPO 及 cached fallback 均记录。
该指标只用于观测，不增加 entropy bonus 或改变优化目标；复用训练前向结果，但会增加全词表归约计算和临时内存。

训练曲线使用 `optimizer_step`，rollout 曲线使用采样组计数，二者不混用。
`--resume` 从 checkpoint 恢复这两类横轴，并用 TensorBoard restart 事件隐藏未提交步的旧曲线；
JSONL 保留这些历史事件用于排查。每次 worker 重启创建独立会话目录，避免把重试轨迹误认为新的训练进度。
日志选项可在续训时调整。原有代码、模型与任务身份校验仍生效。

`logs/**/events.jsonl` 逐条追加时间、进程、事件、指标和选中的轨迹正文，便于训练中查询。
`metrics.jsonl`、`optimizer_metrics.jsonl` 是实时增量指标源；对应 `.json` 在退出时导出，
也可用 `--metrics-export-interval N` 每 N 个 optimizer step 导出。看板轨迹可能截断或抽样，完整轨迹以尚在保留窗口内的 `group-*.json` 为准。
TensorBoard 启动器为 Text 面板保留最多 200 个采样点，可通过 `--samples_per_plugin text=1000` 调整。

成功率沿用评估定义：episode 已终止且 reward >= 1；自定义奖励范围时应主要看 reward 曲线。
采样 `tokens_per_wall_second` 计入整个组的环境等待；`tokens_per_compute_second` 使用累计解码计算耗时。
Metal 指标是各进程自身的 active/cache/peak，不代表整机总 GPU 内存；进程 RSS 也不应简单相加当作物理占用。
`eval/periodic` 是固定 held-out 集的周期评估，默认间隔 100 个 optimizer step，在 batch 边界执行；有 held-out 集时也在初始和最终状态评估。
`eval/before`、`eval/after` 与 QAT 的 `eval/deployed` 保留原有全任务最终对照评估。

- `checkpoints/latest.json` 指向原子提交的完整模型、Adam、RNG、计数器、rollout worker 位置及 replay 状态。已提交 mini-batch 不重复更新，未提交组重新采样。
- 续训要求代码、依赖、模型/tokenizer、任务及训练配置身份一致；可增加 epochs 继续遍历，也可调整可选 updates/max-groups 预算。
- `--max-rollout-restarts 2` 有限恢复可重试故障；持久恢复日志防止 resume 清零未提交的重启次数。
- `--checkpoint-keep 3` 保留完整检查点；训练中清理旧快照与已提交消费的 continuation/result，保护所有保留 checkpoint、replay、未完成任务及 worker 权重租约。
- `run_status.json` 记录 running/completed/failed/interrupted；训练异常不伪造零奖励。
- `optimizer_metrics.jsonl` 记录每个 optimizer step 的加权损失；`metrics.jsonl` 记录更新、rollout 时间区间、replay、拒绝和阶段耗时。对应 JSON 文件是导出视图。
- `training_report.json` 分别报告 visited/accepted/skipped/updated jobs、实际 optimizer steps、策略变化、独立回载和异步重叠。`completed` 表示计划遍历完成；全部跳过时 `training_updates_observed=false`，不表示学到了能力。执行更新却只有 Critic 变化仍报错。
- `peak_metal_gb` 只计 Actor trainer 进程，不能当作整机内存峰值。原始轨迹/诊断日志保留，长训练需自行归档。

```bash
RL_TEST_MODEL=models/Qwen3-4B-Instruct-2507-4bit \
  uv run --extra rl pytest tests/rl -q
uv run --with matplotlib python -m rl.plot_metrics --runs output/rl_runs/ppo-new
```

损失图优先读取逐 optimizer step 数据。短运行、满分基线或损失下降不构成能力提升证据。

## 验收材料

三个不同的合格沙箱清单位于 `output/rl_multitask_manifest.json`，包含两个训练任务和一个独立评估任务。
历史运行包含 PPO 多任务 replay、GRPO、rollout worker 故障恢复、断点续训及全新实例 QAT 回载证据。
历史报告保留其当时的参数与代码哈希；历史验收见下方记录；其中旧版 epochs 表示优化遍历，不代表当前的数据集 epoch。

### 当前统一入口验收

- `output/rl_runs/grpo_unified_replay_01`：从历史真实 GRPO 运行原样选取一组非零方差轨迹，
  组奖励 `[1,0,1,0.2]`，完成两次 replay 更新、4 个 optimizer step；行为 KL 均低于门限，QAT 独立回载奖励 1.0。
  原组来源与选取原因记录在 `grpo_replay_validation_source/selection.json`，此次用于复用功能验证，不估计成功率。
- `output/rl_runs/ppo_unified_replay_01`：两个训练任务各一次新采样和一次 replay 更新，4 次策略更新、8 个 optimizer step；
  独立评估任务不进入训练，三个任务奖励均为 1.0，QAT 全新实例回载概率误差 0。
- 两次运行均检测到策略权重与有效量化权重变化。运行后仅增加了 CLI 禁止参数缩写的防误解析修正，并单独回归。
- CLI 只接受完整参数名，已删除的旧参数不会被缩写匹配到其他参数。

## Rollout 并行与健康检查

每个 rollout worker 持有一份策略模型，环境进程池并行执行独立 episode 的 reset/step/finish。
一个 episode 等待工具或用户模拟器时，rollout worker 可为另一个 episode 生成动作。
同一组始终使用同一策略版本和环境 seed，结果按 episode 槽位顺序收集，保留真实行为概率。
组内模型调用由 rollout worker 主线程调度，避免跨线程共享 MLX RNG。
默认使用动态 GPU 批量解码：每轮将所有已就绪轨迹合为一次模型前向，各生成一个 token。
轨迹保留独立 KV 历史；不同长度缓存通过带左 padding 的批量缓存合并。
动作到达 EOS 或 token 上限后退出批次，进入环境 RPC；工具返回后可重新加入。
prompt prefill 当前仍逐请求执行，批量发生在后续 token 解码；不跨 rollout worker 合并请求。
每个 worker 的解码批次最多为 `min(rollout_concurrency, rollout_group)`，无需新增开关。
PPO 的 `old_values` 和所有算法的 `old_logp` 直接来自实际采样前向，标记
`behavior_statistics_source=batched_sampling_forward`；不使用事后单条重算结果覆盖真实行为统计。
原单条 `sample/iter_sample` 路径仍保留缓存重算校验，训练侧的数值检查与漂移门禁保持不变。
`decode_batch_sizes` 记录每个生成 token 所处的批量大小，可判断环境等待是否导致实际退化到单条。
批量前向耗时按当轮参与轨迹均分计入 `generation_compute_seconds`，求和不重复计算共享前向。
增加 `--rollout-workers` 可增加独立采样模型进程，同时会增加 GPU/内存竞争。

本地生成基准：`.venv/bin/python scripts/benchmark_rl_decode.py --samples 4 --tokens 32`。
比较同一引擎依次处理单条请求与同时处理多条请求，包含 prefill、解码和行为统计，
不包含外部环境、训练或原单条路径的事后重算校验。

独立 watchdog 按固定时间检查所有 rollout worker，即使队列满或 Actor trainer 正在计算也会检查和恢复。
不可恢复错误传递给 Actor trainer，在下一个安全检查点抛出；原生 GPU 调用不会被异步强制中断。
停止信号使用单写者无锁共享标记，策略版本在完整快照写入后用原子文件发布，避免 Rollout worker 被终止时留下共享锁。
环境子进程监视 rollout worker 存活，Rollout worker 被强制终止后会退出，防止继续发送孤立模拟器请求。

`post_behavior_kl` 记录每个实际更新后的行为 KL；`post_kl_exceeded` 为 true 时停止后续 epoch。
当前不回滚刚执行的更新，因此 target-kl 仍是提前停止阈值，不是更新后始终成立的硬上限。
`importance_clip_fraction` 已与 PPO/GRPO 的 `[1-epsilon,1+epsilon]` 裁剪区间一致。
报告中的 `environment_rpc_overlap_observed` 来自不同 episode 的真实 step 执行区间重叠。

### 更新后 KL 验收

`output/rl_runs/parallel_ppo_kl_01` 使用两个 rollout worker、每个 rollout worker 两个环境槽位，使用当时旧参数的 3 次优化遍历、target-kl=1e-6。
第一次更新后实测行为 KL 约 4.56e-5，`post_kl_exceeded=true`，实际只执行 1 个 optimizer step，后续优化遍历停止。
QAT 全新模型回载概率误差 0、奖励 1.0。该运行发生于逐 token 调度加入之前，未观测到环境请求重叠，不作为并行效果证据。

### 并行 Rollout 实跑验收

`output/rl_runs/parallel_grpo_02` 使用 1 个 rollout worker、4 个环境槽位、组大小 4，完成 2 次 GRPO 策略更新、4 个 optimizer step。
第一组奖励为 `[0.2,1,1,1]`，第一次使用新采样，第二次复用同一组的真实行为概率；行为概率重算最大误差为 0。
`environment_rpc_overlap_observed=true`，证明不同 episode 的环境 step 实际重叠；`async_overlap_observed=true`，证明采样与策略训练更新重叠。
策略权重和有效 QAT 权重均发生变化；全新模型实例加载量化导出后，最大概率误差约 3.81e-6，评估奖励 1.0。
总耗时约 486 秒，期间环境进程存在等待外部用户模拟器 HTTPS 响应的长尾；并发没有消除外部服务延迟。
训练前后评估奖励均为 1.0，此次验证并行执行与训练链路正确运行，不代表能力提升。退出后 Actor trainer、rollout worker 和四个环境子进程均已结束。

此前 RL 测试（启用本地模型集成测试）共 56 项通过，覆盖独立环境并发、交错 KV cache、满队列时 watchdog 恢复和裁剪比例边界。

## 数据集 Epoch、Batch、Mini-batch 与 Rollout Group

四个维度独立定义：

- `epochs`：完整遍历训练沙箱数据集多少次，每轮无放回打乱顺序，eval 沙箱不参与。
- `batch-size`：从当前 epoch 按完成顺序收集多少个完整沙箱组。
- `mini-batch-size`：每个 optimizer step 使用多少个沙箱；每个沙箱的全部 rollout 一起进入该 step。
- `rollout-group`：每个沙箱每次访问采样多少条 rollout。同组 task、seed、行为策略版本一致。

例如数据集有 10 个训练沙箱：

```bash
./scripts/train_rl.sh --tasks output/rl_multitask_manifest.json \
  --output output/rl_runs/grpo-batched --algorithm grpo \
  --epochs 2 --batch-size 4 --mini-batch-size 2 --rollout-group 8 \
  --optimization-passes 1 --rollout-concurrency 4
```

上例需要清单实际包含 10 个训练沙箱才能得到以下数量：每个 epoch 的 batch 是 4、4、2 个沙箱，
分别执行 2、2、1 次 optimizer step，总计每轮 5 步、两轮 10 步，采样 160 条新 rollout。
完整 mini-batch 包含 2×8=16 条轨迹。遇到不足量的 batch/mini-batch 保留尾批，按实际沙箱数归一化。
KL 超限、零方差组或零梯度可能减少实际更新步数，报告保留跳过记录。

`optimization-passes` 才表示对已采样 batch 重复优化，默认 1。需要额外重复利用旧数据时显式配置
该参数或 `replay-batches-per-batch`；它们均不增加 dataset epoch 的覆盖次数。
GRPO 优势始终在每个沙箱原始 rollout group 内计算；PPO GAE 按单条 rollout 计算。
物理前向使用右侧 padding 的多 action 张量批次；`--micro-batch-size` 控制每次最多多少段 action，
`--max-tokens-per-micro-batch` 控制包含 padding 的输入 token 数。超出单个物理批次预算时继续梯度累积，逻辑沙箱 mini-batch 不拆散到不同 optimizer step。

Rollout worker 从共享任务队列领取当前 epoch 的沙箱任务，空闲 worker 不受固定分片限制。
训练侧从全局完成队列按就绪顺序组成 mini-batch 和 batch，慢沙箱可在后续 batch 消费。
仍保留 epoch 边界，确保每轮完整覆盖；不会为提高吞吐永久跳过慢任务。
未提交消费的完整候选结果和在途任务数由 `over_sampling_batch_size` 限制，默认是 `2 × batch_size`。
检查点记录 consumed_jobs、inflight_batch、轨迹校验和及 Adam 状态；每个 mini-batch 后提交，
恢复按实际消费 ID 去重。策略快照在训练 batch 边界发布，每个完整 rollout group 固定行为版本，
其未完成续跑所需版本在快照清理时受保护。原有版本滞后和概率漂移门禁继续生效。

共享队列位于 `output/task_queue/`，使用本机 POSIX 文件锁；进程退出释放任务租约。
完整结果先落盘，通知队列满时 worker 仍可继续，训练侧可从持久化结果恢复。
`continuation-N/` 保存策略版本、组进度及各 episode 的环境快照与 RPC 回复。
默认 SandboxEpisode 保存 SQLite、对话、奖励、终止状态和 trace；自定义 runtime 需要实现 JSON 可序列化的
`snapshot()` 与 `restore(state)`，不支持时明确报错，不会悄悄 reset 后冒充续跑。
恢复复用已保存的动作和环境回复；生成到一半、尚未交给环境的动作从动作边界重新采样，
不保存 GPU KV cache，也不承诺中断前后的随机采样逐 token 相同。
已完成并保存的环境动作不会因丢失回复被再次执行；若外部请求已发生但环境快照尚未保存，
恢复仍可能重新请求外部服务，外部副作用的恰好一次语义需要服务端幂等支持。

`optimizer_metrics.jsonl` 及其 JSON 导出分别记录 epoch、dataset_batch、optimization_pass、mini_batch、
mini_batch_sandboxes 和 mini_batch_rollouts。`metrics.jsonl` 记录每个 batch 的沙箱身份与 replay 来源，
以及 dataset_batch_completed 事件；training_report.json 汇总 epochs_completed、fresh_groups 等指标。

PPO 使用共享语言模型骨干加可训练 critic head，优化 clipped value loss；参考策略不创建 critic。
GRPO 的 rollout worker、actor trainer、参考策略均不创建 critic，也不执行 value head，不写入虚构 old_values。
旧版 `--group-size` 更名为 `--rollout-group`；旧版 replay 参数更名为 `--replay-batches-per-batch`。
旧配置和源码版本不可直接 resume，历史 replay 仍需满足任务身份、行为概率与 rollout 数对齐检查。

此前数据集语义回归 71 项通过，包含数据集每轮完整覆盖、尾批、乱序结果归位、rollout worker 分片恢复及 replay 沙箱筛选。
本地真实模型的 PPO/GRPO 各验证 2 个训练沙箱、2 个 epoch、每个沙箱 2 条 rollout、mini-batch 1 个沙箱，
正常执行 4 个 optimizer step；增加至第 3 个 epoch 续训后累计 6 步，无重复覆盖已提交轮次。
此测试使用受控本地奖励并隔离外部沙箱/API，用于验证调度、critic 和优化器恢复，不估计任务成功率。


## P0 / P1：性能指标、物理批量与统一接口

```bash
./scripts/train_rl.sh --tasks output/rl_multitask_manifest.json \
  --output output/rl_runs/batched-ready --algorithm grpo \
  --rollout-workers 2 --rollout-concurrency 4 \
  --epochs 2 --batch-size 4 --mini-batch-size 2 --rollout-group 4 \
  --micro-batch-size 4 --max-tokens-per-micro-batch 8192
```

`actor` 只表示策略训练侧，`src/rl/actor.py` 实现 ActorTrainer；采样位于 `src/rl/rollout_worker.py`。
旧 `--actors`、`--actor-timeout`、`--max-actor-restarts` 分别替换为 `--rollout-workers`、
`--rollout-timeout`、`--max-rollout-restarts`；本地仍是单个 Actor trainer，不虚构多卡 actor 训练能力。

`timings.json` 和 training_report 的 timings 区分：

- rollout 生成实际计算、行为概率校验、工具调用、用户模拟器、奖励计算和队列驻留；
- actor 等待就绪 mini-batch、策略／参考评分、数值交叉检查、前反向、optimizer step、更新后 KL；
- 权重发布、worker 权重加载、轨迹落盘和检查点。

并发 rollout 的累计值是 worker-seconds，不能相加当作运行墙钟时间；prepare/post-KL 等外层阶段包含部分子阶段，
也不能把所有指标相加。前反向指标包含完整梯度计算的同步等待，不假称独立 GPU kernel 时间。

物理批量在 response 预测位置投影词表，prompt、padding 和 loss_mask=0 的位置不参与目标。
每个逻辑 mini-batch 仍按沙箱等权、rollout 等权、有效 assistant token 均值归一化。
`batch_scoring_max_logp_error` 是交叉检查观察到的原始最大值；超过阈值触发拆小或回退，不覆盖真实采样概率。
回退是数值兼容路径，报告中的 batch_numeric_fallbacks 用于判断哪些运行没有获得全部批量加速。

schema-v3 轨迹使用 `sandbox_id → group_id → rollout_id → segment_id`，保留原始 token、行为版本及 loss_mask。
`TrainingSegment` 提供后端无关的段级契约。`--environment-factory module:callable` 可替换 AgentRuntime，
必须实现 reset/step/finish/close，并提供 messages、terminated。默认继续使用合格 AutoEnvAgentRL 沙箱。
该接口为其他 Agent runtime 和轨迹分段提供接入点，不表示已经实现子 Agent 调度或上下文压缩算法。

本地 kernel 对比命令（不调用外部 API，不测沙箱端到端效果）：

```bash
.venv/bin/python scripts/benchmark_rl_batches.py \
  --output output/rl_runs/p01_batch_benchmark.json
```

### 本轮验证与性能边界

本轮 RL 回归 79 项通过，包含真实本地模型的 PPO/GRPO 更新、critic 差异、
mini-batch 提交后故障恢复，以及未提交新权重发布后的快照保留。
更新后 KL 的检查范围是本次就绪 mini-batch，指标标记 `post_kl_scope=ready_mini_batch`。

`output/rl_runs/p01_batch_benchmark_short.json` 使用本地量化 Qwen、2 段各 16 token，
预热后取 3 次中位数。原始批量评分和前反向内核分别约快 5.59 倍、5.75 倍；
但最大 log-prob 偏差约 0.018，超过默认 0.01 的批量数值门限，触发 cached 回退。
包含数值保护的同范围训练阶段从 1.047 秒变为 1.080 秒（0.97 倍），尚未证明默认路径提速。
该基准不执行 optimizer 更新或外部环境交互，不能代替端到端吞吐测量。
后续性能工作应先解决量化模型批量与缓存路径的数值一致性；本轮没有放宽门限来制造加速结果。

### 动态批量生成验证

加入动态批量解码后，RL 回归 83 项通过（本地模型集成测试已启用）。覆盖不同长度 KV 缓存、
中途加入与槽位复用、EOS/长度结束、实际前向 log-prob/value 对齐、环境并发，
以及批量采样数据进入 PPO/GRPO 更新和检查点恢复。测试未调用外部 API。

`output/rl_runs/batch_decode_benchmark.json`：本地 Qwen3-4B 4bit、QAT，4 条轨迹各 32 token，
预热后 3 次中位数，单条依次解码 1.055 秒，动态批量 0.387 秒，约 2.73 倍。
批量路径以 32 次解码前向生成 128 token；此结果不代表含环境等待和训练的端到端加速。
这是生成侧基准，与前述训练侧数值回退导致的性能限制是不同测量范围。

### 共享队列与动作续跑验证

本轮 RL 回归 90 项通过，启用了本地 Metal 模型测试。新增验证覆盖：

- 慢任务仍持有租约时，其他 worker 领取后续任务，完成结果跨原始 batch 边界被消费；
- worker 被终止后租约释放，任务重新领取；完整结果不依赖内存通知队列存活；
- 消费 ID 恢复去重、队列容量和 epoch 覆盖边界；
- 环境 step 回复丢失后的续跑，不重复执行已保存动作；SQLite 与对话状态恢复；
- 未完成组所需旧策略快照保留；真实双 worker、3 个任务的持久化采样；
- PPO/GRPO mini-batch 更新、critic、批量解码及学习器故障恢复的既有回归。

测试使用受控本地环境，没有调用外部用户模拟器，也没有测量生产沙箱端到端吞吐。


## 超采样与自动补采

默认启用。`--over-sampling-batch-size 0` 表示候选池上限为 `2 × batch_size`；显式设置时不得小于
`batch_size`。候选数受当前 epoch 剩余沙箱数限制，不会为了填满候选池跨越 epoch 边界。
`--rollout-max-attempts 3` 表示每个沙箱本轮最多尝试 3 次（含首次）。

```bash
./scripts/train_rl.sh --tasks output/rl_multitask_manifest.json \
  --output output/rl_runs/oversampled --algorithm grpo \
  --rollout-workers 2 --rollout-concurrency 4 --rollout-group 4 \
  --batch-size 4 --mini-batch-size 2 --over-sampling-batch-size 8 \
  --rollout-max-attempts 3
```

仅通过 prepare 门禁的完整组进入优化；显式跳过仍占本轮访问配额。组内奖励零方差（GRPO）、
策略版本过旧或概率漂移不合格会触发原沙箱补采；新尝试使用新 seed、独立续跑目录，并在开始时加载已发布策略。
基础任务 ID 不变，`rollout_attempt` 从 0 递增。旧尝试的迟到通知不参与新尝试消费。
保留每轮任务访问覆盖，不用其他任务冒充原任务。零方差默认重试耗尽后明确跳过，可用 `--zero-variance-policy retry_fail` 保留严格失败行为；版本过旧或概率漂移重试耗尽仍失败。
跳过减少本批有效组数，按实际样本权重归一化；不会补造样本或把跳过计作梯度更新。`max_groups` 非零时限制总候选消费量。

worker 持续补充候选池。多余结果留给后续 batch，再按当时策略检查资格；不会为结束当前 batch 取消其他组。
只有合格组进入 replay。合格但尚未凑够 mini-batch 的记录，以及补采次数，会随检查点保存，恢复后补齐再更新。
`target_kl`、梯度为零等优化阶段的跳过规则仍生效；合格配额不代表保证执行梯度更新。

`training_report.json` 的 `sampling_attempts` 是已消费候选组数，`accepted_groups` 是合格沙箱访问数；
`over_sampling_batch_size` 是生效的候选池上限。拒绝指标记录 dataset_index、rollout_attempt 和原因。
这套超采样保留 epoch 覆盖约束，与允许任意快任务替代慢任务的纯吞吐模式不同。

本轮 RL 回归 94 项通过，包含本地 Metal 模型测试：补采使用新 seed、旧结果过滤、
拒绝组不占合格配额、尝试上限耗尽，以及半个合格 mini-batch 落盘后中断恢复。
受控案例采集 3 个候选组，接受 2 个组，恢复后仅执行 1 次完整 mini-batch 更新。

## 单机 Docker Engine + 沙箱管理器（默认）

`train_rl.sh` 和 `python -m rl.train` 默认使用 Docker。训练开始前自动启动管理器，
容器预热与模型加载重叠；不依赖 Kubernetes，也无需上传镜像到仓库。
库级 `Config` 默认仍是 `local`，方便直接运行已有本地测试和自定义 runtime。

```bash
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --output output/train-docker --sandbox-max-active 4 --sandbox-startup-workers 2

# 调试时明确使用原有进程内环境
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --output output/train-local --sandbox-backend local
```

默认从沙箱 `docker_image_metadata.json` 读取合格镜像 ID，镜像需已存在于本机 Docker。
也可用 `--sandbox-images images.json` 指定 `{ "task-1": "本地镜像名或ID" }`；
多个任务的键对应任务清单 ID。启动前将镜像名解析成不可变 image ID；
容器内仍验证任务代码和数据哈希。缺少镜像会报告失败，不静默换镜像或绕过资格检查。

### 调度和回收

- 每个任务复用一个容器；每条轨迹的会话和数据库独立。
- 默认最多 4 个启动中/活跃容器，最多 2 个启动操作并发。
- worker 只领取已就绪的 Docker 任务，未就绪任务留下启动请求，其他任务先采样。
- 每组 rollout 完成后释放容器租约；有活跃租约的容器不会被空闲/容量回收。
- 默认空闲 300 秒回收；容量满且有其他任务等待时可提前回收未使用容器。
- 默认每容器 1 CPU、1 GiB 内存、128 PID；可用 `--sandbox-cpus`、`--sandbox-memory` 调整。
- 仅绑定 `127.0.0.1` 随机端口，Agent 使用 HTTP 接口，训练进程不导入沙箱 app.py。
- 自动管理模式在训练正常结束或 Python 异常退出时回收本次容器；不删除镜像。

自动管理日志位于 `<训练输出>/docker_services/`：
`services.json` 为实时状态，`manager.jsonl` 为生命周期事件，`*.container.log` 为回收时保存的容器日志。
启动失败可查看 manager.jsonl；健康检查超时查看对应容器日志。
不自动重试有副作用的 HTTP 请求；已提交状态通过既有 continuation 和远程快照恢复。

### 独立预热与复用

如果希望训练命令启动之前预热，并在多次训练之间复用容器：

```bash
# 设置 SANDBOX_TRAINER_API_KEY；独立服务与训练必须使用同一个值。
./scripts/sandbox_services.sh start --sandbox output/sandbox/task-1 \
  --output output/services --max-active 4 --startup-workers 2
./scripts/sandbox_services.sh status --output output/services

./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --sandbox-services output/services/services.json --output output/train-reuse

./scripts/sandbox_services.sh stop --output output/services
```

`start` 启动后台管理进程并完成配置交接（生成 services.json）后返回，不等待容器启动；
返回后可立即将 services.json 交给训练，实际就绪/失败查看状态和 manager.log。
显式传入 `--sandbox-services` 时训练不取得服务所有权，训练结束不自动关闭这些共享服务。
独立 `stop` 通知管理器退出；如果管理器被强制杀死，重新执行 `stop` 会按持久化的所有者标签
回收遗留容器，不影响其他管理器或用户容器。异常退出后先 stop 再重新 start。
当前方案针对单机本地 Docker daemon，多个训练进程使用不同管理输出目录。

## Kubernetes 管理 Docker 沙箱，集群外 HTTP rollout

先构建并发布合格沙箱的 Docker 镜像，再异步提交 Kubernetes Deployment 与 NodePort Service。
训练进程留在本机/集群外，通过节点可达地址访问沙箱 HTTP 服务。
Kubernetes 直接运行 Docker/OCI 镜像，不需要在 Pod 内启动 Docker daemon 或挂载 docker.sock。

沙箱服务使用镜像原有 `app.create_app`，通过平台 HTTP 包装器提供原有 `/v1/*` 接口。
每条 rollout 分配独立会话、应用实例与 SQLite 数据库；不同轨迹的 reset、业务写入和奖励互不影响。
提交前验证本地训练资格，容器启动时校验绑定的业务代码与数据文件哈希（不要求镜像包含构建后生成的验收报告），训练端验证服务返回的身份；不匹配则拒绝 rollout。
镜像应包含原沙箱目录文件（`/app`），使用构建产物的 Dockerfile，且保持合格文件不变。
额外 HTTP 包装器由 ConfigMap 挂载，不修改镜像中的任务实现。

### 1. 准备镜像和集群配置

`images.json` 将训练任务清单的 ID 映射为已发布、集群可拉取的不可变镜像引用：

```json
{"task-1": "registry.example.com/sandbox/task-1@sha256:<64位摘要>"}
```

单任务 `--sandbox` 使用 ID `task-1`。多个任务使用已有 `--tasks tasks.json`。
指定 namespace 须已存在。`--secret` 引用已有 Kubernetes Secret，包含
`SANDBOX_TRAINER_API_KEY` 和沙箱所需的 `SANDBOX_LLM_*` 等配置。
训练进程设置相同的 `SANDBOX_TRAINER_API_KEY`；远程模式不会随机替换此密钥。
凭据不写入镜像、资源清单或 services.json。节点地址和 NodePort 必须从训练机器可达；
当前入口提供 HTTP NodePort，适合已有网络隔离的集群网络，TLS 可由现有网关承接。

### 2. 非阻塞启动

```bash
# 只生成资源清单，不访问集群
.venv/bin/python -m rl.kubernetes render \
  --sandbox output/sandbox/task-1 --images images.json --output output/services

# 提交后即返回，不等待调度、镜像拉取和 readiness
.venv/bin/python -m rl.kubernetes start \
  --sandbox output/sandbox/task-1 --images images.json \
  --namespace training --secret sandbox-env --prefix training-a \
  --host 10.0.0.10 --output output/services
```

`--context` 可选择 kubectl context；默认当前 context。
`--host` 是可从训练机器访问的集群节点 IP/DNS，不是 Service ClusterIP。
NodePort 由 Kubernetes 分配。输出 `resources.json` 和 `services.json`，后者记录完整可达 URL。
不同独立部署使用不同 prefix 和 output。Deployment 每个任务一副本，通过会话隔离并发轨迹；
不要直接增加副本数，否则当前进程内会话可能被路由到不同 Pod。

### 3. 启动训练

```bash
./scripts/train_rl.sh --sandbox output/sandbox/task-1 \
  --sandbox-services output/services/services.json \
  --output output/train-remote --rollout-concurrency 2
```

指定 `--sandbox-services` 自动选择 HTTP runtime，训练进程不再导入沙箱 app.py。
既有 PPO/GRPO、rollout_group、异步 worker、动作级断点恢复保持适用。
服务未就绪时只在使用该服务的环境进程中等待，默认最多 600 秒；单请求默认 120 秒，
可在 services.json 中设置 `startup_timeout` / `request_timeout`。
启动提交不等待所有服务就绪；训练现有的前置基线评估仍需要完成，并非“任何沙箱没就绪也保证开始更新”。

快照通过鉴权的 `/_rl/snapshot` 与 `/_rl/restore` 传输，保存到现有训练 continuation 中；
Pod 重建后可恢复已提交的 SQLite 状态和对话。传输失败不伪造奖励，不自动重试业务写入。
正常关闭回收会话；崩溃遗留会话在闲置一小时后由后续请求清理，服务最多容纳 128 个会话。
这些资源独立于训练生命周期，训练结束不自动删除，便于继续训练或检查日志。

```bash
.venv/bin/python -m rl.kubernetes stop --output output/services
```

停止只删除记录在该输出目录的资源；不删除 namespace 或共享 Secret。
Kubernetes 是可选远程后端；日常单机默认使用上面的 Docker 管理器。

## 第一、第二阶段优化配置

以下选项适用于 MLX PPO/GRPO；默认保持原有训练范围和提交频率。量化底座配合
`--tuning lora` 即本项目的 QLoRA 路径，使用 MLX affine 量化，并非 bitsandbytes NF4。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--lora-targets` | `self_attn.q_proj,self_attn.v_proj` | 支持逗号分隔模块名、`attention`、`all-linear`（attention + MLP） |
| `--layers` / `--rank` / `--lora-scale` | `1` / `8` / `16` | LoRA 最后 N 层、秩和缩放；MLX scale 对应 PEFT `alpha/r` |
| `--lora-dropout` | `0` | RL 要求为零；非零会破坏采样和重评分的一致性，启动时拒绝 |
| `--gradient-checkpointing` | 关闭 | 训练时重算 transformer block，降低 activation 占用；增加计算量 |
| `--logits-chunk-size` | `128` | 分块投影 response logits、logprob 和完整词表 entropy，并在反传重算投影 |
| `--prefill-chunk-size` | `512` | rollout 每个调度 tick 推进一个 prefill chunk，让其他请求继续推进 |
| `--prefix-cache-tokens` | `0` | reference/worker 的精确前缀 LRU token 预算；策略加载时清空 |
| `--packed-inference` | 关闭 | QAT reference/worker 使用 packed 权重，不保留全量 master |
| `--profile-memory` | 关闭 | 按阶段同步并记录 Metal peak/active bytes，有测量开销 |
| `--checkpoint-interval-steps` | `1` | durable checkpoint 的最小 optimizer step 间隔 |
| `--policy-publish-interval-steps` | `1` | worker 新策略版本的最小 optimizer step 间隔 |
| `--eval-interval-steps` | `100` | 已有独立评估间隔，与上述两种间隔分别控制 |

前缀缓存仅用于推理，每个请求使用独立 KV 容器。预算按缓存条目持有的 token 数计算，
不是按独立文本 token 去重；仍需给 KV cache 留出内存。训练 actor 不复用缓存，确保梯度正确。
chunking 保持完整 softmax 分布，不增加 top-k/top-p 截断。

```bash
# 全 28 层 QLoRA；替换 sandbox 路径为已通过验收的沙箱
./scripts/train_rl.sh --sandbox output/sandbox/task-1 --output output/rl-qwen06 \
  --model models/Qwen3-0.6B-4bit --tuning lora --layers 28 \
  --lora-targets all-linear --thinking-mode no-thinking \
  --gradient-checkpointing --logits-chunk-size 32 \
  --prefill-chunk-size 256 --prefix-cache-tokens 4096 \
  --checkpoint-interval-steps 10 --policy-publish-interval-steps 2 \
  --profile-memory --lora-merge-export

# 全参 QAT 可改用以下训练选项：
# --tuning qat --qat-scope full --packed-inference --gradient-checkpointing
```

提交和发布在安全的 batch 边界检查间隔，最后强制执行；实际间隔可能超过配置步数。
增大 checkpoint 间隔会增加崩溃后的重放工作量。恢复时回到最后完整 checkpoint，
恢复当时发布版本，不把未发布 actor 权重冒充旧版本；未来版本的结果、continuation、
master/packed snapshot 移入 recovery 隔离目录。未跨过 durable commit 的更新不保证保留。
packed companion 位于 `snapshots/inference/`，与 master 使用相同版本和保留生命周期。

### 导出与部署验证

LoRA 训练完成自动导出 `lora_adapter/`，可通过
`mlx_lm.load(base, adapter_path=... )` 回载；底座身份、实际 targets、rank/scale 均写入配置。
`--lora-merge-export` 额外产生 `lora_merged/`；再加 `--lora-requantize-export` 时改为
`lora_requantized/`。后者重新量化会引入误差，不能假定奖励保持不变。
`lora_export_validation.json` 记录独立回载的固定 token logits 误差和 KL，native adapter
误差超过阈值会失败；合并/再量化误差用于审查，不替代独立任务评估。
原生 adapter 保留底座量化语义；合并为 dense 后可能因计算内核不同出现小幅数值偏差。

全参 QAT 的 actor 仍保留 FP32 master、梯度和 Adam moments；`--packed-inference`
只节省 reference/worker 常驻权重。0.6B、一个 worker、4-bit/group64 的持久张量下界
约 10.28 GB（不含 activation、KV 和临时缓冲）；4B 即使开启该优化也超过 48 GB。

### 实测内存探针

```bash
.venv/bin/python -m rl.profile_memory --model models/Qwen3-0.6B-4bit \
  --output output/profile-qwen06.json --tuning lora --layers 28 \
  --lora-targets all-linear --gradient-checkpointing --logits-chunk-size 8 \
  --prompt-tokens 128 --response-tokens 32 --sequences 1
```

该命令执行真实 prefill、decode、reference scoring、反传及一次 Adam 更新，记录各阶段
耗时与 Metal 内存；使用合成 token，不证明任务学习效果，也不包含独立 rollout worker 进程。
比较配置时应固定长度、序列数、训练范围，并分别运行新进程。训练日志也记录 swap 用量，
防止把交换内存误认为物理内存容量。启用阶段测量会同步 GPU，不能用其吞吐直接代表无测量运行。

## 转换到 CUDA 生态

入口是 `scripts/convert_to_hf.sh`（或 `python -m rl.convert_cuda`）。
转换器只依赖 CPU PyTorch/Transformers/safetensors/PEFT，不依赖 MLX，支持 Qwen3 架构。
它将 MLX affine 4/8-bit 权重解包成标准浮点 HF safetensors，并可合并 LoRA 或输出 PEFT。
目标不是 GGUF、AWQ、GPTQ 或 NF4；需要这些格式时应对导出的 HF 模型另行量化。

```bash
# 在当前 Mac 保留两组依赖；Linux 转换环境只需 --extra cuda-export
uv sync --extra rl --extra cuda-export

# 全参 QAT；全参浮点训练改为 --model output/run/full_model
./scripts/convert_to_hf.sh --model output/run/qat_model \
  --output output/hf-qwen --dtype bfloat16

# 局部 QAT：需要与训练一致的底座以及本框架新导出的身份文件
./scripts/convert_to_hf.sh --model models/Qwen3-0.6B-4bit \
  --qat-export output/run --output output/hf-qat

# QLoRA 合并为标准 HF 模型
./scripts/convert_to_hf.sh --model models/Qwen3-0.6B-4bit \
  --adapter output/run/lora_adapter --output output/hf-merged

# 保留 PEFT adapter；同时导出解量化的匹配底座 base_model/
./scripts/convert_to_hf.sh --model models/Qwen3-0.6B-4bit \
  --adapter output/run/lora_adapter --format peft --output output/hf-peft

# NVIDIA 机器上的实际验收；安装适合该机器 CUDA 的 PyTorch
python -m rl.verify_cuda --model output/hf-peft --device cuda \
  --output output/cuda-check.json
# 若有 MLX 侧 input_ids/logits NPZ，追加 --reference PATH 可做数值对比。
```

输出目录必须是新目录。默认 BF16 会有舍入误差；跨框架精确对比使用 `--dtype float32`。
`--max-shard-size-mb` 控制分片目标大小（单个 tensor 不拆分）；manifest 记录来源、输出哈希。
底座身份不匹配会拒绝转换。训练的 optimizer/critic 不转入 HF；MLX 续训仍使用原 checkpoint。

PEFT 文件布局遵循 [PEFT checkpoint 格式](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)，
LoRA 缩放对应 [LoraConfig](https://huggingface.co/docs/peft/main/package_reference/lora)。
将整个目录复制到 NVIDIA 机器后，显式加载实际路径，避免沿用 adapter 配置中的原机器绝对路径：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
root = "/path/to/hf-peft"
base = AutoModelForCausalLM.from_pretrained(root + "/base_model", dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base, root + "/adapter").eval()
tokenizer = AutoTokenizer.from_pretrained(root + "/base_model")
# 原版 Qwen3 通过 apply_chat_template(..., enable_thinking=False/True) 控制思考模式。
```

`rl_generation.json` 是本项目元数据，Transformers 不会自动应用它；部署方需设置思考模式和采样温度。
CUDA verifier 明确报告设备，未检测到 NVIDIA GPU 时不会把 CPU 验证称作 CUDA 验证。

### Qwen3-0.6B 验证范围

本地底座由官方 `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` 转为 MLX 4-bit/group64，下载文件 SHA256 已核验。
测试入口 `tests/rl/test_qwen06_integration.py` 覆盖 thinking/no-thinking、单条/批量原生工具调用、
全层 QLoRA PPO/GRPO、全参 QAT PPO/GRPO、故障续训和跨 PyTorch CPU 数值校验。
训练用受控奖励与评估桩，验证真实更新及恢复，不代表真实沙箱任务奖励提升。

0.6B 对提示词敏感：初始中文任务发生直接回答、思考标签不闭合；普通英文任务在 no-thinking
下会输出缺少 `<tool_call>` 标签的 JSON。明确提示原生工具标签后简单工具任务可以完成，
不因此放宽协议解析或宣称所有任务可靠。greedy 用于可重复诊断，不作为 thinking 的采样建议。
本机为 Apple Silicon，独立部署数值验证运行在 PyTorch CPU；NVIDIA CUDA 硬件仍需用上述命令验收。
验证证据保存在 `output/validation/qwen3-06-optimizations/`。

本轮验证（2026-10-03，Apple M5 Max / 48 GiB）：RL 回归 **183 passed / 33 skipped**，
真实 0.6B 集成 **10 passed**，两个独立 packed rollout worker 的集成 **1 passed**。
六组训练各完成 6 次 optimizer step（含续训）；短受控轨迹的训练进程 Metal 峰值为
QLoRA 约 2.75 GB、全参 QAT 约 19.21 GB，不包含独立 worker 或长上下文容量结论。
六种产物在 float32 PyTorch CPU 上的最大 logits 误差为 `4.53e-5` 至 `1.73e-4`。

固定 prompt=16、response=4、单序列的全参 QAT 探针：只将 reference 从 master 改为 packed，
峰值从 **16.29 GB 降至 14.28 GB**，reference 加载后常驻从 4.77 GB 降至 2.76 GB。
该路径没有触发数值回退。prompt=128、response=32 的重复 token QLoRA 探针触发 cached fallback，
优化前后峰值均约 3.46 GB；这组结果不支持重算带来内存收益的结论。梯度等价性由小模型独立测试覆盖。
测量时系统已有约 6.7–6.8 GB swap；这些数值是当前进程的 Metal 张量内存，不是整机总内存，
也不证明运行完全不使用 swap。完整阶段记录及回退计数见验证目录中的 `measured-*.json`。
