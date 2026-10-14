# 训练：SFT 冷启动与强化学习

训练框架位于 `src/rl/`，在 Apple Silicon（MLX / Metal）上运行异步 PPO / GRPO。训练环境是生产线产出的**认证沙箱**：每个 episode 通过沙箱的 HTTP 接口交互，奖励由独立 verifier 计算，策略只能看到公开说明（手册、工具、对话）。

```text
tasks.json（认证沙箱清单）
        │
        ▼
rollout workers ──批量解码──► 策略模型（MLX）
        │  每个 episode：reset → 多轮 step（说话 / 调工具）→ 终局奖励
        ▼
完整轨迹（同一任务同一 seed 的一组）
        │
        ▼
Actor：组内优势 → 截断代理目标 + KL → 更新策略 → 发布新快照
```

## 1. 输入：任务清单

`scripts/run_pipeline.sh` 结束时写出 `tasks.json`，可以直接用于训练：

```json
{"tasks": [{"id": "gen-001", "sandbox": "build/gen-001/bundle", "split": "train", "weight": 1.0}]}
```

- `sandbox` 是认证沙箱目录（相对清单所在目录）；加载时会校验签名，平台代码变化后需要 `python -m env_factory verify BUNDLE --resign` 重新认证。
- `split: eval` 的任务只用于评估和挑选检查点，不参与训练。

## 2. 启动训练

```bash
uv sync --extra rl
./scripts/train_rl.sh --tasks output/run_xxx/tasks.json --model models/Qwen3-0.6B \
  --output output/train-01 --tuning lora --thinking-mode no-thinking \
  --rollout-group 8 --rollout-concurrency 8 --max-steps 30 --advantage-std task
```

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--sandbox-backend` | `docker` | `docker`：每个沙箱从它通过离线验证的加固镜像启动；`local`：在进程内服务，便于开发 |
| `--algorithm` | `grpo` | `grpo` 或 `ppo` |
| `--rollout-group` | 4 | 每个任务同一 seed 采样的轨迹数（GRPO 的组大小），多轮 agent 任务建议 8–16 |
| `--rollout-concurrency` | 2 | 每个 worker 同时推进的环境数，建议不小于组大小 |
| `--max-steps` | 6 | 每局最多的策略动作数；多轮对话建议约 30 |
| `--advantage-std` | `group` | `task`：组内减均值，标准差取该任务最近的奖励，组小时更稳定 |
| `--zero-variance-policy` | `retry_skip` | 组内奖励全部相同时：重采，仍相同则跳过 |
| `--tuning` | `qat` | `qat`、`lora` 或 `full` |
| `--thinking-mode` | `auto` | `thinking` / `no-thinking` 控制 Qwen3 模板 |

完整选项见 `./scripts/train_rl.sh --help`。

## 3. Episode 的协议随机化

为了让策略学会"读说明再行动"，而不是记住某个平台的约定，每局按 seed 随机化交互协议（环境变量控制，默认均为随机）：

| 环境变量 | 取值 | 作用 |
| --- | --- | --- |
| `ENVFACTORY_USER_MODE` | `natural`（默认）/ `structured` | 自由对话的模拟用户，或按主题键询问的结构化用户 |
| `ENVFACTORY_TOOL_NAMES` | `random` / `fixed` / `heldout` | 平台工具名和授权参数名的改名方式 |
| `ENVFACTORY_CONSENT` | `random` / `plain` / `token` | 写操作的同意协议：纯对话确认，或"提议 + 授权凭证" |
| `ENVFACTORY_RESULT_FORMAT` | `random` / `envelope` | 工具返回为平台包装格式，或原始 JSON / `Error: ...` 文本 |
| `ENVFACTORY_SYSTEM_LAYOUT` | `random` / `fixed` | system prompt 的版式（纯文本、标签、Markdown、政策在前） |
| `ENVFACTORY_ANSWER_TOOL` | `never` / `random` / `always` | 是否提供答案工具；不提供时答案从 agent 说过的话里抽取 |
| `ENVFACTORY_TOOL_ERROR_LIMIT` | 3 | 一局内沙箱工具崩溃达到次数即隔离该沙箱 |

## 4. 奖励与无效 episode

- 奖励 = 业务分（最终状态 + 答案各字段的得分）+ 效率分（只在成功时计入）；任何未经同意的写入使整局为 0。
- 失败的自由对话 episode 会审计模拟用户是否出错（漏说、说错、编造、与平台决定相反）。查实的 episode 标记为不可训练：不参与组内基线，也不产生梯度。
- 沙箱自身反复崩溃的任务会被隔离，不再采样。

## 5. SFT 冷启动

小模型直接做 RL 往往缺少成功轨迹，建议先用强模型的示范做 SFT。`experiments/` 下提供了一套工具：

| 脚本 | 作用 |
| --- | --- |
| `collect_teacher.py` | 强模型通过与训练完全相同的接口玩任务，保存完整对话和判分 |
| `regrade_teacher.py` | 判分规则更新后离线重判已有回合（只重放，不再调用模型） |
| `sft_select.py` | 按任务、领域、类型、语言、技能配比筛选，剔除近似重复，输出多样性报告 |
| `sft_train.py` | MLX 上的全参数或 LoRA SFT；整局渲染或按轮渲染；工具改名、返回格式、问候语、版式等增强 |
| `interpolate.py` | SFT 后与基座做权重插值（WiSE-FT） |
| `dev_check.py` | 在留出任务上检查行为：重复失败调用、调用不存在的工具、撞步数上限等 |

## 6. 输出

| 路径 | 内容 |
| --- | --- |
| `OUTPUT/checkpoints/` | 可恢复的检查点（中断后同一命令续训） |
| `OUTPUT/logs/` | 实时指标与事件日志（奖励、成功率、组内方差、KL 等）；TensorBoard 默认开启，`./scripts/train_dashboard.sh` 打开 |
| `OUTPUT/evaluations/`、`evaluation_*.json` | 训练前后及周期性的评估结果 |
| `OUTPUT/best_policy/` | 按留出任务挑选的最好策略 |
| `OUTPUT/training_report.json`、`run_status.json` | 训练报告与运行状态 |

导出到 Hugging Face / CUDA 格式：`./scripts/convert_to_hf.sh`（需要 `uv sync --extra cuda-export`）。
