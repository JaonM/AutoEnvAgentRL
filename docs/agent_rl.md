# Apple Silicon 异步 Agent RL

实现位于 `src/rl/`。独立 rollout worker 在真实沙箱采样，Actor（策略训练侧）在本机 MLX/Metal 更新策略。每个 rollout worker 默认同时推进两条 rollout。
算法只有 **PPO / GRPO**（默认 GRPO），两者共用行为概率比、裁剪目标与有限轨迹复用。

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

## 统一概率比与损失

每条轨迹保存生成时的策略版本、真实逐 token `old_logp`、终局奖励、终止标记；PPO 另外保存行为价值。本次 episode 评分后 bootstrap 为 0。
训练时计算 `ratio = exp(new_logp - old_logp)`，PPO 与 GRPO 共用：

```text
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-epsilon, 1+epsilon) * advantage))
```

- 奖励时机：reset 和中间 action 不读取奖励。episode 自然结束或到达步数上限后，由 RL rollout 框架调用 `finish()`，通过 `GET /v1/reward` 获取一次过程与结果综合评分；重复 finish 复用该分数。
- PPO：中间 action 奖励为 0，最后一个 action 的末 token 接收终局评分，利用保存的行为价值计算 GAE/returns；加 clipped value loss。步数上限也作为本次评分终点，bootstrap 为 0。
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
拒绝或零方差组记录为跳过，仍计入本轮已访问沙箱，不用下一轮沙箱补位；只有实际非零优化才发布策略版本。

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

## QAT 与数值一致性

默认 `--tuning qat`：最后 1 层 attention q/v projection 使用 FP32 master、4-bit affine fake quantization 和 STE。
其余基座量化权重冻结；可用 `--layers`、`--bits 8` 调整。实现为权重 QAT，不含 activation QAT。
另支持 `--tuning lora`。

采样使用逐 token KV cache，Actor 评分／训练采用右侧 padding 的完整序列张量批次，浮点参数使用 FP32。
真实采样 log-prob 与独立 cached 重算的误差门限仍为 0.001；行为概率始终保留。
批量路径的 kernel 数值差异单独测量，超过 `--batch-logp-tolerance`（默认 0.01）的形状自动拆小，单条仍超标时显式回退 cached 路径；回退次数和原始最大误差进入报告。
QAT 导出仅含训练层，需要同一基座和 tokenizer；部署身份清单保存文件哈希。
验收使用全新模型实例加载 packed 层，核对概率误差并执行真实评估。

## 恢复、产物与性能

- `checkpoints/latest.json` 指向原子提交的完整模型、Adam、RNG、计数器、rollout worker 位置及 replay 状态。已提交 mini-batch 不重复更新，未提交组重新采样。
- 续训要求代码、依赖、模型/tokenizer、任务及训练配置身份一致；可增加 epochs 继续遍历，也可调整可选 updates/max-groups 预算。
- `--max-rollout-restarts 2` 有限恢复可重试故障；持久恢复日志防止 resume 清零未提交的重启次数。
- `--checkpoint-keep 3` 保留完整检查点；Rollout worker 停止后清理旧策略快照，保留初始参考和最近版本。
- `run_status.json` 记录 running/completed/failed/interrupted；训练异常不伪造零奖励。
- `optimizer_metrics.json` 记录每个 optimizer step 的加权损失；`metrics.json` 记录更新、replay、拒绝和阶段耗时。
- `training_report.json` 检查策略独立变化、有效量化权重变化、实际更新、独立回载和异步重叠。Critic 单独变化不能代表策略训练成功。
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

`optimizer_metrics.json` 分别记录 epoch、dataset_batch、optimization_pass、mini_batch、
mini_batch_sandboxes 和 mini_batch_rollouts。`metrics.json` 记录每个 batch 的沙箱身份与 replay 来源，
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

仅通过 prepare 门禁的完整组占据 batch/mini-batch 配额。组内奖励零方差（GRPO）、
策略版本过旧或概率漂移不合格会触发原沙箱补采；新尝试使用新 seed、独立续跑目录，并在开始时加载已发布策略。
基础任务 ID 不变，`rollout_attempt` 从 0 递增。旧尝试的迟到通知不参与新尝试消费。
保留每轮数据集完整覆盖：不会永久用容易通过的沙箱替代难沙箱；某沙箱耗尽尝试上限，运行明确失败。
`max_groups` 非零时也限制总候选消费量；不会无限补采或将不足额的训练 batch 声明为完成。

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
