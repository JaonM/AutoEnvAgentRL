# env-factory

代码入口和模块边界见 [代码结构](docs/code_structure.md)。

任务生成与沙箱构建的可恢复循环、真实模型 rollout 和模型默认配置见 [循环工程实验](docs/loop_experiments.md)。

项目当前以“生产级 Agentic RL 训练素材准备”为认证边界；指标、证据和结果语义见
[训练素材生产准备认证](docs/production_readiness.md)。EnvFactory 不执行 RL 训练；该认证只覆盖下游接入前
的任务、环境、轨迹与奖励素材，不宣称能够直接开训、RL 已完成或训练后模型已经提升。

生成契约、工具实现、持久化及验收的最新边界见 [一致性改造说明](docs/runtime_integrity.md)，其中区分静态评分、离线回归与真实训练 rollout 证据。

## 主链路：Code Agent 生成任务并构建沙箱

默认作者、沙箱构建与独立审查均使用 `GPT-6-luna`。任务从真实 Scene 图谱路径生成，
经过前置验证后进入沙箱构建、离线评分和 live 训练门禁。准备好下文 `.env`、Neo4j、
已认证的 `codex` CLI 与 Docker，并执行 `uv sync` 后运行：

```bash
./scripts/run_pipeline.sh --help
./scripts/run_pipeline.sh --dry-run
./scripts/run_pipeline.sh --output output/my_batch
```

默认单轮 **5 个样本**、并发上限 **4**、`pilot` 候选验证、Docker 与 live 验收；
不启用 5 分钟硬截止。默认类别比例 20/30/50，生成阶段防卡死超时 600 秒。
参数、产物位置、恢复方式和退出码见脚本 `--help`；完整调度参数用 `--engine-help`。
最终消费 `status.json.training_ready == true`，构建 `success` 不能替代训练验收。
`pilot` 环境通过门禁不等同于 production 批次认证或已验证学习收益。

## 构建知识图谱

先确认 Neo4j 可访问，并在项目根目录配置 `.env`：

```bash
cp .env.example .env
```

至少配置以下变量：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model
# 可选：不设置时逐字段回退到 LLM_*
ROLLOUT_LLM_MODEL=your-policy-model
SANDBOX_LLM_MODEL=your-user-and-judge-model
LLM_TIMEOUT=60
WIKIPEDIA_API_URL=https://zh.wikipedia.org/w/api.php
WIKIPEDIA_TIMEOUT=10
# WIKIPEDIA_DUMP_DB=data/wikipedia.sqlite3
NEO4J_DATABASE=neo4j
GRAPH_SEEDS_FILE=data/scene_seeds.txt
LOG_LEVEL=INFO
```

`GRAPH_SEEDS_FILE` 指向种子词文本文件，每行一个词语；空行和以 `#` 开头的行会被忽略。
构建成功后，本轮发现的 scene 词语会自动追加到该文件；Neo4j 中已标记为扩展完成的词语下次会跳过。
`LOG_LEVEL` 支持 `DEBUG`、`INFO`、`WARNING` 等级别，默认使用 `INFO`。

统一入口构建 Scene 图谱。配置 `WIKIPEDIA_DUMP_DB` 时使用本地 Wikipedia 索引；否则搜索在线 Wikipedia。LLM 抽取与关系判断使用 `.env` 中的端点。

```bash
./scripts/build_graph.sh
./scripts/build_graph.sh --offline
./scripts/build_graph.sh --online-wikipedia
```

运行 `./scripts/build_graph.sh --help` 查看参数。

指定扩展参数：

```bash
./scripts/build_graph.sh \
  --rounds 2 \
  --max-scene-nodes 500 \
  --max-search-requests 100 \
  --max-workers 2
```

脚本根据 `.env` 选择本地或在线 Wikipedia，批量调用 LLM 抽取词语，增量合并 Scene 节点和关系，最后写入 Neo4j。`task_type` 节点默认写入全部枚举值。

### 大规模构建

大规模任务建议先下载并建立本地索引：

```bash
./scripts/download_wikipedia_dump.sh
./scripts/index_wikipedia_dump.sh data/zhwiki-latest-pages-articles-multistream.xml.bz2
```

然后在 `.env` 中配置：

```dotenv
WIKIPEDIA_DUMP_DB=data/wikipedia.sqlite3
```

配置本地索引后，构建流程不再请求在线 Wikipedia API，直接使用 SQLite FTS5 检索页面正文。数据 dump 体积较大，下载和索引耗时取决于网络与磁盘性能。

## 生成长程任务

任务生成默认使用 Code Agent，从 Scene 图谱路径设计业务数据、工具、奖励与参考轨迹，
由共享编译器执行前置验证，再物化交付物。仅生成任务可运行：

```bash
./scripts/generate_task.sh --count 5
./scripts/generate_task.sh --training-category multi_step_agentic --count 5
```

`spec` 与 `legacy` 保留为显式选择的兼容/对照后端：

```bash
./scripts/generate_task.sh --generation-backend spec --training-category multi_step_agentic --count 5
./scripts/generate_task.sh --generation-backend legacy --hops 3 --training-category multi_step_agentic
```

完整生成契约见[任务生成流水线](docs/task_generation_pipeline.md)与
[Code Agent 编写规范](docs/code_agent_authoring.md)。`spec` 多步原型不调用生成模型，
不代表默认 Code Agent 主链路的模型调用量或质量。

规格流水线从目标和依赖编译 `rule-based` 指标，模型阶段流水线根据任务描述与环境生成 `rule-based/model-based` 指标，统一写入 `Task.metrics`。模型阶段还支持噪声工具，覆盖相关无关与完全无关两类；噪声工具由共享运行时提供无任务关键副作用的通用实现，不占用业务 handler 实现成本，也不产生任务进度奖励。工具生成前会把动作分类为环境操作、Agent 推理和 Agent 回答，只有环境操作可以暴露为工具。任务规模不再绑定具体构建模型，结构有效性由 schema、契约、任务级 readiness、外层验收和训练素材准备就绪门禁统一判断。

沙箱通过普通 acceptance、outer conformance 和 mutation testing 后，还必须通过 `scripts/sandbox/validate_training_readiness.py` 的 RL 环境硬门禁。该门禁执行结构化成功、失败、噪声及反事实轨迹，检查奖励可分离性、确定性和公开 observation 泄漏，并输出 `training_readiness.json`。
默认在 `output/task/task-N/task.json` 写入每个任务的最终文件；可通过 `--output` 指定输出根目录。每次运行会扫描已有 `task-N`，从当前最大编号的下一号开始追加，绝不覆盖已有任务；并发进程通过原子目录预留避免编号冲突。Pipeline 运行日志默认追加写入 `output/task_generation.log`，也会输出到终端，可通过 `--log-file` 指定其他文件。日志记录任务级和阶段级开始、重试、成功、失败、耗时、产物路径和进度，不记录 Prompt 或凭据。任务默认并发生成 4 个，可通过 `--max-workers` 调整并发数。图谱路径的 Neo4j 查询默认 10 秒超时，可通过 `--path-query-timeout` 调整。
使用 `--count N` 可批量新增 N 个独立的 `task-N` 目录；失败任务保留采样身份和失败归因证据，不会修改任何历史任务。

### 任务质量评分与过滤

生成后可使用确定性离线评分器筛选训练样本。评分范围为 0–10，覆盖任务契约、Agentic 难度、环境与工具对齐、奖励可评测性、验收与训练就绪度；不调用 LLM，同一产物会得到相同结果。

```bash
uv run python examples/score_tasks.py output/task \
  --min-score 8 \
  --report output/task_quality_report.json \
  --csv output/task_quality_report.csv
```

复制合格样本到独立目录：

```bash
uv run python examples/score_tasks.py output/task \
  --min-score 8 \
  --accepted-dir output/accepted_tasks
```

只提取高价值 Agentic 样本：

```bash
uv run python examples/score_tasks.py output/task \
  --min-score 8 \
  --high-value-dir output/high_value_tasks
```

评分器拒绝覆盖目标目录中的同名任务。CI 中可增加 `--fail-on-low-score`，只要存在低于阈值或未通过训练资格门禁的任务便返回非零退出码。任务评分采用 0–10 分制；`task_quality_report.json` 的 `score` 是包含资格门禁的有效分数：任一硬门禁失败即为 0 分，`eligible=false`。`raw_score` 保留门禁前的质量分，维度得分和失败原因供诊断；只有资格通过的任务才按质量分排序。

### 沙箱离线评分与过滤

`scripts/sandbox/score_sandbox_offline.py` 不调用模型或网络，重新执行业务验收、pytest、契约一致性、运行时通用性、outer conformance、五类 mutation 和 training-readiness，并结合已有独立语义审查与可执行成功/失败轨迹。沙箱评分采用 0–10 分制；通过全部关键门禁后，再按独立语义审查分和成功轨迹相对已完成负例轨迹的奖励差距计算连续分数。任一关键检查失败即为 0 分、`eligible=false`；`raw_score` 保留门禁前的通过项权重和，`quality_factors` 记录连续分的依据。离线通过仍需 live 验证。交付、契约或语义审查预检失败时，评分器跳过后续昂贵检查，并在检查证据中标明原因。

评分单个沙箱：

```bash
uv run python scripts/sandbox/score_sandbox_offline.py output/sandbox_loop/round-10/task-92
```

批量扫描目录并生成过滤报告：

```bash
uv run python scripts/sandbox/score_sandbox_offline.py output/sandbox_loop/round-10 \
  --threshold 8 \
  --output output/offline_sandbox_scores.json
```

全部沙箱达到阈值时退出码为 `0`，存在低分沙箱时为 `1`，可直接用于 CI 或数据集过滤。默认还会在每个沙箱目录写入 `offline_sandbox_score.json`；传入 `--no-individual` 可只保留汇总报告。

沙箱构建成功后会默认自动执行该离线评分，并写入 `offline_sandbox_score.json` 和 `offline_score.log`。如果上层流程已经安排了独立评分，可向 `develop_sandbox_with_agent.sh` 传入 `--skip-auto-score` 避免重复执行。

任务环境由完整业务数据、数据说明文档、用户 FSM、原子 Agent 动作、工具契约和观测奖励设计组成。任务生成器把这些语义编译为版本化 `task_spec`，显式声明环境 archetype、初始/成功谓词、状态增量、工具输入输出/effect、能力 DAG 和奖励真值来源。episode、持久化、工具注册、User Simulator、奖励聚合与因果门禁由 EnvFactory 共享运行时实现；Code Agent 只补充无法声明化编译的少量业务 handler 或 metric extension。
任务生成结果不再包含 `constraints` 字段。沙箱构建脚本从 `task.json` 生成只读 `BUILD_CONTRACT.json`；`development_plan.json` 由 EnvFactory 根据 TaskSpec 和 archetype 确定性生成，不由 Code Agent 重新设计平台架构。
其中 `user_profile` 不再直接从任务描述臆造，而是从任务关键词随机选择 1 到全部关键词，并发查询其直接 `HIERARCHY` 下位节点，再由 LLM 润色生成；任务描述仅用于生成 `task_info`、状态和执行规则。
观测指标只保留与任务目标强相关的少量关键过程指标和目标结果指标。关键过程指标使用 `hybrid`：外部 LLM 生成当前上下文下的期望工具名和参数，规则引擎再对实际工具调用进行规范化比对；工具或参数错误不作为 penalty。结果指标用于判断目标是否完成，惩罚指标仅保留直接影响任务目标的偏离或无效循环。指标包含 `id`、`category`、`type`、`scope`、`condition/criteria`、`weight` 和 `score_range`，分别用于 step/state/terminal/trajectory 级别的奖励计算。

输出示例：

```json
{"task":"帮我挑选一套合适尺码的衣服并完成购买","task_type":"Event","complexity":"standard" ,"environment":[{"type":"user_profile","field":"interest","description":"用户兴趣","value":"服装","visibility":"observable"},{"type":"task_info","field":"goal","description":"任务目标","value":"提交订单"},{"type":"state","field":"order_status","description":"当前订单状态","value":"pending","visibility":"hidden"},{"type":"action","field":"submit_order","description":"提交订单","value":"submit_order","visibility":"hidden"},{"type":"transition_rule","field":"submit_order_rule","description":"提交订单后的状态变化","value":{"when":"submit_order is called","effect":"order_status becomes submitted"},"visibility":"hidden"},{"type":"termination","field":"success","description":"任务成功条件","value":["order_status == submitted"],"visibility":"hidden"}],"metrics":[{"id":"task_success","type":"rule-based","scope":"terminal","condition":"order_status == submitted","reward":1.0,"penalty":0.0,"once":true,"rubric":"订单提交成功","weight":1.0}]}
```
