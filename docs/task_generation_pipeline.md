# 任务生成与沙箱构建流水线

## 主入口

```bash
./scripts/run_pipeline_legacy.sh --count 1 --generation-hops 3
./scripts/run_pipeline_legacy.sh --count 5 --output output/my_tasks
./scripts/run_pipeline_legacy.sh --code-agent-model gpt-6.1-sol --sandbox-model gpt-6.1-sol --language en
./scripts/run_pipeline_legacy.sh --help
```

脚本调用 `python -m env_factory.task_pipeline`，Python 主入口位于
`src/env_factory/task_pipeline.py`。也可直接运行该模块；默认生成并构建一个任务。
只有 Code Agent 这一条实现链路，支持 Codex、Claude、OpenCode。后两者需要显式传入
`--code-agent-model`。`--sandbox-model` 控制构建与独立审查，不影响 live actor 或用户模拟模型。

不再提供 `--layout`、`--generation-backend`、`--max-rounds`、留出批次或循环实验入口。
旧 `legacy`、`spec` 后端及模型阶段缓存已删除。脚手架和确定性编译器是 Code Agent
链路的组成部分，不是另一个生成后端。

## 执行顺序

1. `src/env_factory/generate_task.py` 预留递增任务编号，并按训练类别配比采样 Scene 多跳路径。
2. `TaskGenerator` 将图谱路径和约束交给 Code Agent，作者输出业务定义。
3. `generation/agent_authoring.py` 编译业务定义。`generation/contract_compiler.py` 提供共享确定性校验、工具和数据物化；业务契约、决策表、状态目标与奖励使用已有共享运行时。
4. 编译回放、反例、前置质量和可构建性门禁拒绝不合格任务。生成完成必须有原子完成状态及匹配的任务指纹。
5. `scripts/develop_sandbox_with_agent.sh` 调用 Code Agent 构建（包括重建已有任务，不能因缺少来源标记而跳过集成验证），执行独立审查和离线门禁，并构建 Docker 镜像。
6. 主入口复核评分，执行 live rollout、轨迹审计及真实奖励反例校准，最后回收临时容器、镜像并写最终状态。

当前生成阶段结束后，按顺序构建已通过前置门禁的任务。部分任务生成失败不阻止同次调用中其他合格任务构建，不会捡取历史任务冒充本次生成成功。

## 输出与日志

- 任务：`<output>/artifacts/artifact-N/task/`。
- 沙箱：`<output>/artifacts/artifact-N/sandbox/`。
- 每次调用日志：`<output>/logs/pipeline-*`。
- 阶段事件：`<output>/pipeline_events.jsonl`。
- 生成日志：`generation.log`、`generation_process.log`。
- 单任务结果：对应目录下 `pipeline_result.json`；构建状态见沙箱 `status.json`。

没有批次、round 或 sample 目录层级；同一输出目录使用文件锁，禁止并发入口争抢编号。
已有沙箱通过同目录恢复机制继续构建，不跨轮复制其他沙箱实现。

## 默认值与验收含义

`--generation-hops N` 配置随机跳数上限，N 范围 0..20，默认 2。每次生成尝试从 0..N（含两端）均匀抽取实际跳数；0 跳表示一个节点、无关系。实际跳数写入作者 request.json 的 graph_context.hops，上限为 max_hops。

默认任务与沙箱模型为 `gpt-6.1-sol`，语种 `zh-CN`，生成与修复模型累计预算 600 秒；
不设置任务生成加构建的 5 分钟硬淘汰线。训练类型默认配比为直接回答 20%、单步 30%、多步 50%。
拟真机制统一记录在 `realism_requirements`；不再提供 `--realism-profile`，也不覆盖用户指定的任务意图。

构建不执行独立真实 rollout、live 奖励校准或轨迹隐私检查，主入口已移除 `--validation`。
保留 Docker 启动与接口检查、参考/失败轨迹、奖励反事实、独立语义审查和数据治理检查。
评分直接采用可执行验收的完整 10 分，不再保留 live 加分。
`qualification_basis=executable_v1`；未执行检查记录在 `validation_checks` 中为 `not_run`，对应 verified 为 false。
`training_ready` 表示通过上述门禁、允许进入训练，不保证真实策略成功率或模型评分准确率。
`passed` 表示通过本次选择的门禁，`status.json.training_ready == true` 才表示通过可训练判定。
保留有界业务修复与基础设施恢复，不保留自动换题、多轮质量晋级或循环实验调度。

退出码：0 表示本次门禁通过；1 表示任务不合格；2 表示配置或基础设施失败。
独立素材认证与签名导出仍使用 `scripts/certify_training_materials.py` 和
`scripts/export_training_materials.py`，不会因入口迁移自动认定素材通过认证。


## 统一拟真机制

在每次调用的多步任务中，对允许 stateful 环境的名额按正常 50%、外部变化 30%、故障恢复 20% 分配要求。
多个名额按最大余数法取整；只有一个适用名额时按概率抽样，避免日常单任务永远只生成正常场景。
分配由运行种子决定，并在 `sample_manifest.json` 中保存，候选重采样及修复沿用同一要求。
明确限制只读环境时不强塞外部写入；直接回答、单步任务保持原训练类别分配。
“正常”表示没有额外强制事件／故障要求，仍可按业务需要声明更多机制。

作者可声明 clarification、authorization、external_change、fault_recovery、idempotency、compensation。
外部事件必须实际改变数据；故障必须实际触发并在成功轨迹中恢复；幂等要求必须实际重放；补偿工具必须实际成功执行。
恢复场景不必同时叠加无关的外部事件。所有路径继续通过共享目标、奖励、反例和独立语义审查。
用户确认阶段必须明确授权权限及条款，平台在提交前检查参数、提交后在同一事务内检查批准的业务状态，失败回滚。
只有用户确实没有限定条款时，才能声明 operation 范围授权；独立审查核对授权内容与用户原话。

`realism_summary.json` 统计本次调用各机制的请求数、生成成功数、进入构建数、通过数、可训练数和基础设施失败数。
同时记录比例；机制可重叠，不能相加得到总样本数。未创建任务目录的请求单列为 untracked_requests，不把它计为合格。



### 环境模式

使用 `./scripts/run_pipeline_legacy.sh --environment-mode stateful` 限定新任务的业务环境形态，不改变 Docker 启动方式。生成结果不符合指定模式时会被前置验收拒绝。

| 枚举值 | 含义 | 示例 |
| --- | --- | --- |
| `stateless` | 无业务数据表，依据公开输入完成任务 | 阅读材料回答问题 |
| `reference_data` | 查询参考数据，无需修改业务状态 | 查询库存、比较方案 |
| `stateful` | 修改并持久化业务状态，要求实际写入工具 | 创建订单、修改预约 |
| `external_capability` | 使用已配置的外部能力 | 调用外部服务完成任务 |

默认不指定，由 Code Agent 在可用且与训练类别兼容的模式中选择。默认可用模式为前三种；配置非空 `SANDBOX_EXTERNAL_CAPABILITY_URL`，或 `SANDBOX_EXTERNAL_FIXTURES` 指向存在的文件后，额外允许 `external_capability`。未配置时显式选择外部模式会报错。

**Fixture（测试数据或模拟响应文件）**：预先准备外部服务的数据或响应，例如固定的物流状态、查无订单的响应或服务失败场景，供沙箱可重复地模拟外部能力。通过 `SANDBOX_EXTERNAL_FIXTURES=/path/to/fixtures.json` 指定文件。生成入口检查文件是否存在，以决定是否允许该模式；文件存在不代表内容有效或沙箱已通过验收。

该参数不自动修改训练类别配比，指定模式仍需满足训练类别的兼容性约束。

### 交叠执行与目录

生成端和构建端交叠运行：单个任务的 manifest 标记 `completed` 且 `task.json` 校验和匹配后立即进入构建，无需等待其他任务生成完成。当前最多一个生成进程和一个沙箱构建同时工作；构建侧逐个消费就绪任务，失败样本独立记录。

每个样本保存在 `output/artifacts/artifact-N/`，其中 `task/` 保存任务交付物，`sandbox/` 保存构建及验收产物。已有旧目录不会自动移动。

实际抽到 H 跳后，要求路径恰好有 H 条关系、H+1 个节点；路径采样重试不会擅自降低 H。
