# 循环工程实验

`scripts/run_sandbox_build_loop.py` 使用可恢复实验调度器。默认不复用历史高分，构建与独立审查模型固定为 `gpt-6-luna`。
新任务默认沿知识图谱＋已审核数据集路径生成；`--generation-source dataset` 可显式运行旧数据集路径。Kaggle 与 DATA.GOV.HK 目录元数据可先用 `scripts/diagnostics/sync_graph_dataset_links.py` 导入 Neo4j，实验生成时会再次核对已审核关系和原始数据。

正式良品率实验建议每轮生成 10 个新任务，固定使用
`direct_response=20%`、`simple_agentic=30%`、`multi_step_agentic=50%`。
实验分为两个阶段。开发阶段连续两轮达到任务良品率、条件构建良品率、端到端良品率、
分路由下限和合格样本均分后，自动进入生产准备留出集：

```bash
uv run python scripts/run_sandbox_build_loop.py \
  --generate-count 10 --max-rounds 0 --consecutive-rounds 2 \
  --validation live --build-mode clean \
  --sandbox-runtime docker \
  --experiment-seed 20260925 \
  --target-task-yield 0.85 --target-build-yield 0.80 \
  --target-end-to-end-rate 0.70 --target-qualified-mean 8.5 \
  --target-category-rate 0.60 \
  --output output/loop_experiment_v1 --hypothesis baseline
```

`--max-rounds 0` 是默认值，表示开发阶段没有轮数上限；仍受累计活跃时间预算和人工暂停控制。
默认 `--certification-profile production` 连续执行 3 个独立批次，每批使用 400 个留出请求并要求
至少形成 300 个全新任务；每个合格沙箱执行至少 10 次 rollout。各批次采用独立 seed、从零构建，
内容和 seed 不得与开发轮或其他留出批次
重复；数字、标点或轻微措辞改写形成的近重复任务家族同样不得跨越开发集和任何留出批次。
完成后由 `scripts/certify_training_materials.py` 根据置信区间、类别覆盖、重复率、跨分区家族隔离、运行时隔离、
奖励反事实和轨迹完整性生成 `production_readiness.json`。只有停止原因
`production_prepared_for_agentic_rl` 表示生产级训练素材准备认证通过。

需要快速验证候选实现时，可显式使用
`--certification-profile pilot --holdout-count 30 --holdout-rollout-episodes 3 --holdout-end-to-end-rate 0.70`。
Pilot 通过只表示可以进入更大规模认证，
不能表示生产准备完成。Pilot 只执行一个批次；生产认证的三个批次均为一次性留出集，失败后保留证据
并退出，不能回流调参后继续冒充留出集。

固定回归集可使用 `--task-ids 45,78,92,175`，不能与 `--generate-count` 混用。
`--build-mode repair` 允许固定任务继承上轮扩展实现，不复制 SQLite 状态及验收报告；其结果表示修复能力，不表示从零构建能力。新任务始终不继承旧任务实现。

## 配置和恢复

- `--threshold` 默认 8，统一采用大于等于；production 模式不允许低于 8。该值会进入冻结实验配置，
  并与 canonical 认证策略及其摘要绑定；不能通过 CLI、认证 API 或重写报告降低其他生产门槛。
  任务静态评分也是前置筛选，失败任务仍计入总请求数。
- `--validation offline` 只执行离线门禁，不能证明真实训练质量；默认 `live` 会追加真实模型 rollout。
- `--certification-profile production` 强制 `--sandbox-runtime docker`，每个候选环境必须实际构建内容寻址
  镜像并通过安全容器 smoke test；不能用 `none` 生成生产认证。pilot 可显式设置
  `--sandbox-runtime none`，但其结果仍只表示候选流程验证。并发 worker 使用互不相同的本地镜像 tag；
  smoke evidence 写入后立即移除临时镜像，避免大规模留出集发生 tag 串样或耗尽 Docker 存储。
  基础镜像可复用本机缓存，但必须具有匹配仓库和平台的内容摘要；Docker 构建或 smoke test 失败会保留失败状态。
- `--rollout-episodes 3 --rollout-steps 20` 控制开发阶段每个沙箱的轨迹次数和步数。开发阶段默认只要求至少
  一条成功证据；`--rollout-min-success-rate` 可提高该门槛。生产留出集默认使用 10 次 episode 和
  `--holdout-rollout-success-rate 0.6666666666666666`。
- live 模式的最终 10 分由离线可执行证据占 9 分、真实 rollout 占 1 分组成；任何 live 硬失败仍直接取消训练资格，不能依靠离线高分抵消。
- 离线评分通过后、任何外部模型调用前，流水线先运行数据治理审计并生成 `data_governance.json`。
  缺少有效数据来源声明、发现凭证或无法确定 Agent/User/Judge provider 时停止该样本；疑似 PII 会被记录。
  合成 fixture 及带来源和哈希的 Kaggle / DATA.GOV.HK 数据均可继续。该技术审计不替代实际外部端点所需的组织授权。
- live rollout 通过后会以 `SANDBOX_EVALUATOR_MOCK=0` 执行奖励反事实校准，写入
  `agentic_training_value_live.json`。真实 evaluator 的成功轨迹、失败轨迹、无工具、错参数、跳步、乱序和
  噪声轨迹不满足奖励分离时，样本仍不合格；离线 mock 报告不能替代该证据。
- production profile 保留完成冒烟验证的本地镜像直到 live rollout 结束；Agent 驱动器在宿主侧运行，
  但所有工具、状态持久化、User Simulator 和奖励调用都通过随机 loopback 端口进入实际容器。容器停止后
  才删除镜像。轨迹中的 image ID 与构建 provenance 不一致，或退回进程内 app，都会取消生产资格。
- live rollout 与奖励校准之间执行 policy-visible 隐私审计，写入 `trajectory_privacy.json`。User
  Simulator 内部 outcome/FSM 标签只进入 trainer-only evidence，Agent 可见 `result` 只保留实际传给
  Agent 的 `user_query`；发现凭证或隐藏控制字段时停止样本，不能进入便携训练素材包。
- `--build-timeout`、`--generation-timeout`、`--score-timeout`、`--rollout-timeout` 分阶段限时，超时清理进程组。
- `--max-total-seconds` 默认 259200，只累计实验进程的活跃执行时间；正常暂停不消耗预算。`runtime_state.json` 保存累计活跃时间和运行状态。质量循环不再因“停滞”或固定 20 轮提前结束；显式设置正数 `--max-rounds` 才启用轮数预算。基础设施超时属于异常中止而非质量收敛。
- 同一个实验目录只允许一个运行进程；每个任务完成即原子保存。相同命令重启可恢复，不重复执行已经完成的任务；未完成构建使用新的 attempt 目录。中断的生成不会自动重新抽样，缺失任务计为失败。
- 代码、配置或固定输入变化时必须换实验目录，以免把不同版本的结果混为一谈。实验还会冻结 Python、
  OS/架构、项目直接依赖的已安装版本以及 `pyproject.toml`、`uv.lock` 摘要；这些执行环境信息发生漂移时
  同样拒绝恢复。快照不采集主机名、路径、环境变量或密钥。旧版 history 不能直接作为新实验续跑。

`experiment.json` 保存配置及代码/输入指纹；每个生成请求在模型调用前写入
`sample_manifest.json`，记录 batch index、route、run seed、sample seed、可用环境能力和生成 provider。
每次 route attempt 还记录实际响应模型计数、finish reason、token usage 与 provider response ID 哈希；
不保存 prompt、响应正文、凭证或原始 response ID。生产认证会独立验证 provider、seed 和 attempt 链。
生成失败时保留目录并写入 `failure.json`，不会再删除失败样本的身份和归因证据。
`round-*/round_report.json` 保存任务状态；`history.json` 同时保存生成完成率、任务良品率、
条件构建良品率、端到端良品率、合格样本均分、分类型统计和停止原因。

开发阶段可从单个冻结实验的逐请求记录重算分路由良品率与失败码分布：

```bash
uv run python scripts/diagnostics/report_development_yield.py \
  output/loop_experiment_v1/history.json \
  --output output/loop_experiment_v1/development_yield.json
```

报告只用于开发诊断，不能代替生产认证。构建器在 `status.json.failed_phase` 保留最终失败门禁；
实验调度器据此区分业务构建、语义审查、运行时完整性、mutation 和 Docker 等失败。
Code Agent 本地 app-server 启动失败会写入 `failure_code=INFRA` 并立即停止该样本的构建重试，
不计作模型生成的业务沙箱缺陷。
历史实验缺少 `failed_phase` 时仍保留原来的通用 `BUILD_BUSINESS` 归因，不根据日志猜测或改写旧证据。
任务评分与构建前检查共用必需 HTTP 端点清单；缺少 Trainer `/v1/state` 等端点时，
保留描述性质量分，但取消训练资格并以 `TASK_RUNTIME_INTERFACE` 在 Code Agent 启动前拒绝。
任务生成器在每次候选完成后先写出 `task.json`，运行同一构建前检查；任务质量、验收 fixture 或工具链 schema
不合格的候选进入该路由的剩余重采样次数，不发布为成功任务。构建前检查对成功轨迹中的 capture/`$ref`
执行保守的 schema 兼容性分析：若上游输出必含下游明确禁止的字段，以 `TASK_TOOL_CHAIN_SCHEMA` 拒绝，
避免 Code Agent 在不可变契约上反复修复。动态键对象的 `additionalProperties` schema 由外层契约检查和
共享运行时一致接受，并在调用时验证动态值类型。

## 模型默认值

Rollout Agent 按字段优先使用 `ROLLOUT_LLM_API_KEY`、`ROLLOUT_LLM_BASE_URL`、`ROLLOUT_LLM_MODEL`
和 `ROLLOUT_LLM_TIMEOUT_SECONDS`；缺失或空值回退到任务生成使用的对应 `LLM_*`。User Simulator 和
奖励 LLM 同样优先使用 `SANDBOX_LLM_*` 并逐字段回退到 `LLM_*`。显式配置某个字段不会覆盖其他字段，
三个角色的 provider 身份分别冻结和审计；缺少实际所需角色的模型或密钥时明确报错，不静默使用 mock。
正式认证的前置检查还要求 Rollout Agent 与 User/Judge 的 provider 主机和模型名称均不同，且签名私钥与受信公钥配对。

CLI 从项目 `.env` 加载环境变量；独立沙箱/容器应由启动器注入这些变量，不复制 `.env` 或密钥进入沙箱文件。离线模式仍强制 mock。

## Rollout 证据

Agent 只获得题面、公开工具及观测，使用 JSON 动作协议自主调用工具或回复用户；不读取成功答案、奖励规则和隐藏业务状态。User Simulator 使用真实 LLM，协议失败的 fallback 会单独标记并阻止 live 验证通过。

保存 transition schema v2：每一步包含完整公开模型输入、原始输出、解析动作、工具/User Simulator 结果、
前后观察、奖励、terminated/truncated 和调用用量；同时保留 HTTP trace 与 replay。轨迹绑定任务哈希、
沙箱可执行输入摘要、模型及 provider 摘要。检查无操作高奖励、重复奖励不稳定，以及 stateful 目标不满足
却得到高奖励。模型没有完成任务记录为缺少成功证据，不直接断言环境有错。

这里的“User Simulator 结果”对策略侧仅指公开 `user_query`；outcome category、transition ID、match
status 和 termination reasoning 属于 trainer-only metadata。便携 JSONL 采用字段白名单重新投影，不会
因为未来在内部 rollout 结构中增加调试字段而自动把它们泄漏给训练策略。

开发阶段的默认 live 门要求至少一条成功轨迹、全部轨迹无已检测环境问题且无 LLM fallback；
发布留出集将成功率门提高到至少 2/3，并要求轨迹池同时包含成功和失败样本。
报告将失败责任区分为 `agent`、`environment` 和 `infrastructure`；Agent 未完成任务不再自动归咎于沙箱实现。
它仍是有限成功证据，不是训练质量的完备证明；同模型担任 Agent/User/Judge 有相关性偏差。

循环工程的版本单位是“冻结源码的一次实验”，不是在同一源码上反复抽样。每个候选版本先使用相同
`--experiment-seed` 做配对回归，再用新 seed 测量分布外良品率。一次候选版本只修复一个可复用根因；
不允许在运行中的实验目录对应源码上继续修改。最终候选冻结后由调度器自动运行生产留出集，
每个新构建还必须通过 `task_lineage.json` 证明生成任务与沙箱内运行任务字节一致，且没有发生旧式绝对
manifest 路径迁移；否则以 `TASK_LINEAGE` 失败，在评分和 live rollout 前终止。
`history.json` 的最终停止原因只有 `production_prepared_for_agentic_rl` 才表示生产准备认证通过；
`holdout_target_met` 仅属于 pilot 门禁。

生产认证通过统计、真实性和不可变性门禁后，还必须成功导出并复验
`training_materials_bundle/`。该包使用相对路径和内容摘要，可直接迁移到后续 RL 数据转换/训练系统；
仅存在指向本机输出目录的清单不再足以触发生产准备停止原因。包验证器会从原始 rollout 重建
transition 投影并验证 episode/step/终止关系；它证明训练素材可摄取，不宣称已经执行 RL 训练。
Bundle v17 内置 `certification.json`、`dataset_card.json`、`experiment_contract.json` 和机器可读
`consumer_contract.json`；后者固定
transition JSON Schema、记录顺序、任务生成来源、环境重建入口以及 policy/trainer 可见性边界。数据集卡的构成统计、模型偏差、用途限制与
内部使用边界会同实际 transition 交叉验证，不能通过重新计算 manifest 哈希伪造更宽泛的认证结论。
每条 transition 还必须通过独立消费者记录校验；消息和 transition 都采用精确字段集合，不能夹带未声明的
trainer-only 元数据。
环境交付同时要求 `portable_build_context_ready`：平台规范化 `.dockerignore`，递归清点全部 Docker 重建
输入，并拒绝凭证文件、运行数据库、reviewer 日志、符号链接和未声明隐藏路径进入构建上下文。
导出的验收 rollout 会被机器标记为认证证据而非已认证的直接策略优化目标；生产准备认证覆盖环境的
新鲜 rollout 采集能力，不替代下游算法对轨迹用途的审批。
每次 Agent 与 User Simulator/Reward Judge 的外部响应还会保存实际返回模型的聚合 provenance 和响应 ID
哈希，避免仅凭请求时配置的模型别名声明运行身份；三类实际响应模型还必须落在实验启动前冻结的 allowlist 中。
生产 profile 还必须传入 `--bundle-signing-private-key` 与 `--bundle-trusted-public-key`（或在 `.env` 中设置
`ENVFACTORY_BUNDLE_SIGNING_PRIVATE_KEY`、`ENVFACTORY_BUNDLE_TRUSTED_PUBLIC_KEY`）。私钥应由组织密钥管理
系统保管且不得提交到仓库；实验清单只冻结公钥身份。未签名包和只携带自声明公钥的包均不能通过生产门禁。
正式实验获得输出锁后、启动任务前会写入 `production_preflight.json`，以非敏感方式检查 Codex、Docker daemon、
OpenSSL、模型/runtime 配置、签名密钥配对及工作区余量；默认至少需要 10 GiB，可通过
`ENVFACTORY_MIN_FREE_GIB` 提高。preflight 不向模型 provider 发请求，联网授权与可用性仍由运行方负责。
任务会在每个训练类别内按近重复任务家族确定性分层为 80% train、10% validation、10% test；同一家族及
一个任务的所有 episode/transition 共享同一 split。生产包要求每个类别覆盖三个 split，且任务家族不得
跨类别或跨 split，防止模板变体泄漏到下游评估集。

## 2026-09-27 开发诊断

两轮源码冻结的 30 请求开发批次分别保存在
`output/development_task_yield_20260927_frozen30/` 和
`output/development_task_yield_20260927_fix1_30/`。两轮均生成 21/30 个可构建任务，
Wilson 95% 下界约为 0.52，未达到 0.85 的开发目标。第一轮多步 Agentic 为 8/15；
加入公开素材 fixture 和联合类型参数修复后，第二轮多步 Agentic 为 6/15，简单 Agentic 为 9/9，
直接回答为 6/6。两轮 seed 不同，不能把类别变化直接归因于代码改动。

每轮在生成结果出来前按固定哈希规则选定 10 个构建探针。第一轮 6 个生成成功且全部通过构建；
第二轮 8 个生成成功，6 个通过构建、2 个在第三轮修复后仍因验收失败而终止。
两轮固定样本的端到端结果均为 6/10，验证模式均为 `offline_mock`，不含 live rollout。
具体样本、失败代码、模型、离线分数和 Docker 证据见各轮的 `task_yield_summary.json` 与
`build_probe_summary.json`。这些诊断结果不构成生产认证。

针对多步瓶颈的第三轮冻结实验保存在 `output/development_multi_step_20260927_fix2_15/`：
15 个多步请求仅 5 个生成成功，Wilson 95% 下界约 0.152。10 个失败中，3 个公开输入已足够、
3 个上下游工具投影字段不兼容、2 个初始状态已满足目标、1 个素材内容占位、1 个必需对象为空。
事前选定的 5 个构建探针中 3 个生成成功，但只有 1 个在修复后通过离线评分与 Docker 校验；
另外 2 个在第三轮修复后仍于验收阶段失败。端到端结果为 1/5。该实验表明多步任务需要
 跨任务描述、业务数据、工具 schema、声明式实现和可执行验收的字段契约，单点 fixture 修复
 不足以提升生产良品率。

第四轮冻结实验保存在 `output/development_multi_step_20260927_fix3_15/`。针对上轮问题加入
跨工具投影缺口审计与有限修复后，15 个多步请求有 9 个通过生成门禁，Wilson 95% 下界约
0.357；源码摘要与批次启动时一致。6 个最终失败包括 2 个工具链 schema 不兼容，以及缺少
真实 capture/$ref 依赖、缺少高风险主题权威来源、使用未配置的外部能力、写入工具参数无法编译，
各 1 个。事前按哈希选定的 5 个构建探针中只有 2 个生成成功；两者均经最多 3 轮修复后通过
独立语义审查、10 分离线评分和 Docker 无网络冒烟测试，端到端为 2/5。第 3 轮修复后仍可
执行一次只读完整验收，因此状态中的 `attempt=4` 不代表第 4 轮代码修复。两轮 seed 不同，
不能把良品率变化直接归因于投影修复；本轮仍是开发诊断，未执行 live rollout。

后续构建中，`develop_sandbox_with_agent.sh --max-attempts` 对每个开发节点及每个独立缺陷
分别计数，不再用整个沙箱共享的修复轮数。每轮仍先完整验收、只修一个有证据的根因，再执行
语法、测试、缺陷专项验证和完整验收；同一缺陷达到上限后本候选失败。缺陷计数写入
`defect_attempts.json`，`--resume` 保留计数，从零重建则清空。该变更需要新的冻结批次验证
实际良品率，不能回填第四轮结果。

第五轮冻结实验 `output/development_multi_step_20260927_fix4_15/` 把编译后的 `capture/$ref`
schema 检查前移到成功轨迹生成。15 个多步请求仍只有 9 个生成成功，最终失败不再出现
`TASK_TOOL_CHAIN_SCHEMA`，但数据约束、空表、占位参数、未提供来源和 stateful 写入契约
仍各有失败。事前选定 5 个构建探针，4 个生成成功、3 个被构建脚本判为通过（均为 10 分
离线评分及 Docker 冒烟通过），原始端到端为 3/5。额外奖励审计证明其中 2 个不可计为
可信合格样本：task-2 的过程指标对一份字面 CSV 给分，调换有效数据行后分数由 1.0
降为 0.0；task-5 的回答质量依赖由任务实现替换 `EpisodeStore.replay` 后注入，未声明于
原始奖励契约。审计后的端到端为 1/5。task-1 的成功场景引用了上游未投影的字段，
导致验收 400；构建 Agent 无法通过修改受保护契约修复。独立语义审查曾放行上述奖励
问题，因此下一轮必须增加确定性奖励反事实和捕获路径门禁，不能只依赖审查模型。

第五轮后的生成侧修正已针对这三个可复现根因：捕获路径必须在上游输出 schema 中声明；
非 stateful 任务的 outcome 奖励必须显式依赖最终回答；对长篇字面工具参数给分的过程指标，
仅在后续指标捕获其解析结果时才能删除并重新归一化权重。最终任务组装和构建预检都检查
奖励契约。旧 task-2、task-5 分别触发 `TASK_LITERAL_PROCESS_REWARD` 和
`TASK_TERMINAL_REWARD_UNDECLARED`，旧 task-3 通过该检查。这是门禁回归证据，
尚未证明新生成批次的合格率改善；下一轮需要冻结源码做同 seed 配对回归，
并对通过样本执行奖励反事实、离线评分和 Docker 验证。

离线和 live 奖励校准新增 `wrong_final_answer` 反事实：保持成功轨迹中的工具调用不变，
仅把最终答复替换为与任务无关的内容；非 stateful 任务必须按已声明的终局结果权重出现
足够的奖励下降，认证器也复核同一条证据。旧 task-3 的离线奖励由 1.0 降到 0.3，
符合其 0.7 的终局权重。单表数据生成现在会在跨表阶段前定向修复空数据、重复主键、
非空字段和 CHECK 约束错误。第六轮同 seed 开发批次已预备指纹和探针，但因外部模型
数据外发与 API 费用的自动审批拒绝，未启动、没有新的良品率证据。
同 seed 批次完成后可运行 `scripts/diagnostics/compare_development_batches.py BASELINE CANDIDATE`；
比较器先核对逐请求 seed/路由身份和预选探针，再分别报告生成通过状态迁移、失败类型迁移、
原始构建迁移和有审计证据覆盖的端到端迁移，避免只展示某一类缺陷下降。
