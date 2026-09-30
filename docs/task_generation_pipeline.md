# 任务生成流程

## 默认主链路

`./scripts/run_pipeline.sh` 运行 Code Agent 任务生成、沙箱构建和最终训练验收。
`examples/generate_task.py`、`scripts/loop_experiment.py` 与 `TaskGenerator` 的
默认生成后端统一为 `code_agent`。作者、构建、独立审查使用 GPT-6-luna；
真实图谱、编译前置验证、奖励门禁和 live 验收仍按既有契约执行。
使用 `./scripts/run_pipeline.sh --help` 查看默认值与例子。

## 规格驱动的多步任务生成（显式兼容后端）

显式传入 `--generation-backend spec` 可生成原型多步任务。当前支持三个版本化原型：

| 原型 | 意图 | 必要业务链 | 结果真值 |
| --- | --- | --- | --- |
| `lookup_join_sum` | calculate | 查询私有编号 → 关联查询明细 → 本地求和 | 当前数据的动态关联汇总 |
| `lookup_update` | modify | 查询申请及状态 → 按私有编号更新 | 目标行状态与无关记录保持性 |
| `constraint_create` | execute | 查询申请约束 → 查询满足数量/价格条件的供应商 → 创建预订 | 唯一目标预订、外键与无关记录保持性 |

流水线顺序为实例化规格、编译工具和奖励、执行参考轨迹及反例、渲染题面、物化交付物。题面采用确定性文本模板，规格生成不调用模型；仅生成多步任务时不需要 Neo4j 或模型凭据。当前原型覆盖线性依赖、唯一候选约束选择和单次写入，尚未覆盖多分支规划、外部 API 与自定义媒体能力。数据与名称变化不应被当作新的推理结构。

```bash
.venv/bin/python examples/generate_task.py \
  --generation-backend spec --training-category multi_step_agentic \
  --count 5 --seed 20260930 --output output/spec_batch
```

省略 `--task-prototype` 时多步样本轮换三个原型；可显式指定原型或兼容的 `--task-intent`。不支持的组合会输出结构化缺陷并立即失败，不会静默换意图或回退到模型生成。直接回答、单步任务仍使用模型流水线；比较旧版多步生成时显式传 `--generation-backend legacy`。

所有生成交付物仍在批次根目录 `task/task-N/`：`source_spec.json` 保存私有源规格，`task.json` 保存运行契约，`spec_verification.json` 保存编译器摘要、源规格摘要、阶段耗时与前置验证证据。失败时写 `generation_defect.json`，包含责任阶段、错误码、证据与修复方向；确定性的规格/编译错误不进入候选重试。`sample_manifest.json` 对程序生成记录空模型提供商和零调用轨迹，使用编译来源证明，不伪装成模型生成。

构建前执行真实声明式运行时与奖励门禁，检查成功路径、错误答案、无工具、跳步、逆序；只读汇总额外检查新增明细、修改各目标数值、修改关联键和无关数据变化，状态任务额外检查错误终态和无关写入，创建任务还检查重复业务键写入被原子拒绝。固定目标原型的用户 FSM 在 goal_satisfied 时直接进入终态；结果是否真实完成仍由状态/数值奖励独立核验。汇总原型使用封闭的单数值输出格式，额外未校验声明不获得结果奖励。工具语义、数据约束、输出契约与构建门禁继续独立验证。

实验入口 `--sample-build-budget` 默认 `0`，不设置样本总时长硬上限。5 分钟仅作为性能统计目标，超过该目标不影响质量判定；显式传入正数才启用总预算，耗尽时标记 `SAMPLE_BUILD_BUDGET`。Code Agent 生成阶段另有 `--code-agent-timeout`（默认 600 秒）的防卡死保护，构建阶段也独立配置超时。live rollout 单独计时。离线通过只证明可执行性和相应门禁，最终 Agentic 良品必须另有 live 证据。

沙箱继续复用共享声明式运行时和 Docker 缓存层；这些原型不需要 Code Agent 编写业务 handler，但仍执行独立语义审查、Docker/HTTP 验收和 live 验证。当前没有将逐样本镜像构建改为共享镜像挂载，也没有把原型审查缓存当作逐样本验收证据。

工具执行、事务持久化、目标真值、阶段检查点及评分证据的最新约束见 [一致性改造说明](runtime_integrity.md)。

## 门禁证据与评分口径（2026-09-30）

- 语义审查覆盖 `business_tools`、`reward`、`user_simulator`、`runtime_contract`。高危奖励意见可提交声明式 `reproduction`，由外层在独立临时数据库执行；不执行模型提供的代码或 shell 命令。目前支持 `no_tool_reward` 和 `invalid_goal_reward` 两种不变量，模型裁判型奖励及其他语义意见无法由这两个探针自动裁决。
- 没有复现、复现不可用、覆盖不足或被探针反驳的报告，最多进行一次带证据复核；仍不明确记为 `REVIEW_UNRESOLVED`，单独归类为 review，候选仍不通过，但不消耗业务代码修复次数。反驳一个例子不会自动给整个沙箱生成通过分数。
- 变异测试只有独立外层断言或直接观察到平台变异行为才计为 killed。生成的 acceptance 非零退出本身不再构成 kill 证据；超时与服务启动失败按 infrastructure_error 处理。结果写入 `mutation_report.json`。
- `training_readiness.json` 记录无关状态保护的执行探针：成功轨迹后改变受保护字段，确认目标仍满足而奖励不超过 0.2；约束拒绝不冒充奖励验证，没有适用变更时明确标记 not_applicable。
- 隐私扫描只对完整的平台 `payload.tool_call_id=call-<32位十六进制>` 按不透明标识处理，凭据扫描仍执行，真实身份证号和自由文本不豁免。
- 参数反例携带合法必填字段，仅加入一个未声明字段，并检查实际错误原因。状态前置条件按表、行选择器、字段和值类型检查；旧的无作用域文本 delta 需要重新编译为结构化目标。
- 构建全部门禁通过后，外层保存 `.node_checkpoints/qualification.json`。离线评分可以复用代码、构建输入及证据文件哈希完全匹配的结果；任意绑定项变化都会失效。当前缓存保守地整体失效，尚未实现按修改文件精确选择单项门禁。`score_sandbox_offline.py --fresh` 强制重跑，生产复验使用此参数。
- 任务分只表示构建前资格；沙箱分只表示离线资格；live 分别报告环境健康、可解性见证和指定 Agent 的成功率。默认 live 最低成功率统一为 2/3，可显式提高；历史实验的显式阈值保持原值。过程奖励当前仍是完整因果链门控的稀疏奖励，不宣称已实现阶段性稠密奖励。
- 生产策略 1.9 不再强制成功 live 中混入失败 episode；负例由独立反事实证据承担。新规格任务显式声明 fixed_goal 对话策略，其他任务保留 interactive 覆盖要求。
- 对 fixed_goal 且所有奖励均可声明式执行的任务，平台先验证已提交答案、业务目标和完整因果奖励达到满分，再应用脚本已有的 goal_satisfied 终止迁移；终止确认不调用模型，轨迹明确标记 decision_source=executable_goal。未完成的任务仍使用配置的用户模拟模型，并向其提供 Agent 已执行的工具证据。模型裁判和互动任务保留原有语义判断。这样不会让模拟器在任务完成后要求违反题面格式的额外材料。

模型配置采用 `LLM_*`（任务生成）、`ROLLOUT_LLM_*`（live Agent）、`SANDBOX_LLM_*`（用户模拟和 Judge）三个角色。当前本机三个角色均已切换 deepseek-flash；规格生成仍零调用，构建/独立审查仍由 Code Agent 配置控制。同一服务商的 Agent/Judge 可用于 pilot，但不满足现有生产策略的裁判独立性要求。

## 兼容的模型生成路径

以下描述适用于 `--generation-backend legacy`，以及仍使用模型生成的其他训练类别。

`examples/generate_task.py` 先从 Neo4j 的 `Scene` 节点抽取 `SAME_EVENT_ELEMENT` 路径，长度在 0 到 `--hops` 之间；若当前长度没有可用路径则逐级回退。生成器从节点名称和别名选取最多三个关键词，再按训练路由选择兼容的 `task_intent`，交给分阶段外部 LLM 生成用户任务。环境业务数据由后续流水线生成。每个阶段进入下一步前执行 JSON 结构校验。

采样节点、跳数和关键词保存在任务的 `artifacts.graph_context`；任务描述和初始用户请求必须保留至少一个采样主题词，并使用真实业务措辞。生成器拒绝脱离图谱主题、在用户可见文本中提及沙箱或图谱内部术语、把公开记录原样复制进私有业务表，以及把前序工具结果的求和、模拟、预测或格式化伪装为新的业务工具。多步只读任务的后续工具必须声明新的私有数据访问；单纯引用前一步结果不能证明第二个工具有必要。

代码按职责组织：`task_pipeline.py` 只保留领域阶段编排和跨阶段契约；`pipeline_stage.py`
负责单阶段 LLM 调用、缓存、结构校验、错误反馈与重试；`user_simulation_contract.py`
负责用户画像/FSM 协议校验、确定性剧本和运行时输入物化。`TaskGenerationPipeline` 保留原有
方法入口，因此调用方和已生成任务契约不受模块拆分影响。

当前任务意图包括：`query`（查询）、`explain`（解释）、`compare`（比较）、`recommend`（推荐）、`diagnose`（诊断）、`modify`（修改）、`execute`（执行）、`plan`（规划）、`summarize`（总结）、`create`（创建）、`extract`（提取）、`classify`（分类）、`validate`（验证）、`audit`（审查）、`calculate`（计算）、`estimate`（估算）、`schedule`（排程）、`monitor`（监控）、`troubleshoot`（排障）、`transform`（转换）、`decide`（决策）和 `simulate`（模拟）。只有 `plan` 意图允许生成策划或执行方案，其他意图必须保持对应的目标和输出形式。可通过 `--task-intent` 指定意图，不指定时随机选择。

```text
Scene 路径
  0. keyword_quality_filter（规范化、低信息/高风险/乱码过滤，最多 3 个）
       ↓
主题/关键词
  1. task_description
  1b. environment_plan（stateless/reference_data/stateful/external_capability）
  2. environment_data（仅 reference_data/stateful 生成业务实体和表）
  3. user_profiles 与 user_scripts（画像使用多样化固定脚手架，剧本参考任务目标）
  4. agent_actions
  5b. capability_plan（区分环境操作、Agent 推理和最终回答）
  6. openai_tools（只接收环境操作）
  6b. tool_implementation_specs（为可声明化的单表只读查询生成编译规格）
  7. reward_key_steps
  8. observations_rewards（基于关键步骤生成）
  8b. metric_implementation_specs（为规则指标生成可由共享运行时执行的 DSL）
  9. acceptance_contract（EnvFactory 独立业务验收基线）
  9b. acceptance_executable_scenarios（确定性基线 + 可选语义增强）
  10. training_contract_consistency（题面、数据、动作、工具、指标、fixture 跨阶段一致性）
  11. task_readiness（生成阶段硬门禁与复杂度重算）
  12. TaskSpec compiler（环境 archetype、能力 DAG、状态增量、工具输出契约）
       ↓
     task.json
```

关键词进入 LLM 前先执行确定性质量过滤：剔除纯数字、URL、乱码、控制字符、低信息泛词和在无来源任务中容易诱发高风险事实生成的词；每个 Scene 优先从名称及别名中保留可用候选，去重后最多使用 3 个关键词，避免为了覆盖过长图路径而强行拼接互不相关的主题。若整条路径没有可用关键词，则拒绝该样本，不把噪声传入后续 pipeline。

`environment_plan` 只依据任务描述中的明确目标选择运行模式。`stateless` 任务直接处理用户提供的文本或结构化输入，跳过实体、表结构、rows、一致性和持久化文档生成；其 data manifest 明确标记 `environment_mode=stateless` 且允许空 `tables`。`extract`、`summarize`、`classify`、`transform`、`explain` 等意图在没有明确保存、更新或外部实时能力要求时会被确定性收敛为 `stateless`，避免用户模拟中的扩展请求污染正式任务范围。`reference_data` 生成只读资料，`stateful` 生成可持久化业务状态，`external_capability` 描述外部能力边界。

任务描述进入环境规划前先经过独立 grounding 审计，检查任务是否由声明的运行时用户输入、业务资料或工具能力完成，是否要求猜测未提供的价格、成分、属性或排名，以及预期结论是否可推导。审计失败时先修复完整任务描述；连续失败则拒绝生成，避免将隐藏答案或臆造事实带入后续用户模拟和奖励。

计算任务若已在公开材料中交付报价、数值和完整计算目标，且用户没有请求查询内部业务记录或外部专用能力，不能为了满足 Agentic 路由再添加计算工具。任务评分会将这类旧产物判为不具备训练资格。

任务描述同时生成规范化 `public_input`：`initial_user_message` 是 episode 的初始公开请求，
`materials` 保存题面明确引用的文本或结构化输入。若题面提到“用户提供的资料”“以下文本”或
“给定数据”，但没有交付实际材料，任务会在生成阶段被拒绝。`public_input` 会写入 task、
TaskSpec、User Simulator FSM，并由沙箱 observation 和真实 rollout 暴露给 Agent；它不能包含
业务数据库中的隐藏真值。

生成器还接收运行时能力目录。未配置 `SANDBOX_EXTERNAL_CAPABILITY_URL`，且没有有效的
`SANDBOX_EXTERNAL_FIXTURES` 文件时，能力目录不包含 `external_capability`，任务不得依赖实时搜索、
天气、行情或其他外部服务。沙箱构建前会再次执行 buildability preflight；任务契约、数据清单、
外部能力或奖励 DSL 不可实现时直接归还任务生成阶段，不消耗 Luna 构建预算。

数据型环境还执行独立 grounding 审计：业务 rows 必须覆盖任务所需事实；唯一推荐、排序、合规判断或首选结论必须具有唯一且可追溯的决定性证据；成功答案引用的数值、属性和理由必须与 rows 一致。确定性关键词对齐也在该阶段执行，失败会反馈给 `environment_data_consistency.repair`，只重做数据而不重跑已通过的任务描述。动作、工具、奖励和 fixture 同样在各自边界反馈并重试。`reference_data`/`stateful` 只要包含业务表，就必须至少暴露一个读取或操作数据的业务工具，不能把读取私有数据误标为 Agent 自身推理；`reference_data` 若模型遗漏业务工具，会从已验证的数据访问动作生成通用只读工具和绑定。data manifest 的 `environment_mode` 与顶层环境计划保持一致。

写出任务前执行最终训练契约一致性门，重新联合检查题面、requirements、业务数据、动作、工具绑定、奖励指标和成功 fixture。该门不负责掩盖上游错误，而是作为最后一道防线阻止跨任务实体、凭空数值或悬空绑定进入训练集；错误信息会标明污染所在的契约类别，供下一次局部生成或循环工程分析使用。

`acceptance_contract` 同时包含机器可执行的 `executable_scenarios` 和
`argument_probes`。新任务的 mutation 与黑盒验收直接消费这些结构化字段，
不再从自然语言步骤中用正则恢复工具调用；旧任务仍保留兼容解析。

最终 `task.json` 的顶层字段是下游沙箱使用的唯一任务定义；任务文件不包含 `constraints` 字段；`artifacts` 只保存业务数据、用户模拟、工具文件、媒体生成和 Pipeline 版本等文件引用，不重复保存任务、动作、工具或指标。任务生成在工具和奖励校验完成后，确定性生成 `requirements.runtime_interface`：它声明 HTTP 协议、系统接口、每一个 LLM tool 的请求 schema，以及 reward function 接口。沙箱构建脚本将 `task.json` 去除顶层 `actions` 字段后生成只读的 `BUILD_CONTRACT.json`，因此该 HTTP 约定会原样进入构建契约；不补充 task 中不存在的平台约束、能力、接口、评测或动作字段。沙箱只实现 `tools` 中声明的工具，不建立独立的 Trainer Action 注册表。

第二阶段不是一个大提示词，而是按依赖顺序拆成以下独立提示词，每一步都校验输出后才进入下一步：

1. `environment_entities`：识别完成任务所需的最小必要业务实体，只保留需要查询、修改或评测的业务事实。
2. `environment_table_design`：根据最小必要实体设计原子数据库表，只生成表结构，不生成 rows。关系数据遵循 3NF，消除重复组、部分依赖和传递依赖；但 3NF 不意味着机械拆表。优先使用最少数量的表；只有存在独立生命周期、独立查询/更新需求或明确关系时才拆表或建立关联表，静态说明优先作为字段、枚举、JSON 或文本保存。
3. `environment_table_data.<table_name>`：按表并发生成完整、非空、可直接持久化的 rows，单张表失败可以单独重试。
4. `environment_data_consistency`：检查并修正主键、外键、字段完整性、业务关系和任务覆盖度，同时生成业务记录汇总。
5. `environment_data_document`：从已验证的最终表结构和数据确定性渲染供 Code Agent 使用的持久化说明文档，不再调用模型。
6. `environment_media_generation`：仅当第一阶段声明需要媒体时生成媒体数据程序、依赖、入口和输出目录。

这些提示词之间传递结构化结果，业务数据只根据任务描述和任务要求进行模拟。每张表的 schema 和 rows 会分别写入 `schemas/<table>.json` 与 `rows/<table>.jsonl`，`task.json` 通过 `artifacts.data_manifest` 保存文件清单；Code Agent 根据清单读取文件并初始化数据库。实体、表、数据、文档、媒体、动作、工具和奖励阶段各自只负责自己的输出。媒体不是每个任务的必选项，第一阶段根据 `requirements.input_modalities` 判断是否执行媒体生成；媒体生成代码由 Code Agent 在沙箱构建阶段执行，媒体识别和媒体评测不属于任务生成阶段。

生成结果的 `artifacts` 只包含 `data_manifest`、可选的 `media_generation`、用户模拟 manifest、`tools_manifest` 和 Pipeline 版本信息。`tools_manifest.file` 指向独立的 `tools.json`。用户交互通过 Trainer-only 的 `POST /v1/user_simulator` 进入 User Simulator；待训练 Agent 的最终自然语言输出由 Trainer 通过 `POST /v1/agent_response` 提交并持久化为 `final_agent_response`，它不是 LLM Tool。业务工具使用 `POST /v1/tools/{tool_name}`，奖励使用 `GET /v1/reward`。这样最终回答无需伪装成工具，同时成功、失败和噪声验收轨迹可以真实覆盖最终结果奖励。

运行时接口同时声明 Trainer Bearer 鉴权、Agent/Trainer 访问边界、episode 隔离、seed 重置、幂等键、`/v1/replay` 回放、外部 LLM 适配器和 evaluator mock 配置。沙箱构建完成后，EnvFactory 会重新生成独立的外层 conformance；构建流程不会自动连接外部已运行服务，也不会默认启动沙箱。

`acceptance_contract` 由 EnvFactory 根据业务数据、原子动作、工具、关键奖励步骤和指标确定性生成，包含业务场景、数据不变量、工具非法输入、成功/失败奖励样例、数据变异策略和 mutation test 清单。工具与指标均有可执行实现且基线通过结构校验时，直接使用该契约；存在自定义扩展或基线校验失败时，才调用模型补充并重新校验。Code Agent 不能修改该契约；外层验收使用它执行黑盒轨迹、前后数据快照、反事实奖励和实现缺陷注入测试。

用户画像由 EnvFactory 从固定、多样化脚手架生成，结构化描述身份、知识、沟通和决策特征；画像只影响表达与行为，不提供任务业务真值，也不再消耗逐任务模型调用。用户剧本由 EnvFactory 确定性构造为有限状态机：`initial_state` 指向初始状态，`states` 定义用户行为和终止状态，`transitions` 通过 `from_state`、`to_state`、`condition`、`outcome_category`、`should_end` 和 `updates` 描述转移。生成器检查状态引用、可达性、终止路径和变量更新。任务生成阶段不再生成或落盘模拟对话；真实对话只在 rollout 时由外部 User LLM 根据画像、FSM 与实时上下文产生。任务生成 CLI 每次从已有最大 `task-N` 的下一号开始追加，并用原子目录创建支持并发进程。

启用噪声工具时，工具生成阶段至少生成一个噪声工具；默认上限为 3 时至少同时覆盖 `related_irrelevant` 和 `unrelated` 两类。候选工具还要经过独立的反事实有用性审查：凡是能提供原因分析证据、关键事实、比较依据、验证手段或排查资料的工具都不能作为噪声，会从候选集合中自动剔除；若候选集合被全部剔除，则注入一个不读取或修改业务状态的通用无关工具，避免正确的审计结论导致整项任务生成失败。噪声调用惩罚由共享运行时使用 `trajectory.events` 和 `none_tool_calls` 运算符确定性执行，不交给外部 LLM 判断。

用户画像和用户剧本不内嵌到 `task.json`，而是写入任务目录内的便携相对路径 `data/user_simulation/`，并由 `user_simulation_manifest` 引用；业务 fixture 同样固定在 `data/business_data/`。沙箱构建保持这些相对 root 不变，避免同一任务在生成与运行阶段产生两个不同契约身份。运行时 User Simulator 加载画像和状态机，每轮通过外部 `RuntimeLLMClient` 根据完整实时对话、当前画像、状态、变量和合法出边生成下一条用户消息。协议固定八类结果：`goal_satisfied`、`information_required`、`user_correction`、`user_rejection`、`user_acceptance`、`agent_off_topic`、`agent_premature_completion`、`unrecognized`。前五类走正常转移，后三类走不推进业务状态的有界恢复；超过恢复上限后以 `unresolved_dialogue` 结束。
User Simulator 只能依据用户可见对话和公开证据判断结果，不能因为看不到内部工具 trace 而拒绝一份
可核查的答案。JSON schema 与 FSM 跨字段语义在同一个 LLM 重试边界内校验；非法 outcome、错误
transition 或 match_status 冲突会连同具体错误反馈给模型修复，耗尽有界重试后才进入保守 fallback。

`agent_actions` 阶段只根据任务目标、环境计划及业务实体、表和字段拆解 Agent 操作，不读取模拟对话。其后由 `capability_plan` 逐项分类为 `environment_operation`、`agent_reasoning` 或 `agent_response`。只有必须读取或修改沙箱私有状态、调用外部系统或使用确定性专用能力的 `environment_operation` 可以生成工具；比较、分析、选择和最终回答保留给待训练 Agent。每个动作必须提供 `atomicity_rationale`、`inputs`、`outputs`、`preconditions`、`effects`。后续 `openai_tools` 同样不读取对话，只接收通过资格判断的环境动作。

动作拆解阶段本身只接收业务模型，不接收业务实例数据。业务模型仅包含实体、表、字段、关系和约束定义；业务数据行、数据文档、隐藏真值、内部记录、具体字段值和数据库 ID 不会传入该阶段。这样可以让动作拆解判断“需要查询什么类型的数据”，但不能把某条真实模拟记录泄露到动作或工具定义中。

奖励生成分为两个阶段：`reward_key_steps` 先根据任务目标、业务模型、原子动作和工具梳理完成任务真正必需的关键步骤；`observations_rewards` 再读取该列表生成过程、结果和惩罚指标。过程指标的 `target_action` 必须属于 `reward_key_steps`，不会因为工具存在就自动获得过程奖励；如果没有关键工具步骤，process 指标可以为空。

每个 `rule-based` 指标必须具有对应的受限 DSL `metric_implementations`。无法编译的自然语言规则会被提升为 `model-based/external_llm_judge`，不会以空实现进入沙箱。只读数值计算任务在成功回答生成后，还会尝试编译 `numeric_targets`：公式只允许公开数值常量、按条件读取私有业务行、按条件汇总业务行和受限算术运算；生成器核对公式与成功回答及当前业务 rows 一致，并修改被引用的私有数值验证旧答案失效。该阶段无法给出可复现公式时拒绝当前任务候选。任务输出前还会执行 task-level readiness 门禁，要求工具只绑定环境动作、规则指标实现完整、成功/失败轨迹齐全，并在启用噪声工具时要求 `noise_selection` 轨迹。成功场景使用独立生成的完整回答 fixture 并断言 `reward >= 0.5`；失败场景断言 `reward <= 0.2`；噪声场景断言 reward 非正。实际复杂度根据业务工具、关键步骤和指标数量重新计算。

成功 fixture 必须绑定信息最完整的一段真实运行时对话，忠实保留用户提供的实体、数值、价格、规则和格式范围；不得用另一批示例替换，也不得编造缺失事实。需要同时比较最终回答与业务数据的规则无法由单路径 DSL 表达，会提升为读取 `business_data` 与 `final_agent_response` 的外部语义评估。所有任务最终使用统一 observation schema，避免模型返回空结构或自创字段布局。

共享运行时将 `final_agent_response` 保存为原始字符串，因此生成器禁止使用 `$.field` 访问虚构的文档字段；需要判断文档条目数量、结构或语义完整性的指标会提升为外部语义评估。噪声工具调用只保留一个确定性的 `penalty_noise_tool_usage`，其通过分数固定为 `0`、违规分数为 `-1`。readiness 同时检查指标 source/path 类型、正负奖励方向和噪声惩罚唯一性。没有业务工具而只有噪声工具的任务标记为 `task_readiness.training_profile=tool_abstention`，不会被误归类为多步工具调用训练样本。

## 分层训练任务路由

批量生成默认按 `direct_response=20%`、`simple_agentic=30%`、`multi_step_agentic=50%` 分配任务。分配使用最大余数法，因此每批的类别数量确定，随后随机打散以支持并发生成。`--training-category` 可固定单类，`--training-mix` 可覆盖比例。

- `direct_response`：stateless、无业务工具，训练正确不调用工具。
- `simple_agentic`：恰好一个不可替代的业务工具。
- `multi_step_agentic`：至少两个业务工具，目标是依赖链或条件分支。

路由与任务意图采用显式兼容矩阵，避免先随机意图再为不合适的语义机械补工具：

| 路由 | 兼容意图 |
| --- | --- |
| `direct_response` | `explain`、`summarize`、`extract`、`classify`、`transform`、`create`、`calculate`、`compare` |
| `simple_agentic` | `query`、`validate`、`calculate`、`estimate`、`modify`、`monitor`、`execute`、`recommend` |
| `multi_step_agentic` | `query`、`execute`、`plan`、`diagnose`、`modify`、`audit`、`schedule`、`monitor`、`troubleshoot`、`simulate`、`validate`、`decide`、`calculate` |

未指定 `--task-intent` 时，从当前路由的兼容池抽样；指定意图但未指定路由时，只在兼容路由之间按训练集比例分配；同时显式指定不兼容组合时，在调用模型前直接报错。`explain` 和 `summarize` 因而只进入直接回答路由，不会再被机械改造成多步工具任务。兼容意图列表同时写入 `training_contract.allowed_intents`，供下游审计。

评分器按路由分别应用硬门槛；直接回答任务不会因缺少业务工具失败，而 Agentic 路由必须满足对应工具数量和真实依赖。对于只读数据任务，能力说明若表明后续工具只加工前一步已得到的数据、未声明新的环境访问，该任务不具训练资格。数值计算任务若结果仅由模型判定、没有确定性结果指标，会在奖励维度扣分，不能仅凭完整的提示词获得最高档。沙箱构建后还会修改私有数值输入并沿用原答案，检查奖励是否下降；只校验工具结果随数据变化不足以证明奖励正确。报告的 `score` 是有效分数，任一资格硬门禁失败即为 0 分；`raw_score` 保留门禁前的诊断分，维度得分和 `eligibility_failures` 保留具体原因。阶段缓存命中时重新执行该阶段的语义与结构校验，无效缓存会重新生成。阶段日志的 `duration_ms` 是含重试的累计耗时，`attempt_duration_ms` 是本次尝试耗时。

文件级任务评分在沙箱构建前加载真实业务行，执行工具链与奖励契约静态检查，并预演只读成功轨迹中的声明式工具。工具结果取值路径为空、工具执行失败，或过程奖励从有不同业务标识的多行结果中按固定位置捕获目标记录时，任务直接失去训练资格，报告保留具体指标和工具路径。构建预检仍保留相同检查作为独立防线；评分通过不代替沙箱验收和真实模型评估。

每个任务还会写入机器可读的 `training_contract` 和版本化 `task_spec`。前者声明训练路由、工具数量、验收场景和 `sandbox_profile`；后者是下游执行的规范化 IR，声明环境 archetype、工具输入输出、前置条件/effect、目标状态增量和能力 DAG。多步任务的依赖先在关键步骤中声明并校验只能引用先前步骤，再由动作到工具绑定确定性编译为 DAG；LLM 验收轨迹不再是依赖关系的唯一来源。沙箱使用统一 runtime，但按 profile 物化不同训练语义：`direct_response` 验证不调用工具，`single_tool` 验证单工具必要性和参数敏感性，`dependent_tool_chain` 验证跳步、乱序和参数损坏。

示例：

```bash
python examples/generate_task.py \
  --user-script-count 3 \
  --output output
```

每个任务最终写入 `output/task/task-N/task.json`；业务数据、用户模拟文件和 `tools.json` 也保存在同一个 `task-N` 目录中。重复运行命令会依次新增 `task-(N+1)`，不会覆盖已有目录。

模型阶段生成路径通过 `--generation-backend legacy` 保留，用于尚未覆盖的任务与对照实验；规格路径见上文。

用户剧本的每个分支都包含布尔字段 `should_end`。User Simulator 每轮消费当前合法分支并输出用户消息与终止状态；Agent 不生成用户终止信号。

观测与奖励设计规则：只保留与任务目标完成强相关的关键过程指标和目标结果指标，不能为每个普通动作机械创建指标；任务无需关键工具动作时 process 指标可以为空，存在多个关键动作时不限制过程指标数量。模型阶段可使用 `hybrid` 关键过程指标，使用精简字段 `target_action`、`evaluation_inputs`、`criteria` 和固定 `condition=llm_expected_tool_call_exact_match`。共享评估运行时根据该指标调用外部 LLM，结合当前 Context、可用工具和 `criteria` 生成期望工具名及参数真值；随后由规则引擎对 Agent 实际工具名和规范化参数进行确定性精确比对，LLM 不直接输出最终过程分数，任务 JSON 也不嵌入完整 prompt 或 expected-call schema。结果指标判断任务目标是否完成或关键业务数据是否达到目标，可使用 `rule-based`、`model-based` 或 `hybrid`。惩罚指标只有在直接影响任务目标时才保留，用于偏离用户诉求、无效循环或业务数据偏离预期；工具选择错误和工具参数错误不作为惩罚项。

规格编译及已完成调用契约编译的过程指标采用 `rule-based`，由实际工具调用、参数引用和依赖门禁计算。

过程指标和结果指标是正反馈，`score_range` 固定为 `[0,1]`；惩罚指标是负反馈，`score_range` 固定为 `[-1,0]`。所有正反馈指标的权重之和独立归一化为 1，所有负反馈指标的权重之和独立归一化为 1；结果指标权重总和必须大于过程指标权重总和。最终奖励使用正负反馈分别归一化后相加的公式：

```text
R = clip(
  sum(w_pos_i * score_pos_i) / sum(w_pos_i)
  + sum(w_neg_j * score_neg_j) / sum(w_neg_j),
  -1, 1
)
```

其中正反馈得分位于 `[0,1]`，负反馈得分位于 `[-1,0]`；两组权重互不干扰，因此最终奖励始终落在 `[-1,1]`。`hybrid` 指标同时包含确定性 `condition` 和外部 LLM 语义 `criteria`。

用户交互不生成 `ask_user` LLM Tool。工具生成阶段只生成业务工具；`POST /v1/user_simulator` 是仅供 RL Trainer 调用的内部接口，接收完整的 `messages` 对话数组，由 User Simulator 返回 `user_query` 和 `should_end`，不暴露给待训练 Agent。`GET /v1/reward` 同样标记为 `access=rl_trainer_only`，只有 Trainer 显式调用时才计算奖励，不暴露给待训练 Agent。
# Code Agent 统一链路（开发中，2026-09-30）

新增显式入口 `--generation-backend code_agent`，模型为 `gpt-6-luna`。
它直接使用 Code Agent 背后的模型设计业务，不生成调用其他模型的程序。
当前默认后端暂未切换；新链路未达到目标良品率前，不以旧原型结果作为验证。

链路：真实 Neo4j Scene 多跳路径 → Luna 编写 `authoring/source.json` →
通用编译器产出 TaskSpec/数据/工具/奖励/参考场景 → 前置执行与可构建性检查 →
独立 Luna 业务集成节点 → 原有沙箱验收、语义审查及 live。
所有任务交付物仍位于批次根目录 `task/task-N/`。

Code Agent 路由要求至少两跳，图谱采样失败不会退化为单节点或固定模板。
支持 direct_response、simple_agentic、multi_step_agentic；业务写入要求类型化目标。
作者可使用 [紧凑规格接口](code_agent_authoring.md)，由平台生成重复的动作绑定、
capture/ref 依赖、过程奖励、reset/reward 步骤和验收断言。
当前作者接口使用声明式工具与规则奖励；更广泛的自定义工具/文本业务奖励仍需验证和扩展。

`code_agent_generation.json` 记录调用、耗时、退出码、事件/源码/输入摘要。
CLI 未暴露模型内部请求次数时 `llm_calls=null`；`agent_invocations` 和
`completed_turns` 单独记录，不能将旧 HTTP 客户端 trace 的 0 当作没有模型调用。
生成来源证据新增 v1.2，携带并验证 Code Agent 证据摘要。

开发验证：首个多步冒烟任务生成约 139 秒，但构建未通过，不计为良品。
首个混合开发批次 `output/code_agent_batch_v2` 生成 2/5：直接回答 1/1、
简单调用 1/2、多步调用 0/2，三例因 150 秒生成预算用尽失败。
此批次是在开发校验器期间运行的诊断批次，不是冻结版本的效果基线。
发现并修复了验收步骤引用缺失、前置忽略原始验收断言、必要数据表未参与
数值奖励表达式的问题；紧凑接口尚需新批次实测。85% 良品率目标尚未实现。

混合开发批次 `output/code_agent_unified_v5` 在取消 5 分钟硬预算后生成 5/5，
原始构建通过 2/5，最终合格 1/5。不能将生成通过率作为沙箱良品率。
两例多步任务被私有数据反事实探针误拒绝：查询条件变异使轨迹不可执行、
候选额度提前耗尽，以及多个结果评分项的合理部分奖励被当作奖励失效。
修订探针后，同一沙箱的 Agentic 门禁均通过；原始结果保留，尚不计入最终良品。
探针现在为后续工具保留候选额度，遍历可执行变异，并在单字段仅影响部分结果时
组合已证实会降低奖励的变异，再重复执行两次；原有奖励下降阈值保持不变。

本批次还发现两例生成契约缺陷：题面要求数组而参考答案/奖励使用字符串，
以及奖励要求 JSON 但公开题面未告知。后者的 live 回答业务事实正确，仍因隐藏
格式要求失败。下一阶段必须建立公开输出契约与奖励的一致性前置检查；
不能把这类失败归因于 live 模型能力，也不能仅靠构建 Agent 修改测试消除。

`code_agent` 作者现在交付 `answer_contract`：JSON Schema 的字段必须与结果奖励
一一对应，参考答案必须符合类型。编译器把 Schema 写入真实用户输入；有限条件
分支的文本标签会完整公开为枚举，单数值奖励的标签和单位也会公开。运行时支持
类型严格的数组/对象字面量和动态数组，禁止以包含几个关键词代替最终业务结果。
前置阶段在独立临时目录重建脚手架，执行与最终构建共用的契约/工具检查和 Agentic
门禁；父进程重复检查，不采信作者修改后的预览应用。零工具仅对 direct_response 合法。

冻结开发批次 `output/code_agent_unified_v6` 生成 5/5，构建通过 3/5，最终合格 2/5：
simple_agentic 2/2，multi_step_agentic 0/2，direct_response 0/1。生成+构建耗时
约 240–469 秒；5 分钟不是硬门禁。批次中的三类阻断分别为隐藏的决策大小写标签、
只读奖励绑定私有 ID，以及旧脚本拒绝空工具列表。直接回答奖励还存在关键词捷径。
上述格式与空工具问题已在批次结束后修复，原始结果不覆盖、不追记为成功。

只读奖励的关联缺陷有同目标正向反事实证据：保持用户查询 Cedar Veil 不变，调整
目录名称对应的 ID 后，三步工具链返回正确的新记录，但正确答案仅获 0.2。
证据位于 `diagnostic_same_goal_rebinding.json`。下一阶段需要把奖励依赖绑定到公开
目标与查询链，并检查变化后正确答案仍能得分；仅验证旧答案失分是不够的。
状态写入可用实验入口 `--generation-intent modify` 单独覆盖，目前尚无新链路的
代表性状态写入批次和 ≥85% 最终良品率证据。

开发批次 `output/code_agent_unified_v7` 原始结果：请求 5、生成 4、构建通过 3、
最终合格 2（一个简单调用、一个多步调用）。三处失败分别为：生成 Agent 返回
未修复的 JSON；零工具直接回答任务被要求杀死工具变异；简单查询要求未公开的
精确私有分类值，live 仅 1/3 成功。原批次已结束，未追溯修改其合格统计。

现已增加父进程候选验收反馈：同一图谱、路由和业务目标最多一次定向修复，
两次 CLI 调用共享安全超时，完整保留尝试源码、缺陷、事件摘要和用量。
仅候选 JSON/编译/前置验收缺陷进入该修复；传输错误、未完成模型回合和保护
文件变更直接失败。精确字符串查询参数须来自公开输入/参数契约或前序 capture，
防止依赖私有标签猜测。直接回答且确实无工具时，工具变异标记为不适用，
奖励与鉴权变异仍必须通过。原直接回答样本在独立副本上通过修订门禁；这只是
缺陷回归证明，不计为全链路重新合格。新冻结批次和状态写入覆盖仍待实测。

冻结开发批次 `output/code_agent_unified_v8`：生成 5/5、构建通过 3/5、最终合格
1/5。未达到 85%，也不能把生成通过率提升称为整体效果提升。失败揭示了更深的
契约缺口：工具未暴露奖励所需的私有容量；“首两个行程点”是否包含开场有歧义；
构建测试误算部分奖励后修复 Agent 错判根因；直接回答用自证正确的布尔值替代
实际地质比较内容。原始结果与逐例诊断完整保留。

已补原始 reward lookup/aggregate 的必要字段可观测性前检，原容量样本会在
生成阶段被拒绝。它只证明字段有工具暴露，不证明所有行可达或业务语义正确。
运行中的 Code Agent 调用计数也会立即写入证据；紧凑 select 的 result_field
采用运行时已有的 records 默认值。后续重点是完整数据依赖、公开业务选择规则、
独立语义审查前置及状态写入覆盖；不继续仅凭结构测试通过就宣称任务已合格。

生成阶段现已加入独立只读 Luna 源码审查，位于确定性前检之后、TaskSpec 交付之前。
审查结果有模型、候选/事件摘要、用量和检查依据；作者与审查回合均计入真实调用
次数。语义缺陷回到原 source 定向修复，仍最多一次，之后重新编译和审查；审查
基础设施失败不触发作者修改。首个可执行版本的业务目标作为修复约束随审查保存。

真实诊断保留在 `output/code_agent_source_review_diagnostic`：原 v8 直接回答的
“布尔自证”缺陷被前置审查拦截；原 v8 合格简单调用样本在明确公开契约优先后
通过审查。首次审查曾因私有 rubric 的额外输出要求误拒绝，原证据也保留。
这些是定向诊断，不是新批次良品率。构建计划同时提供奖励组件、权重、答案字段，
要求测试按组件判断部分奖励，避免把错误总分断言误当成业务实现缺陷。
