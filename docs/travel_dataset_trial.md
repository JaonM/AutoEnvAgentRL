# Travel & Tourism 数据集优先生成试验

## 输入与数据核验

- 来源：[Kaggle: Travel & Tourism](https://www.kaggle.com/datasets/madhavw/travel-and-tourism/data)，版本 1，文件 `Travel And Tourism.csv`。
- 当前本地原始文件：`data/sources/kaggle/madhavw/travel-and-tourism/v1/raw/Travel And Tourism.csv`；SHA-256：`64bd52ba65f2afbdcfc3f81ce3fd1766c063d80ac8a5c419744501abcd9ab552`。同目录的 `source_manifest.json` 保存来源、版本、许可和文件校验值。
- 1000 行、29 列；所有列无空值；`Booking_ID` 唯一；无完全重复行。26 个目的地城市，状态为 Completed 772、Cancelled 141、Confirmed 79、Pending 8。`Total_Trip_Cost` 范围 11000 至 1639200。
- 原文件包含顾客姓名、ID、年龄及评论。本试验的工具业务数据只投影 `Booking_ID`、`Destination_City`、`Booking_Status`、`Total_Trip_Cost`，不把顾客字段写入沙箱。
- 授权标注为 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。当前结果限本地试验；商业训练、发布数据衍生物或对外分发前，需要确认相应权利或改用许可合适的数据集。

其他公开 Kaggle 数据集可用 `python3 scripts/diagnostics/download_kaggle_dataset.py owner/slug` 下载。`https://www.kaggle.com/datasets` 是目录页，需要先选定具体数据集；下载器按版本保存到 `data/sources/kaggle/owner/slug/vN/raw/`，默认限制压缩包和解压后数据各 1 GB。下载本身不会自动生成任务：不同数据集仍需检查字段、许可、隐私、工具操作与可验证奖励。本试验可把上述 CSV 路径传给 `probe_travel_dataset.py --source`。

## 三类任务的适配性与样本

| 类型 | 适配判断 | 数据与任务 | 源数据答案 | 奖励核对 |
| --- | --- | --- | --- | --- |
| 直接回答 | 适合；公开输入需提供足够记录，避免隐含查询 | 给出原始预订 1、2 的费用，比较谁更便宜及差额；无需业务工具 | 预订 2，差 58200 | 最终回答须同时命中预订号与差额；调用噪声工具扣分 |
| 单工具 | 适合；唯一预订 ID 支持确定性查询 | 内部表取原始前 10 行；查询预订 1 的目的地、状态、费用 | Leh、Completed、115000 | 正确工具调用与三个事实分别核验；错误参数、空答、噪声工具均有反例 |
| 多步工具 | 适合；城市和状态支持依赖前序结果的过滤 | 从原始记录取前 20 条 Rishikesh 及 10 条其他城市；先查预订 5 的城市，再用该返回值筛选同城 Completed 记录并找最低费用 | Rishikesh、预订 130、11500 | DAG 的第二次查询参数必须引用第一次结果；核验最低费用与预订号，跳步、逆序和错误参数均有反例 |

单工具和多步样本使用原始数据集的**确定性子集**，所有业务行及取值均来自源文件，没有合成记录。子集选择与源文件哈希写在各自的 `source_selection.json`；独立答案键与奖励覆盖检查写在 `dataset_oracle.json`。子集试验可验证任务形态与运行链路，尚未验证千行全量载入时的规模表现。

## 生成路径与运行验证

试验脚本 `scripts/diagnostics/probe_travel_dataset.py` 从指定 CSV 生成任务描述，直接传入原始业务行及确定性答案键。`TaskGenerationPipeline.generate` 因此跳过知识图谱关键词采样、描述模型生成及业务行合成，从数据审计、工具、奖励、接受场景等后续阶段继续。奖励模型提出指标，源数据答案键写入语义结果指标；生成后另用独立 oracle 检查答案覆盖和多步依赖。

候选任务：

- `output/dataset_trial_travel_tourism_v1/direct_response_final/task.json`
- `output/dataset_trial_travel_tourism_v1/simple_agentic/task.json`
- `output/dataset_trial_travel_tourism_v1/multi_step_agentic_corrected/task.json`

前两类直接通过候选生成和数据 oracle。多步候选首次生成的奖励漏掉最低费用与预订号，虽通过项目构建性检查，却未通过数据 oracle；`multi_step_agentic_corrected` 是将已生成候选的结果指标按源数据答案键确定性修正后的样本。另一次重新生成在能力计划阶段失败，故本试验不能据此声称多步生成稳定。

三个候选均已通过任务构建性检查。可执行沙箱目录分别为 `build_direct_response_fixed`、`build_simple_agentic`、`build_multi_step_agentic`，位于同一输出根目录。三个沙箱均完成本地构建与独立语义审查；离线可执行评分均为 10/10，汇总在 `output/dataset_trial_travel_tourism_v1/offline_score_summary.json`。沙箱使用 `gpt-6-luna` 开发及审查。运行验收采用离线模拟评估器：正确轨迹为 1.0，空答为 0.0；单工具无工具调用为 0.0；多步跳过或颠倒依赖步骤为 0.0。直接回答样本曾暴露“空答仍有结果奖励”的共享门控缺陷，现已修复。

## 当前结论与下一步门槛

该方式**可用于生成并本地验证三种任务原型**，单工具与多步工具样本有真实业务读取和可执行的依赖链。当前证据不支持“稳定生产合格 Agentic RL 训练沙箱”的结论：样本量每类仅一个，模型生成阶段仍有遗漏与失败；验收使用 `offline_mock`，没有独立 Rollout Agent 与 User/Judge 实际端点的在线轨迹；多步业务行只试了 30 条；本数据集的许可限制也需要解决。

下一轮应把源数据映射、答案键计算、奖励覆盖和依赖边校验做成通用的确定性输入门禁，再批量试验不同记录与任务模板，统计生成成功率、构建成功率、奖励反例通过率与真实在线 rollout 成功率。进入正式训练前还需核验数据权利与真实端点评估。

## 口语化描述迭代（2026-09-28）

前一版 `probe_travel_dataset.py` 把三类英文测试式描述直接写为用户消息。现在先用数据确定业务目标、公开材料、工具依赖与答案键，再调用任务生成模型只改写用户消息。改写风格从原 `TaskGenerator.STYLES` 抽样，也可用 `--style` 指定；`--seed`、风格和最终描述均记录在 `source_selection.json`。改写后的消息必须保留必要问题，不得公开隐藏答案、引入数据集没有的约束或出现工具实现术语。未通过校验则重试，不能退回测试式模板；完整描述仍经原任务流水线的 grounding、工具、奖励和验收门禁。

本轮样本：

| 类型 | 风格 | 用户消息 |
| --- | --- | --- |
| 直接回答 | 日常对话 | 我手头有两个预订的费用数据，能帮我看看哪个更便宜吗？顺便告诉我差多少钱。 |
| 单工具 | 带背景说明 | 我这边有一笔预订，编号是 1，麻烦帮我查一下它的目的地、预订状态和总费用。谢谢！ |
| 多步工具 | 遇到问题寻求建议 | 我手头有个预订号 5，想先看看它到底是要去哪儿，然后再查查那个地方已经完成的所有预订里，哪一单总费用最低、预订号是多少。麻烦帮我理一下。 |

对应任务位于 `output/dataset_trial_travel_tourism_v2/{direct_response,simple_agentic,multi_step_agentic_final}/task.json`；三者通过任务构建性检查与源数据答案 oracle。多步生成曾暴露能力规划额外添加动作，以及结果指标被晋升为语义指标后丢失答案键的问题，现已分别通过限定源数据任务的动作集合和在最终指标定型后再次写入答案键修复。oracle 按实际工具名称核验 DAG 与成功场景的 capture/`$ref`，不再依赖固定工具名。

三个新任务均已用 `gpt-6-luna` 完成实际沙箱构建和独立审查；`output/dataset_trial_travel_tourism_v2/offline_score_summary.json` 显示离线可执行评分均为 10/10。多步沙箱首次实现把源数据的 `Completed` 错写为小写状态，导致工具返回空记录；构建流程按缺陷修复后重新通过完整验收，且新增了读取源数据大小写的回归用例。这次验证仍采用进程内、离线模拟评估器，没有证明真实在线 rollout 或生产认证良品率。
