# 代码结构

项目按运行边界组织，避免同一契约在生成器、构建器和认证器中各自实现一份。

| 路径 | 职责 |
| --- | --- |
| `src/env_factory/graph/` | 图谱模式、构建、扩展、场景关系与维基百科数据源 |
| `src/env_factory/tasks/` | 任务规范、质量评分、路由、相似性与可迁移性 |
| `src/env_factory/task_pipeline.py` | 从图谱路径到任务、数据、工具、奖励与验收产物的生成编排 |
| `src/env_factory/generation/` | 任务生成器、阶段提示、响应解析与重试、缓存和用户模拟契约 |
| `src/env_factory/contracts/` | 生成、构建预检和认证共用的运行接口、工具链与奖励契约检查 |
| `src/env_factory/sandbox_runtime.py` | 沙箱进程内共享的工具、数据、奖励与 HTTP 运行时 |
| `src/env_factory/evidence/` | 训练素材、认证策略、数据治理、来源与可迁移性校验 |
| `scripts/` | 图谱与任务入口、沙箱开发、实验调度和素材认证入口 |
| `scripts/sandbox/` | 构建预检、脚手架、可执行校验、评分与缺陷修复辅助工具 |
| `scripts/rollout/` | live rollout 与数据治理、轨迹隐私审计 |
| `scripts/diagnostics/` | 只读开发批次统计与同 seed 配对比较 |
| `tests/generation/` | 图谱、任务生成、阶段响应与流水线回归 |
| `tests/tasks/`、`tests/runtime/` | 任务规则、沙箱与运行时回归 |
| `tests/evidence/` | 素材、证据与生产认证回归 |
| `tests/construction/`、`tests/integration/` | 构建与跨阶段流程回归 |
| `tests/diagnostics/` | 开发报告与配对比较回归 |

旧的 `env_factory.<模块名>` 导入路径由包入口映射到新位置，已有集成可继续使用；
新代码直接从对应子包导入。

主要调用链：`examples/generate_task.py` → `TaskGenerationPipeline` →
`scripts/sandbox/assess_task_buildability.py` → `scripts/develop_sandbox_with_agent.sh` →
`scripts/loop_experiment.py` 的 rollout/留出集 →
`scripts/certify_training_materials.py` → `scripts/export_training_materials.py`。

共享契约放在 `contracts/`，由生成末端和构建前预检同时使用；生产认证继续独立复核
冻结产物和运行证据。开发报告只比较实验，不参与认证判定。
