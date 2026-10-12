# 代码结构

项目按运行边界组织，避免同一契约在生成器、构建器和认证器中各自实现一份。

| 路径 | 职责 |
| --- | --- |
| `src/env_factory/graph/` | 图谱模式、构建、扩展、场景关系与维基百科数据源 |
| `src/env_factory/tasks/` | 任务规范、质量评分、路由、相似性与可迁移性 |
| `src/env_factory/task_pipeline.py` | Code Agent 任务生成、沙箱构建、Docker 与 live 验收主入口 |
| `src/env_factory/generation/` | Code Agent 作者、确定性契约编译器、审查与用户模拟契约 |
| `src/env_factory/contracts/` | 生成、构建预检和认证共用的运行接口、工具链与奖励契约检查 |
| `src/env_factory/sandbox_runtime.py` | 沙箱进程内共享的工具、数据、奖励与 HTTP 运行时 |
| `src/env_factory/evidence/` | 训练素材、认证策略、数据治理、来源与可迁移性校验 |
| `scripts/` | 图谱与任务入口、沙箱开发和素材认证入口 |
| `scripts/sandbox/` | 构建预检、脚手架、可执行校验、评分与缺陷修复辅助工具 |
| `scripts/rollout/` | live rollout 与数据治理、轨迹隐私审计 |
| `tests/generation/` | 图谱、Code Agent 任务生成与契约回归 |
| `tests/tasks/`、`tests/runtime/` | 任务规则、沙箱与运行时回归 |
| `tests/evidence/` | 素材、证据与生产认证回归 |
| `tests/construction/`、`tests/integration/` | 构建与跨阶段流程回归 |

旧的 `env_factory.<模块名>` 导入路径由包入口映射到新位置，已有集成可继续使用；
新代码直接从对应子包导入。

主要调用链：`scripts/run_pipeline_legacy.sh` → `src/env_factory/task_pipeline.py` →
`src/env_factory/generate_task.py` / `generation/task_generator.py` / `generation/code_agent.py` →
`scripts/sandbox/assess_task_buildability.py` → `scripts/develop_sandbox_with_agent.sh` →
离线评分、Docker、live rollout 与奖励校准。素材认证和导出仍由各自脚本处理。

`generation/contract_compiler.py` 保存共享的确定性校验和产物编译函数，不调用生成模型。
旧分阶段 LLM 生成器、规格原型生成器、循环实验和跨轮复用代码已删除。

共享契约放在 `contracts/`，由生成末端和构建前预检同时使用；生产认证继续独立复核
冻结产物和运行证据。当前调用的拟真机制统计由主流水线输出到 `realism_summary.json`。固定样本模型对比工具保留在 `scripts/experiments/`，其结果不参与训练认证判定。


## RL 训练代码

`src/rl/` 按训练编排、算法、MLX 后端、采样、数据、环境、检查点、监控和导出组织：

```text
rl/train.py → trainer/runner.py → trainer/loop.py（TrainingSession）
                                    ├── trainer/state.py（TrainingState）
                                    ├── trainer/optimization.py → backends/mlx/actor.py
                                    ├── rollout/manager.py → worker.py → scheduler.py
                                    │                                      └── environments/
                                    ├── trainer/publication.py / persistence.py
                                    └── trainer/periodic.py / finalization.py
```

`config.py` 管参数；`data/` 管契约和队列；`checkpointing/` 管恢复与保留；
`observability/` 管日志看板；`export/` 管 HF/PEFT 产物。
旧 `rl.<模块名>` 文件保留为兼容入口，内部实现通过规范子包相互引用。

模块职责、执行顺序及恢复兼容范围见 [RL 模块组织与训练编排](agent_rl.md#rl-模块组织与训练编排)。
