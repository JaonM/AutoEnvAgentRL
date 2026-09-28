# Kaggle 任务数据目录

主目录索引保存于 `data/sources/kaggle/task_dataset_index.json`，SQLite 检查点保存在同目录的 `task_dataset_index.sqlite3`。当前 JSON 含 10,000 个去重的公开数据集候选，来自 1,269 个已采集列表页；只保存目录元数据，不下载业务数据文件。筛选条件为 CSV、列表标称大小不超过 100 MB、许可标识为 CC0、CC BY 4.0、Apache 2.0 或 MIT。这是**任务生成的数据集候选索引**，不是已生成的 10,000 个任务，也不代表数据质量、真实授权状态或三类任务适配性已经验证。

更新目录（默认不下载）：

```bash
python3 scripts/diagnostics/index_kaggle_tasks.py --target 10000
```

脚本使用 SQLite 逐页记录进度，重复执行会从检查点续跑。每个候选记录 `ref`、标题、标称大小、许可和匹配关键词；生成任务时再用 `ref` 按需下载：

```bash
python3 scripts/diagnostics/download_kaggle_dataset.py owner/slug
```

文件存入 `data/sources/kaggle/owner/slug/vN/raw/`，`source_manifest.json` 记录版本、来源、许可与 SHA-256。正式任务入口 `./scripts/generate_task.sh --dataset-ref owner/slug` 会按需下载并从支持的表格文件构造可核算任务；缺少唯一 ID、可重复分组或数值业务字段的候选会被拒绝。当前已入库的 10,000 条目录仍是 CSV 筛选结果，非 CSV 数据集可通过明确指定 `--dataset-ref` 使用。自动字段检查不能代替许可、隐私和任务适配审查。JSON 目录索引纳入 Git；SQLite 抓取断点和下载的数据仍由 `.gitignore` 忽略。
