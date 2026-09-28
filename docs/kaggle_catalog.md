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

批量下载两个目录索引中的全部候选：

```bash
python3 scripts/diagnostics/download_indexed_datasets.py --scope all \
  --max-dataset-gb 100 --max-total-gb 1400 --reserve-gb 100
python3 scripts/diagnostics/download_indexed_datasets.py --status
```

只处理指定数据集时使用重复的 `--dataset` 参数，先用 `--dry-run` 核对：

```bash
python3 scripts/diagnostics/download_indexed_datasets.py \
  --dataset kaggle:lalit7881/warehouse-and-retail-sales \
  --dataset data_gov_hk:cc-pricewatch-pricewatch --dry-run
python3 scripts/diagnostics/download_indexed_datasets.py \
  --dataset kaggle:lalit7881/warehouse-and-retail-sales \
  --dataset data_gov_hk:cc-pricewatch-pricewatch
```

格式必须是 `kaggle:owner/slug` 或 `data_gov_hk:id`，且键必须存在于对应索引。显式指定时可选择未准入的目录候选，不受默认 `--scope approved` 限制；传入顺序即下载顺序。未知键会报错，不会触发下载；`--dataset` 不能与 `--offset`、`--limit` 同用。

默认范围为已准入来源；`--scope all` 才覆盖索引全部候选。可用 `--platform kaggle`、`--offset`、`--limit` 划分批次。脚本校验 SHA-256、路径和 Kaggle 索引许可，记录每个来源的最近状态；重新执行会核验并跳过已完成数据。下载超限、失效链接或许可变化会记入状态文件，需要逐项处理。DATA.GOV.HK 会按官方实时元数据下载每个资源，保存于 `data/sources/data_gov_hk_bulk/`。远端 API、实时馈送等可能无法作为完整静态文件下载，失败会留痕；目录候选不会因下载而自动获准用于任务或训练。
