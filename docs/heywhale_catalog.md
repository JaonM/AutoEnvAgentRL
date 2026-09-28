# 和鲸中文数据集目录

运行 `python3 scripts/diagnostics/index_heywhale_datasets.py` 可从[和鲸社区公开数据集目录](https://www.heywhale.com/home/dataset)采集元数据。脚本逐页写入 SQLite 检查点，可中断续跑；不会下载数据文件。

当前采集完成 104 页，页面报告总数与去重结果均为 10,382 条。其中 8,945 条标题包含中文字符，写入 `data/sources/heywhale/chinese_dataset_index.json`；完整目录写入 `data/sources/heywhale/dataset_index.json`；检查点为 `data/sources/heywhale/dataset_index.sqlite3`。索引记录标题、简述、来源链接、作者、平台主题、文件扩展名与数量、标称体积、许可标识及更新时间。

完整目录及中文子目录的 JSON 索引纳入 Git；SQLite 抓取断点与下载的数据文件留在本地。

文件扩展名统计中，含 CSV 的数据集有 4,919 个，含 ZIP 的有 2,156 个，含 XLSX 的有 1,607 个。平台主题包括科技互联网、商业、气象、地理、医疗健康、人文社科、经济、电商和交通出行等。7,007 个数据集在目录中未标注许可；即使标注了许可，也需要在实际下载和用于训练素材前核对详情页及源文件。索引仅供发现候选，尚未做任务适配、字段、隐私或质量验证。
