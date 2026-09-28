"""Build the knowledge graph from seed words and persist it to Neo4j."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory import (
    GraphExpansionConfig,
    LLMClient,
    Neo4jGraphStore,
    WikipediaClient,
    LocalWikipediaClient,
    SeedGraphExpander,
)
from env_factory.graph.dataset_planner import (
    catalog_rows, reviewed_links, sync_catalog_datasets, sync_reviewed_links,
)
from env_factory.graph.llm_dataset_linker import append_llm_links, propose_llm_links
from env_factory.graph.local_dataset_linker import build_local_candidate_links

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建并持久化任务知识图谱")
    parser.add_argument("--datasets-only", action="store_true",
                        help="跳过 Wikipedia/LLM Scene 扩展，构建目录及 Scene→Dataset 关系")
    parser.add_argument("--links-only", action="store_true",
                        help="跳过 13,822 条数据集目录同步，仅构建场景与数据集关系")
    parser.add_argument("--incremental", action="store_true",
                        help="目录指纹未变时跳过同步；Scene 扩展仍按 Neo4j 检查点续跑")
    parser.add_argument("--offline", action="store_true",
                        help="要求使用本地 Wikipedia 索引；数据集只读本地文件，LLM 仍使用配置端点")
    parser.add_argument("--online-wikipedia", action="store_true",
                        help="即使已配置本地索引，也使用在线 Wikipedia 搜索")
    parser.add_argument("--local-dataset-links", action="store_true",
                        help="为已下载但未准入的候选数据集增量构建非训练用途的 Scene 关系")
    parser.add_argument("--max-local-datasets", type=int, default=20,
                        help="本轮最多新审核的本地数据集数，默认 20")
    parser.add_argument("--local-dataset-key", action="append",
                        help="仅审核指定的已下载数据集键，可重复传入")
    parser.add_argument("--dataset-key", action="append",
                        help="指定索引中的数据集键；自动按已准入或候选关系处理，可重复传入")
    parser.add_argument("--skip-llm-dataset-links", action="store_true",
                        help="跳过外部 LLM 的 Scene→Dataset 匹配，只同步关系注册表")
    parser.add_argument("--llm-links-dry-run", action="store_true",
                        help="只读试运行 LLM 新关系；需与 --datasets-only 一起使用")
    parser.add_argument("--llm-max-links-per-dataset", type=int, default=4,
                        help="每个已核验数据源最多保留的 LLM 新关系数，默认 4")
    parser.add_argument("--llm-max-datasets", type=int, default=None,
                        help="本轮最多分析的数据源数；默认分析全部已核验来源")
    parser.add_argument("--llm-dataset-key", action="append", default=None,
                        help="只用 LLM 匹配指定的已准入数据集键，可重复传入")
    parser.add_argument("--rounds", type=int, default=None, help="最大扩展轮次，默认使用配置值 3")
    parser.add_argument(
        "--max-scene-nodes",
        type=int,
        default=1000,
        help="最多保留的 Scene 节点数量，默认 1000",
    )
    parser.add_argument(
        "--max-search-requests",
        type=int,
        default=500,
        help="本轮最多发起的 Wikipedia 搜索请求数，默认 500",
    )
    parser.add_argument(
        "--term-batch-size",
        type=int,
        default=16,
        help="每次交给 LLM 抽取术语的种子词数量，默认 16",
    )
    parser.add_argument(
        "--max-terms-per-seed",
        type=int,
        default=30,
        help="每个种子词最多抽取的术语数量，默认 30",
    )
    parser.add_argument(
        "--merge-batch-size",
        type=int,
        default=50,
        help="每批参与语义合并的新术语数量，默认 50",
    )
    parser.add_argument(
        "--relation-batch-size",
        type=int,
        default=40,
        help="每批进行关系抽取的新 Scene 节点数量，默认 40",
    )
    parser.add_argument(
        "--relation-candidate-limit",
        type=int,
        default=20,
        help="关系抽取时为每批新增节点保留的候选上下文节点数，默认 20",
    )
    parser.add_argument("--max-workers", type=int, default=2, help="搜索 API 并发数，默认 2")
    return parser.parse_args()


def load_seed_words() -> tuple[str, ...]:
    """Read one seed word per line from GRAPH_SEEDS_FILE."""

    path = seed_file_path()
    seeds = tuple(
        dict.fromkeys(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )
    if not seeds:
        raise ValueError(f"seed file is empty: {path}")
    return seeds


def seed_file_path() -> Path:
    seed_file = os.getenv("GRAPH_SEEDS_FILE")
    if not seed_file:
        raise ValueError(
            "未找到 GRAPH_SEEDS_FILE 配置。原因：.env 中没有配置种子文件路径；"
            "请新增 GRAPH_SEEDS_FILE=data/scene_seeds.txt"
        )
    configured_path = Path(seed_file).expanduser()
    path = configured_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        reason = "路径不存在"
        if path.exists():
            reason = "路径存在但不是普通文件"
        raise FileNotFoundError(
            f"种子文件读取失败：{path}\n"
            f"原因：{reason}；GRAPH_SEEDS_FILE 当前配置为 {seed_file!r}，"
            f"相对路径会基于项目根目录 {PROJECT_ROOT} 解析。\n"
            "修复：确认文件存在，或在 .env 中设置 GRAPH_SEEDS_FILE=data/scene_seeds.txt。"
        )
    logging.getLogger(__name__).info("读取种子文件：%s", path)
    return path


def append_seed_words(words: tuple[str, ...]) -> int:
    path = seed_file_path()
    existing = {line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines()}
    new_words = tuple(dict.fromkeys(word.strip() for word in words if word.strip()))
    pending = tuple(word for word in new_words if word.casefold() not in existing)
    if pending:
        with path.open("a", encoding="utf-8") as file:
            file.write("\n" + "\n".join(pending) + "\n")
    return len(pending)


def main() -> None:
    # The project .env is the source of truth for graph construction.  In
    # particular, this prevents an old exported GRAPH_SEEDS_FILE from
    # silently overriding the repository configuration.
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    if args.skip_llm_dataset_links and args.llm_links_dry_run:
        raise ValueError("--skip-llm-dataset-links and --llm-links-dry-run conflict")
    if args.offline and args.online_wikipedia:
        raise ValueError("--offline and --online-wikipedia conflict")
    if args.llm_links_dry_run and not args.datasets_only:
        raise ValueError("--llm-links-dry-run requires --datasets-only to avoid graph writes")
    if args.max_local_datasets <= 0:
        raise ValueError("--max-local-datasets must be positive")
    if args.local_dataset_key and not args.local_dataset_links:
        raise ValueError("--local-dataset-key requires --local-dataset-links")
    if args.dataset_key and (args.local_dataset_key or args.llm_dataset_key):
        raise ValueError("--dataset-key cannot be combined with source-specific dataset keys")
    targeted_approved: list[str] | None = None
    targeted_local: list[str] | None = None
    if args.dataset_key:
        catalog = {row["key"]: row for row in catalog_rows()}
        missing = set(args.dataset_key) - set(catalog)
        if missing:
            raise ValueError(f"dataset keys are absent from the indexes: {sorted(missing)}")
        targeted_approved = [key for key in dict.fromkeys(args.dataset_key) if catalog[key]["approved"]]
        targeted_local = [key for key in dict.fromkeys(args.dataset_key) if not catalog[key]["approved"]]
    groups = ()
    edges = 0
    added_seeds = 0
    if not args.datasets_only:
        seeds = load_seed_words()
        config = GraphExpansionConfig(
            max_scene_nodes=args.max_scene_nodes,
            max_search_requests=args.max_search_requests,
            max_rounds=args.rounds or 3,
            term_batch_size=args.term_batch_size,
            max_terms_per_seed=args.max_terms_per_seed,
            merge_batch_size=args.merge_batch_size,
            relation_batch_size=args.relation_batch_size,
            relation_candidate_limit=args.relation_candidate_limit,
        )
        llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
        dump_db = None if args.online_wikipedia else os.getenv("WIKIPEDIA_DUMP_DB")
        if args.offline and not dump_db:
            raise ValueError("--offline requires WIKIPEDIA_DUMP_DB")
        search = (
            LocalWikipediaClient(dump_db)
            if dump_db
            else WikipediaClient(timeout=float(os.getenv("WIKIPEDIA_TIMEOUT", "10")))
        )
        expander = SeedGraphExpander(search, llm, max_workers=args.max_workers, config=config)

    with Neo4jGraphStore(database=os.getenv("NEO4J_DATABASE", "neo4j")) as store:
        store.verify_connectivity()
        if args.llm_links_dry_run:
            linker_llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
            proposals = propose_llm_links(
                linker_llm, store.get_scene_nodes(),
                max_new_per_dataset=args.llm_max_links_per_dataset,
                max_datasets=args.llm_max_datasets,
                only_dataset_keys=targeted_approved if targeted_approved is not None else args.llm_dataset_key,
                require_local_sources=True,
            )
            for item in proposals:
                print(f"LLM 候选：{item['scene']} → {item['dataset_key']} "
                      f"({item.get('group_value') or '全部分组'})，"
                      f"业务称呼：{item['business_label']}，依据：{item['evidence']}")
            print(f"只读试运行完成：{len(proposals)} 条通过校验的新候选")
            return
        if not args.datasets_only:
            logging.getLogger(__name__).info("开始扩展 Scene 图谱")
            builder, groups = expander.expand_and_build(store, seeds, rounds=args.rounds)
            edges = len(builder.edges())
            added_seeds = append_seed_words(
                tuple(word for group in groups for word in (group.name, *group.words))
            )
        rows = () if args.links_only else catalog_rows()
        catalog_hash = (hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()
                        if rows else None)
        catalog_count = 0
        if rows and (not args.incremental or store.catalog_fingerprint() != catalog_hash):
            catalog_count = sync_catalog_datasets(store, rows)
            store.set_catalog_fingerprint(catalog_hash)
        if not args.links_only:
            expected = {platform: sum(row["platform"] == platform for row in rows)
                        for platform in ("kaggle", "data_gov_hk")}
            if store.catalog_dataset_counts() != expected:
                catalog_count = sync_catalog_datasets(store, rows)
                store.set_catalog_fingerprint(catalog_hash)
                if store.catalog_dataset_counts() != expected:
                    raise RuntimeError("Neo4j dataset catalog does not match committed indexes")
        new_link_count = 0
        if not args.skip_llm_dataset_links and (targeted_approved is None or targeted_approved):
            linker_llm = (llm if not args.datasets_only else
                          LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60"))))
            proposals = propose_llm_links(
                linker_llm, store.get_scene_nodes(),
                max_new_per_dataset=args.llm_max_links_per_dataset,
                max_datasets=args.llm_max_datasets,
                only_dataset_keys=targeted_approved if targeted_approved is not None else args.llm_dataset_key,
                require_local_sources=args.offline,
            )
            new_link_count = append_llm_links(proposals)
        link_count = sync_reviewed_links(store, require_local=args.offline)
        local_stats = None
        if args.local_dataset_links and (targeted_local is None or targeted_local):
            local_llm = (llm if not args.datasets_only else
                         LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60"))))
            local_stats = build_local_candidate_links(
                store, local_llm, store.get_scene_nodes(), rows or catalog_rows(),
                max_datasets=args.max_local_datasets,
                only_keys=targeted_local if targeted_local is not None else args.local_dataset_key,
            )
        persisted = {link for platform in ("kaggle", "data_gov_hk")
                     for _, link in store.supported_dataset_links(platform)}
        if not set(reviewed_links()).issubset(persisted):
            raise RuntimeError("Neo4j did not persist every reviewed Scene -> Dataset relation")
        scene_count = len(store.get_scene_nodes())
        supported_scenes = len({link.scene_name for link in persisted})

    print(
        f"图谱构建完成：扩展 {len(groups)} 个 Scene、{edges} 条 Scene 关系，"
        f"新增种子词 {added_seeds} 个；同步目录 {catalog_count} 条、"
        f"本轮 LLM 新关系 {new_link_count} 条、"
        f"已核验 Scene→Dataset 关系 {link_count} 条；"
        f"有数据支持的 Scene {supported_scenes}/{scene_count}"
        + (f"；本地候选审核 {local_stats['audited']} 个、"
           f"候选关系 {local_stats['candidates']} 条、跳过未变 {local_stats['unchanged']} 个、"
           f"结构不适用 {local_stats['unusable']} 个、失败 {local_stats['failed']} 个"
           if local_stats is not None else "")
    )


if __name__ == "__main__":
    main()
