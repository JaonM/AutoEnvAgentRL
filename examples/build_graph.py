"""Build the knowledge graph from seed words and persist it to Neo4j."""

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建并持久化任务知识图谱")
    parser.add_argument("--datasets-only", action="store_true",
                        help="跳过 Wikipedia/LLM Scene 扩展，仅构建数据集目录与已审核场景关系")
    parser.add_argument("--links-only", action="store_true",
                        help="跳过 13,822 条数据集目录同步，仅核验并同步场景关系")
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
        dump_db = os.getenv("WIKIPEDIA_DUMP_DB")
        search = (
            LocalWikipediaClient(dump_db)
            if dump_db
            else WikipediaClient(timeout=float(os.getenv("WIKIPEDIA_TIMEOUT", "10")))
        )
        expander = SeedGraphExpander(search, llm, max_workers=args.max_workers, config=config)

    with Neo4jGraphStore(database=os.getenv("NEO4J_DATABASE", "neo4j")) as store:
        store.verify_connectivity()
        if not args.datasets_only:
            logging.getLogger(__name__).info("开始扩展 Scene 图谱")
            builder, groups = expander.expand_and_build(store, seeds, rounds=args.rounds)
            edges = len(builder.edges())
            added_seeds = append_seed_words(
                tuple(word for group in groups for word in (group.name, *group.words))
            )
        catalog_count = 0 if args.links_only else sync_catalog_datasets(store)
        if not args.links_only:
            expected = {platform: sum(row["platform"] == platform for row in catalog_rows())
                        for platform in ("kaggle", "data_gov_hk")}
            if store.catalog_dataset_counts() != expected:
                raise RuntimeError("Neo4j dataset catalog does not match committed indexes")
        link_count = sync_reviewed_links(store)
        persisted = {link for platform in ("kaggle", "data_gov_hk")
                     for _, link in store.supported_dataset_links(platform)}
        if not set(reviewed_links()).issubset(persisted):
            raise RuntimeError("Neo4j did not persist every reviewed Scene -> Dataset relation")
        scene_count = len(store.get_scene_nodes())
        supported_scenes = len({link.scene_name for link in persisted})

    print(
        f"图谱构建完成：扩展 {len(groups)} 个 Scene、{edges} 条 Scene 关系，"
        f"新增种子词 {added_seeds} 个；同步目录 {catalog_count} 条、"
        f"已核验 Scene→Dataset 关系 {link_count} 条；"
        f"有数据支持的 Scene {supported_scenes}/{scene_count}"
    )


if __name__ == "__main__":
    main()
