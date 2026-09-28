"""Index both catalogs, then verify and synchronize reviewed graph relations."""

import argparse

from env_factory.graph.dataset_planner import (
    catalog_rows, reviewed_links, sync_catalog_datasets, sync_reviewed_links,
)
from env_factory.graph.graph_builder import Neo4jGraphStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links-only", action="store_true", help="仅更新已审核的原始数据关系")
    parser.add_argument("--suggest-scene", help="显示与该场景同主题的目录候选，供人工审核")
    args = parser.parse_args()
    with Neo4jGraphStore() as store:
        catalog_count = 0 if args.links_only else sync_catalog_datasets(store)
        if not args.links_only:
            counts = store.catalog_dataset_counts()
            expected = {platform: sum(row["platform"] == platform for row in catalog_rows())
                        for platform in ("kaggle", "data_gov_hk")}
            if counts != expected:
                raise RuntimeError(f"Neo4j catalog counts differ from committed indexes: {counts} != {expected}")
        count = sync_reviewed_links(store)
        persisted = {link for platform in ("kaggle", "data_gov_hk")
                     for _, link in store.supported_dataset_links(platform)}
        if not set(reviewed_links()).issubset(persisted):
            raise RuntimeError("Neo4j did not persist every reviewed Scene -> Dataset relation")
        candidates = (store.catalog_candidates_for_scene(args.suggest_scene)
                      if args.suggest_scene else ())
    print(f"Indexed {catalog_count} catalog datasets; verified {count} reviewed Scene -> Dataset links")
    if args.suggest_scene:
        print("Metadata candidates only; raw source approval and scene review are still required:")
    for item in candidates:
        print(f"{item['key']}\t{item['title']}\t{item['topic']}\t"
              f"catalog_approved={item['approved']}\t{item['url']}")
