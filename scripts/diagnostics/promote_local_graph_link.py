#!/usr/bin/env python3
"""Inspect or promote an approved, locally verified Scene relation into SUPPORTED_BY."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory.graph.dataset_planner import sync_reviewed_links, verify_link_source
from env_factory.graph.graph_builder import Neo4jGraphStore, SceneDatasetLink
from env_factory.graph.llm_dataset_linker import append_llm_links


PROJECT = Path(__file__).resolve().parents[2]


def promotion_rows(dataset_key: str, candidates: tuple[dict, ...], scenes: list[str]) -> tuple[dict, ...]:
    """Require explicit scene selection and revalidate source approval before writing."""
    by_scene = {row["scene"]: row for row in candidates}
    if not scenes or len(scenes) != len(set(scenes)) or set(scenes) - set(by_scene):
        raise ValueError("--scene must name distinct existing candidate scenes")
    selected = []
    for scene in scenes:
        row = by_scene[scene]
        link = SceneDatasetLink(scene, dataset_key, row["source_sha256"], row["evidence"],
                                row.get("group_field"), row.get("group_value"),
                                row["business_label"], "llm_source_grounded_v2")
        verify_link_source(link, require_local=True)
        selected.append({"scene": scene, "dataset_key": dataset_key,
                         "source_sha256": link.source_sha256, "evidence": link.evidence,
                         "business_label": link.business_label,
                         **({"group_field": link.group_field, "group_value": link.group_value}
                            if link.group_field else {}),
                         "review_method": "llm_source_grounded_v2",
                         "llm_model": "human_selected_local_candidate"})
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--scene", action="append", help="scene to promote; repeat after review")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env", override=True)
    with Neo4jGraphStore(database=os.getenv("NEO4J_DATABASE", "neo4j")) as store:
        store.verify_connectivity()
        candidates = store.local_link_candidates(args.dataset_key)
        if not args.scene:
            print(json.dumps({"dataset_key": args.dataset_key, "candidates": candidates},
                             ensure_ascii=False, indent=2))
            return
        selected = promotion_rows(args.dataset_key, candidates, args.scene)
        added = append_llm_links(selected)
        synced = sync_reviewed_links(store, require_local=True)
        print(json.dumps({"promoted": added, "reviewed_links_synced": synced},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
