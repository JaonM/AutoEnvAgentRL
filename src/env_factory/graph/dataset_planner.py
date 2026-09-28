"""Reviewed graph relations that bind business scenes to eligible raw datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from env_factory.generation.dataset_task_generator import (
    DEFAULT_ALLOWLIST, PROJECT, DatasetTaskGenerator, TaskGenerationError,
    _columns, _sample_source, _sha256,
)
from env_factory.generation.dataset_formats import source_extension
from env_factory.generation.dataset_source_registry import eligible_hk_ids
from env_factory.graph.graph_builder import Neo4jGraphStore, SceneDatasetLink
from env_factory.graph.knowledge_graph import DatasetNode, FieldNode, ResourceNode, SceneNode


DEFAULT_LINKS = PROJECT / "config" / "graph_dataset_links.json"


def reviewed_links(path: Path = DEFAULT_LINKS) -> tuple[SceneDatasetLink, ...]:
    """Load the explicitly reviewed mappings; metadata catalog rows do not qualify."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1 or not isinstance(document.get("links"), list):
        raise TaskGenerationError("graph dataset link registry is invalid")
    result = []
    for row in document["links"]:
        link = SceneDatasetLink(
            row["scene"], row["dataset_key"], row["source_sha256"], row["evidence"],
            row.get("group_field"), row.get("group_value"),
        )
        if (not link.scene_name.strip() or not link.evidence.strip()
                or (link.group_field is None) != (link.group_value is None)
                or len(link.source_sha256) != 64):
            raise TaskGenerationError("graph dataset link has invalid evidence or field constraint")
        result.append(link)
    return tuple(result)


def verify_link_source(link: SceneDatasetLink, *, llm: Any = None) -> tuple[Path, str, str, str | None]:
    """Recheck approval, raw hash and observed field values before graph use."""
    platform, separator, source_id = link.dataset_key.partition(":")
    if not separator or not source_id:
        raise TaskGenerationError("graph dataset key is invalid")
    if platform == "kaggle":
        approved = json.loads(DEFAULT_ALLOWLIST.read_text(encoding="utf-8"))["datasets"]
        approval = next((row for row in approved if row.get("ref") == source_id and
                         (row.get("source_sha256") or row.get("csv_sha256")) == link.source_sha256), None)
        if approval is None:
            raise TaskGenerationError("graph Kaggle source is not approved at this hash")
        generator = DatasetTaskGenerator(llm, dataset_ref=source_id)
    elif platform == "data_gov_hk":
        if source_id not in eligible_hk_ids():
            raise TaskGenerationError("graph DATA.GOV.HK source is not approved")
        generator = DatasetTaskGenerator(llm, dataset_id=source_id)
    else:
        raise TaskGenerationError("graph dataset platform is unsupported")
    source = generator._source(random.Random(0), platform)
    if _sha256(source[0]) != link.source_sha256:
        raise TaskGenerationError("graph dataset source hash changed")
    if platform == "kaggle" and (
        source[0].parent.parent.name != f"v{approval['version']}"
        or source[2] != f"https://www.kaggle.com/datasets/{source_id}"
        or source[3] != approval["license"]
    ):
        raise TaskGenerationError("graph Kaggle approval version, URL or license changed")
    if link.group_field:
        headers, rows = _sample_source(source[0])
        _, group, _ = _columns(headers, rows)
        if group != link.group_field or sum(row[group] == link.group_value for row in rows) < 3:
            raise TaskGenerationError("graph relation has no support in raw source fields")
    return source


def sync_reviewed_links(store: Neo4jGraphStore, links: tuple[SceneDatasetLink, ...] | None = None) -> int:
    """Install only relations that still match source bytes and source approval."""
    selected = links if links is not None else reviewed_links()
    store.verify_connectivity()
    store.ensure_schema()
    for link in selected:
        source, title, url, _ = verify_link_source(link)
        store.upsert_scene(SceneNode(link.scene_name))
        store.upsert_dataset(DatasetNode(link.dataset_key, title, url, link.source_sha256))
        resource_key = f"{link.dataset_key}:{link.source_sha256}"
        store.upsert_resource(ResourceNode(
            resource_key, link.dataset_key, link.source_sha256,
            source_extension(source.name),
        ))
        headers, rows = _sample_source(source)
        key, group, numeric = _columns(headers, rows)
        for role, field in (("identifier", key), ("group", group), ("value", numeric)):
            store.upsert_field(FieldNode(f"{resource_key}:{field}", resource_key, field, role))
        store.link_scene_dataset(link)
    store.reconcile_reviewed_links(selected)
    return len(selected)


class GraphDatasetTaskGenerator:
    """Plan from supported graph scenes, then use verified rows as task truth."""

    def __init__(self, store: Neo4jGraphStore, llm: Any, *, max_source_bytes: int,
                 user_script_count: int = 3, noise_tool_max: int = 3) -> None:
        self.store = store
        self.llm = llm
        self.max_source_bytes = max_source_bytes
        self.user_script_count = user_script_count
        self.noise_tool_max = noise_tool_max

    def generate(self, hops: int = 0, task_type: str | None = None,
                 task_style: str | None = None, artifact_dir: str | Path | None = None,
                 task_intent: str | None = None, training_category: str = "multi_step_agentic",
                 seed: int | None = None, dataset_platform: str = "kaggle"):
        del hops  # Scene selection is constrained by reviewed data links.
        candidates = self.store.supported_dataset_links(dataset_platform)
        reviewed = set(reviewed_links())
        candidates = [(scene, link) for scene, link in candidates
                      if link in reviewed and
                      (training_category == "direct_response" or link.group_field is None)]
        if not candidates:
            raise TaskGenerationError(
                f"no reviewed, source-backed Scene relation for {dataset_platform}; "
                "run scripts/diagnostics/sync_graph_dataset_links.py"
            )
        scene, link = random.Random(seed).choice(candidates)
        verify_link_source(link, llm=self.llm)
        platform, source_id = link.dataset_key.split(":", 1)
        generator = DatasetTaskGenerator(
            self.llm, dataset_ref=source_id if platform == "kaggle" else None,
            dataset_id=source_id if platform == "data_gov_hk" else None,
            max_source_bytes=self.max_source_bytes,
            user_script_count=self.user_script_count, noise_tool_max=self.noise_tool_max,
        )
        return generator.generate(
            task_type=task_type, task_style=task_style, artifact_dir=artifact_dir,
            task_intent=task_intent, training_category=training_category, seed=seed,
            dataset_platform=platform, graph_link=link,
        )
