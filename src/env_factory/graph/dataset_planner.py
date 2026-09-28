"""Reviewed graph relations that bind business scenes to eligible raw datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from env_factory.generation.dataset_task_generator import (
    DEFAULT_ALLOWLIST, PROJECT, DatasetTaskGenerator, TaskGenerationError,
    _columns, _number, _sample_source, _sha256,
)
from env_factory.generation.dataset_formats import source_extension
from env_factory.generation.dataset_source_registry import eligible_hk_ids
from env_factory.graph.graph_builder import Neo4jGraphStore, SceneDatasetLink
from env_factory.graph.knowledge_graph import DatasetNode, FieldNode, ResourceNode, SceneNode


DEFAULT_LINKS = PROJECT / "config" / "graph_dataset_links.json"
KAGGLE_INDEX = PROJECT / "data" / "sources" / "kaggle" / "task_dataset_index.json"
HK_INDEX = PROJECT / "data" / "sources" / "data_gov_hk" / "dataset_index.json"


def catalog_rows(
    kaggle_index: Path = KAGGLE_INDEX, hk_index: Path = HK_INDEX,
    allowlist: Path = DEFAULT_ALLOWLIST,
) -> tuple[dict[str, Any], ...]:
    """Read both checked-in catalogs as metadata, preserving separate approval."""
    kaggle = json.loads(kaggle_index.read_text(encoding="utf-8"))
    hk = json.loads(hk_index.read_text(encoding="utf-8"))
    approved_kaggle = {
        row["ref"]: row["license"]
        for row in json.loads(allowlist.read_text(encoding="utf-8"))["datasets"]
    }
    for name, document in (("Kaggle", kaggle), ("DATA.GOV.HK", hk)):
        if (not isinstance(document.get("datasets"), list)
                or document.get("count") != len(document["datasets"])):
            raise TaskGenerationError(f"{name} catalog count does not match its dataset entries")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in kaggle["datasets"]:
        ref = item.get("ref")
        if not isinstance(ref, str) or not ref or "/" not in ref:
            raise TaskGenerationError("Kaggle catalog contains an invalid ref")
        topic = str(item.get("matched_query") or "").strip().casefold()
        row = {"key": f"kaggle:{ref}", "platform": "kaggle",
               "title": str(item.get("title") or ref),
               "url": f"https://www.kaggle.com/datasets/{ref}",
               "provider": ref.split("/", 1)[0], "category": topic,
               "approved": approved_kaggle.get(ref) == item.get("license")
               and ref in approved_kaggle,
               "license": str(item.get("license") or ""),
               "resource_count": 0, "resource_formats": [],
               "topic_key": f"kaggle:{topic}" if topic else "", "topic_name": topic}
        rows.append(row)
    for item in hk["datasets"]:
        dataset_id = item.get("id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise TaskGenerationError("DATA.GOV.HK catalog contains an invalid id")
        topic = str(item.get("category") or "").strip()
        row = {"key": f"data_gov_hk:{dataset_id}", "platform": "data_gov_hk",
               "title": str(item.get("title") or dataset_id),
               "url": str(item.get("url") or ""),
               "provider": str(item.get("provider") or ""), "category": topic,
               "approved": item.get("approved") is True,
               "license": str(item.get("license") or ""),
               "resource_count": int(item.get("resource_count") or 0),
               "resource_formats": list(item.get("resource_formats") or []),
               "topic_key": f"data_gov_hk:{topic}" if topic else "", "topic_name": topic}
        rows.append(row)
    for row in rows:
        if row["key"] in seen:
            raise TaskGenerationError(f"duplicate catalog dataset key: {row['key']}")
        seen.add(row["key"])
    return tuple(rows)


def sync_catalog_datasets(store: Neo4jGraphStore, rows: tuple[dict[str, Any], ...] | None = None) -> int:
    """Load the complete local catalogs into Neo4j without creating reviewed links."""
    selected = rows if rows is not None else catalog_rows()
    store.verify_connectivity()
    store.ensure_schema()
    return store.upsert_catalog_datasets(selected)


def reviewed_links(path: Path = DEFAULT_LINKS) -> tuple[SceneDatasetLink, ...]:
    """Load the explicitly reviewed mappings; metadata catalog rows do not qualify."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1 or not isinstance(document.get("links"), list):
        raise TaskGenerationError("graph dataset link registry is invalid")
    result = []
    pairs = set()
    extensions = []
    for row in document["links"]:
        link = SceneDatasetLink(
            row["scene"], row["dataset_key"], row["source_sha256"], row["evidence"],
            row.get("group_field"), row.get("group_value"), row.get("business_label"),
        )
        if (not link.scene_name.strip() or not link.evidence.strip()
                or (link.group_field is None) != (link.group_value is None)
                or not isinstance(link.business_label, str) or len(link.business_label.strip()) < 2
                or len(link.source_sha256) != 64):
            raise TaskGenerationError("graph dataset link has invalid evidence or field constraint")
        pair = (link.scene_name, link.dataset_key)
        if pair in pairs:
            raise TaskGenerationError(f"duplicate graph Scene -> Dataset relation: {pair}")
        pairs.add(pair)
        parent = row.get("parent_scene")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()
                                   or parent == link.scene_name or not link.group_field):
            raise TaskGenerationError("graph scene extension needs a parent and observed group")
        if parent:
            extensions.append((parent, link.dataset_key))
        result.append(link)
    if any(parent not in pairs for parent in extensions):
        raise TaskGenerationError("graph scene extension parent lacks a source-backed relation")
    return tuple(result)


def scene_extension_parents(path: Path = DEFAULT_LINKS) -> dict[tuple[str, str], str]:
    """Return reviewed parent scenes for source-backed, group-specific extensions."""
    document = json.loads(path.read_text(encoding="utf-8"))
    reviewed_links(path)
    return {(row["scene"], row["dataset_key"]): row["parent_scene"]
            for row in document["links"] if row.get("parent_scene")}


def discovery_terms(path: Path = DEFAULT_LINKS) -> dict[str, tuple[str, ...]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    mapping = document.get("discovery_terms", {})
    if not isinstance(mapping, dict):
        raise TaskGenerationError("graph discovery terms must be a mapping")
    result = {}
    for scene, terms in mapping.items():
        if (not isinstance(scene, str) or not isinstance(terms, list)
                or not terms or any(not isinstance(term, str) or len(term.strip()) < 2
                                 for term in terms)):
            raise TaskGenerationError("graph discovery terms contain an invalid scene or term")
        result[scene] = tuple(dict.fromkeys(term.strip().casefold() for term in terms))
    return result


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
        _, group, numeric = _columns(headers, rows)
        members = [row for row in rows if row[group] == link.group_value]
        if (group != link.group_field or len(members) < 3
                or len({_number(row[numeric]) for row in members}) < 2):
            raise TaskGenerationError("graph relation has no support in raw source fields")
    return source


def sync_reviewed_links(store: Neo4jGraphStore, links: tuple[SceneDatasetLink, ...] | None = None) -> int:
    """Install only relations that still match source bytes and source approval."""
    selected = links if links is not None else reviewed_links()
    # Check every source before mutating the graph, so a stale hash cannot
    # leave a half-updated registry in Neo4j.
    sources = [(link, verify_link_source(link)) for link in selected]
    parents = scene_extension_parents()
    store.verify_connectivity()
    store.ensure_schema()
    for link, (source, title, url, _) in sources:
        store.upsert_scene(SceneNode(link.scene_name))
        parent = parents.get((link.scene_name, link.dataset_key))
        if parent:
            store.upsert_scene(SceneNode(parent))
            store.link_scene_extension(parent, link.scene_name)
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
    for scene, terms in discovery_terms().items():
        store.set_scene_discovery_terms(scene, terms)
    store.reconcile_reviewed_links(selected)
    store.reconcile_scene_extensions(tuple((parent, link.scene_name) for link in selected
                                           if (parent := parents.get((link.scene_name, link.dataset_key)))))
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
