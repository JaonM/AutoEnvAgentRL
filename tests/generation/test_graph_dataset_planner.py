"""The graph may plan only against reviewed relations and unchanged raw bytes."""

import random
from unittest.mock import patch

import pytest

from env_factory.generation.dataset_task_generator import (
    DatasetTaskGenerator, TaskGenerationError, _choose_rows, _columns,
    _description, _project, _sample_source,
)
from env_factory.graph.dataset_planner import (
    GraphDatasetTaskGenerator, catalog_rows, reviewed_links, sync_catalog_datasets,
    sync_reviewed_links, verify_link_source,
)
from env_factory.graph.graph_builder import SceneDatasetLink
from env_factory.graph.knowledge_graph import SceneNode


class RecordingStore:
    def __init__(self, links=()):
        self.links = links
        self.scenes = []
        self.datasets = []
        self.resources = []
        self.fields = []
        self.relations = []
        self.reconciled = ()
        self.catalog = ()
        self.discovery = {}
        self.scene_edges = []
        self.reconciled_scene_edges = ()

    def verify_connectivity(self):
        pass

    def ensure_schema(self):
        pass

    def upsert_scene(self, node):
        self.scenes.append(node)

    def upsert_dataset(self, node):
        self.datasets.append(node)

    def upsert_resource(self, node):
        self.resources.append(node)

    def upsert_field(self, node):
        self.fields.append(node)

    def link_scene_dataset(self, link):
        self.relations.append(link)

    def link_scene_extension(self, parent, child):
        self.scene_edges.append((parent, child))

    def reconcile_scene_extensions(self, pairs):
        self.reconciled_scene_edges = pairs

    def reconcile_reviewed_links(self, links):
        self.reconciled = links

    def upsert_catalog_datasets(self, rows):
        self.catalog = tuple(rows)
        return len(self.catalog)

    def set_scene_discovery_terms(self, scene, terms):
        self.discovery[scene] = terms

    def supported_dataset_links(self, platform):
        return tuple((SceneNode(link.scene_name), link) for link in self.links
                     if link.dataset_key.startswith(platform + ":"))


def test_reviewed_relations_are_backed_by_approved_raw_sources():
    links = reviewed_links()
    assert len(links) == 54
    assert {link.dataset_key.split(":", 1)[0] for link in links} == {
        "kaggle", "data_gov_hk",
    }
    for link in links:
        source, _, _, _ = verify_link_source(link)
        assert source.is_file()


def test_complete_catalogs_are_metadata_only_until_source_review():
    rows = catalog_rows()
    assert len(rows) == 13_822
    assert sum(row["platform"] == "kaggle" for row in rows) == 10_000
    assert sum(row["platform"] == "data_gov_hk" for row in rows) == 3_822
    assert sum(row["approved"] for row in rows) == 9
    assert len({row["key"] for row in rows}) == len(rows)
    store = RecordingStore()
    assert sync_catalog_datasets(store, rows) == len(rows)
    assert not store.relations


def test_graph_sync_writes_only_verified_source_relations():
    store = RecordingStore()
    assert sync_reviewed_links(store) == 54
    assert len(store.relations) == 54
    assert len(store.scene_edges) == 43
    assert store.reconciled_scene_edges == tuple(store.scene_edges)
    assert {node.key for node in store.datasets} == {
        "kaggle:mohammadtalib786/retail-sales-dataset",
        "kaggle:ankitbansal06/retail-orders",
        "kaggle:sophietwohey/synthetic-hotel-dataset",
        "kaggle:prince7489/online-retail-transactions-dataset",
        "kaggle:mahmoudmansour22/retail-store-sales-transactions-20222024",
        "kaggle:alexhuitron/supermarket-sales",
        "kaggle:mehmettahiraslan/customer-shopping-dataset",
        "kaggle:arunkumaroraon/indian-sales-transactions-dataset-2025",
        "data_gov_hk:cc-pricewatch-pricewatch",
    }
    assert len(store.resources) == 54
    assert {field.role for field in store.fields} == {"identifier", "group", "value"}
    assert store.reconciled == reviewed_links()
    assert store.discovery["商品价格核对"] == ("价格", "格价", "物价")


def test_hotel_relation_uses_pinned_reservations_table():
    link = next(link for link in reviewed_links()
                if link.dataset_key == "kaggle:sophietwohey/synthetic-hotel-dataset")
    source, *_ = verify_link_source(link)
    assert source.name == "reservations.csv"


def test_stale_source_hash_rejects_batch_before_graph_mutation():
    good = reviewed_links()[0]
    stale = SceneDatasetLink(good.scene_name, good.dataset_key, "0" * 64,
                             good.evidence, good.group_field, good.group_value,
                             good.business_label)
    store = RecordingStore()
    with pytest.raises(TaskGenerationError, match="not approved"):
        sync_reviewed_links(store, (good, stale))
    assert not store.relations and not store.scenes


def test_all_reviewed_relations_have_usable_task_rows():
    """A reviewed relation must support every training route it advertises."""
    for link in reviewed_links():
        source, *_ = verify_link_source(link)
        headers, rows = _sample_source(source)
        key, group, numeric = _columns(headers, rows)
        if link.group_field:
            rows = [row for row in rows if row[group] == link.group_value]
        unique_rows = list({row[key]: row for row in reversed(rows)}.values())
        unique_rows.reverse()
        routes = [("direct_response", "minimum")]
        if not link.group_field:
            routes += [("simple_agentic", "minimum")]
            routes += [("multi_step_agentic", operation)
                       for operation in ("minimum", "maximum", "average", "count")]
        for category, operation in routes:
            selected = _choose_rows(unique_rows, key, group, numeric, category,
                                    random.Random(17), extreme=operation)
            assert selected, (link.scene_name, link.dataset_key, category, operation)


def test_graph_generator_ignores_unreviewed_relationships():
    link = reviewed_links()[0]
    forged = SceneDatasetLink(link.scene_name, link.dataset_key, link.source_sha256,
                              "unreviewed evidence")
    generator = GraphDatasetTaskGenerator(RecordingStore((forged,)), object(),
                                          max_source_bytes=100_000_000)
    with pytest.raises(TaskGenerationError, match="no reviewed"):
        generator.generate(dataset_platform="kaggle")


def test_graph_generator_pins_dataset_and_scene():
    link = reviewed_links()[0]
    generator = GraphDatasetTaskGenerator(RecordingStore((link,)), object(),
                                          max_source_bytes=100_000_000)
    with patch.object(DatasetTaskGenerator, "generate", return_value="task") as run:
        assert generator.generate(dataset_platform="kaggle", training_category="direct_response", seed=7) == "task"
    assert run.call_args.kwargs["graph_link"] == link
    assert run.call_args.kwargs["dataset_platform"] == "kaggle"


def test_agentic_routes_skip_links_that_reveal_hidden_group():
    narrow = reviewed_links()[0]
    broad = reviewed_links()[1]
    generator = GraphDatasetTaskGenerator(RecordingStore((narrow, broad)), object(),
                                          max_source_bytes=100_000_000)
    with patch.object(DatasetTaskGenerator, "generate", return_value="task") as run:
        generator.generate(dataset_platform="kaggle", training_category="multi_step_agentic", seed=7)
    assert run.call_args.kwargs["graph_link"] == broad


def test_graph_maximum_plan_has_verified_extreme():
    rows = [{"id": str(index), "category": "Food", "amount": str(index * 10)}
            for index in range(1, 12)]
    chosen = _choose_rows(rows, "id", "category", "amount", "multi_step_agentic",
                          random.Random(2), extreme="maximum")
    projected = _project(chosen, "id", "category", "amount")
    description, facts = _description("multi_step_agentic", projected,
                                      "id", "category", "amount", "Food",
                                      extreme="maximum")
    assert facts["highest_value"] == max(row["amount"] for row in projected)
    assert facts["starting_record_id"] != facts["highest_record_id"]
    assert "最高" in description["task"]


def test_graph_direct_comparison_can_ask_for_higher_value():
    rows = [{"id": "A", "category": "Food", "amount": 15.0},
            {"id": "B", "category": "Food", "amount": 21.0}]
    description, facts = _description("direct_response", rows, "id", "category",
                                      "amount", "Food", comparison="higher")
    assert facts == {"higher_label": "B", "difference": 6.0}
    assert "较高" in description["task"]


@pytest.mark.parametrize("operation, fact_key", [
    ("average", "average_value"), ("count", "group_count"),
])
def test_graph_group_aggregates_use_selected_source_rows(operation, fact_key):
    rows = [{"id": str(index), "category": "Food", "amount": str(index * 10)}
            for index in range(1, 12)]
    chosen = _choose_rows(rows, "id", "category", "amount", "multi_step_agentic",
                          random.Random(2), extreme=operation)
    projected = _project(chosen, "id", "category", "amount")
    description, facts = _description("multi_step_agentic", projected,
                                      "id", "category", "amount", "Food",
                                      extreme=operation)
    expected = (sum(row["amount"] for row in projected) / len(projected)
                if operation == "average" else len(projected))
    assert facts[fact_key] == expected
    assert description["task_intent"] == "calculate"
    assert len(description["route_plan"]["environment_operations"]) == 2
