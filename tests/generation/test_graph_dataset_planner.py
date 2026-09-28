"""The graph may plan only against reviewed relations and unchanged raw bytes."""

import random
from unittest.mock import patch

import pytest

from env_factory.generation.dataset_task_generator import (
    DatasetTaskGenerator, TaskGenerationError, _choose_rows, _description, _project,
)
from env_factory.graph.dataset_planner import (
    GraphDatasetTaskGenerator, reviewed_links, sync_reviewed_links, verify_link_source,
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

    def reconcile_reviewed_links(self, links):
        self.reconciled = links

    def supported_dataset_links(self, platform):
        return tuple((SceneNode(link.scene_name), link) for link in self.links
                     if link.dataset_key.startswith(platform + ":"))


def test_reviewed_relations_are_backed_by_approved_raw_sources():
    links = reviewed_links()
    assert len(links) == 4
    assert {link.dataset_key.split(":", 1)[0] for link in links} == {
        "kaggle", "data_gov_hk",
    }
    for link in links:
        source, _, _, _ = verify_link_source(link)
        assert source.is_file()


def test_graph_sync_writes_only_verified_source_relations():
    store = RecordingStore()
    assert sync_reviewed_links(store) == 4
    assert len(store.scenes) == len(store.relations) == 4
    assert {node.key for node in store.datasets} == {
        "kaggle:mohammadtalib786/retail-sales-dataset",
        "data_gov_hk:cc-pricewatch-pricewatch",
    }
    assert len(store.resources) == 4
    assert {field.role for field in store.fields} == {"identifier", "group", "value"}
    assert store.reconciled == reviewed_links()


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
