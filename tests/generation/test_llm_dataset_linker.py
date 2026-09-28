"""External LLM links must be constrained by approved raw sources and Scenes."""

import json
from types import SimpleNamespace

import pytest

from env_factory.graph import llm_dataset_linker
from env_factory.graph.dataset_planner import reviewed_links
from env_factory.graph.knowledge_graph import SceneNode
from env_factory.graph.llm_dataset_linker import append_llm_links, propose_llm_links
from env_factory.generation.dataset_task_generator import TaskGenerationError


class FakeLLM:
    model = "fake-external-model"

    def complete(self, prompt, **kwargs):
        assert kwargs["response_format"] == "json_object"
        context = json.loads(prompt)
        assert "货品分类2" == context["fields"]["group"]
        if "accepted_ids" in kwargs["system_prompt"]:
            return SimpleNamespace(content='{"accepted_ids":[0]}')
        return SimpleNamespace(content=json.dumps({"links": [
            {"scene": "不存在的场景", "group_value": "面包", "business_label": "面包价格",
             "reason": "原始分组和价格字段支持该场景"},
            {"scene": "食品价格核对", "group_value": "不存在的分组", "business_label": "食品价格",
             "reason": "原始分组和价格字段支持该场景"},
            {"scene": "食品价格核对", "group_value": "面包", "business_label": "面包价格",
             "reason": "货品分类和价格可回答面包价格问题"},
        ]}, ensure_ascii=False))


def test_llm_linker_rejects_unknown_scene_and_unobserved_group():
    base = next(link for link in reviewed_links()
                if link.dataset_key == "data_gov_hk:cc-pricewatch-pricewatch"
                and link.group_field is None)
    proposals = propose_llm_links(
        FakeLLM(), (SceneNode("商品价格核对"), SceneNode("食品价格核对")),
        links=(base,), max_datasets=1,
    )
    assert len(proposals) == 1
    assert proposals[0]["scene"] == "食品价格核对"
    assert proposals[0]["group_value"] == "面包"
    assert proposals[0]["review_method"] == "llm_source_grounded_v2"


def test_independent_llm_review_can_reject_grounded_candidate():
    base = next(link for link in reviewed_links()
                if link.dataset_key == "data_gov_hk:cc-pricewatch-pricewatch"
                and link.group_field is None)

    class RejectingLLM(FakeLLM):
        def complete(self, prompt, **kwargs):
            if "accepted_ids" in kwargs["system_prompt"]:
                return SimpleNamespace(content='{"accepted_ids":[]}')
            return super().complete(prompt, **kwargs)

    proposals = propose_llm_links(
        RejectingLLM(), (SceneNode("商品价格核对"), SceneNode("食品价格核对")),
        links=(base,), max_datasets=1,
    )
    assert not proposals


def test_llm_link_registry_append_is_idempotent(tmp_path):
    path = tmp_path / "links.json"
    path.write_text(json.dumps({"version": 1, "links": []}), encoding="utf-8")
    proposal = {"scene": "食品价格核对", "dataset_key": "data_gov_hk:example",
                "source_sha256": "a" * 64, "evidence": "原始食品价格字段支持该场景",
                "business_label": "食品价格"}
    assert append_llm_links((proposal,), path) == 1
    assert append_llm_links((proposal,), path) == 0
    assert len(reviewed_links(path)) == 1


def test_llm_linker_can_start_from_approved_source_without_manual_link(monkeypatch):
    source_key = "kaggle:vinamratas29/bangalore-food-delivery-orders-clean-dataset"
    monkeypatch.setattr(llm_dataset_linker, "reviewed_links", lambda: tuple(
        link for link in reviewed_links() if link.dataset_key != source_key))

    class DeliveryLLM:
        model = "fake-external-model"

        def complete(self, prompt, **kwargs):
            context = json.loads(prompt)
            assert context["fields"] == {"identifier": "ID", "group": "Type_of_order",
                                         "numeric_value": "delivery_time_mins"}
            if "accepted_ids" in kwargs["system_prompt"]:
                return SimpleNamespace(content='{"accepted_ids":[0]}')
            return SimpleNamespace(content=json.dumps({"links": [{
                "scene": "点外卖", "group_value": None, "business_label": "外卖订单",
                "reason": "订单类型和配送时长支持外卖配送场景",
            }]}, ensure_ascii=False))

    proposals = propose_llm_links(
        DeliveryLLM(), (SceneNode("点外卖"),), only_dataset_keys=(source_key,),
    )
    assert len(proposals) == 1
    assert proposals[0]["dataset_key"] == source_key
    assert "group_field" not in proposals[0]


def test_read_only_link_run_never_downloads_missing_source(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_dataset_linker, "KAGGLE_ROOT", tmp_path)
    source_key = "kaggle:vinamratas29/bangalore-food-delivery-orders-clean-dataset"
    with pytest.raises(TaskGenerationError, match="cached raw source"):
        propose_llm_links(object(), (SceneNode("点外卖"),),
                          only_dataset_keys=(source_key,), require_local_sources=True)
