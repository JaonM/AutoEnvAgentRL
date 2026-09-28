"""Downloaded sources can create incremental provisional graph relations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from env_factory.generation.dataset_task_generator import TaskGenerationError
from env_factory.graph.knowledge_graph import SceneNode
from env_factory.graph.local_dataset_linker import (
    _numeric_semantics_match, build_local_candidate_links, propose_local_candidates, usable_table,
)


PROMOTE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnostics/promote_local_graph_link.py"
PROMOTE_SPEC = importlib.util.spec_from_file_location("promote_local_graph_link", PROMOTE_SCRIPT)
assert PROMOTE_SPEC and PROMOTE_SPEC.loader
promote_module = importlib.util.module_from_spec(PROMOTE_SPEC)
PROMOTE_SPEC.loader.exec_module(promote_module)


class FakeLLM:
    calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if "accepted_ids" in kwargs["system_prompt"]:
            return SimpleNamespace(content='{"accepted_ids":[0]}')
        assert "订单" in json.loads(prompt)["existing_scenes"]
        return SimpleNamespace(content=json.dumps({"links": [{
            "scene": "订单", "group_value": None,
            "business_label": "零售订单", "reason": "订单编号、渠道和金额支持订单查询业务",
        }]}))


class FakeStore:
    def __init__(self):
        self.fingerprints = {}
        self.saved = []
        self.scenes = []

    def local_link_fingerprint(self, key):
        return self.fingerprints.get(key)

    def upsert_scene(self, scene):
        self.scenes.append(scene)

    def save_local_link_candidates(self, key, fingerprint, source_hash, candidates):
        self.saved.append((key, candidates))
        self.fingerprints[key] = fingerprint


def _download(root: Path) -> Path:
    target = root / "shop/orders/v1"
    raw = target / "raw"
    raw.mkdir(parents=True)
    payload = raw / "orders.csv"
    payload.write_text("order_id,channel,amount\n1,web,10\n2,web,20\n3,web,30\n4,store,40\n5,store,50\n6,store,60\n",
                       encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = target / "source_manifest.json"
    manifest.write_text(json.dumps({"ref": "shop/orders", "title": "零售订单",
                                    "files": [{"path": "orders.csv", "bytes": payload.stat().st_size,
                                               "sha256": digest}]}), encoding="utf-8")
    return manifest


def test_local_candidate_relations_are_incremental_and_not_reviewed(tmp_path: Path) -> None:
    kaggle_root = tmp_path / "kaggle"
    manifest = _download(kaggle_root)
    key = "kaggle:shop/orders"
    store = FakeStore()
    llm = FakeLLM()
    args = (store, llm, (SceneNode("订单"),), ({"key": key, "approved": False},))
    kwargs = {"kaggle_root": kaggle_root, "hk_root": tmp_path / "hk"}
    first = build_local_candidate_links(*args, **kwargs)
    assert first["audited"] == first["candidates"] == 1
    assert store.saved[0][1][0]["scene"] == "订单"
    assert store.saved[0][1][0]["group_field"] is None
    assert "reviewed" not in store.saved[0][1][0]
    assert build_local_candidate_links(*args, **kwargs)["unchanged"] == 1
    assert llm.calls == 2
    assert usable_table(manifest, key)[0].name == "orders.csv"
    (manifest.parent / "raw/orders.csv").write_text("changed", encoding="utf-8")
    assert build_local_candidate_links(*args, **kwargs)["failed"] == 1


def test_approved_downloads_are_not_provisional(tmp_path: Path) -> None:
    kaggle_root = tmp_path / "kaggle"
    _download(kaggle_root)
    result = build_local_candidate_links(
        FakeStore(), FakeLLM(), (SceneNode("订单"),),
        ({"key": "kaggle:shop/orders", "approved": True},),
        kaggle_root=kaggle_root, hk_root=tmp_path / "hk",
    )
    assert result["ineligible"] == 1 and result["audited"] == 0


def test_unusable_download_is_skipped_until_local_files_change(tmp_path: Path) -> None:
    root = tmp_path / "kaggle"
    manifest = _download(root)
    raw = manifest.parent / "raw/orders.csv"
    raw.write_text("name\nitem\n", encoding="utf-8")
    document = json.loads(manifest.read_text())
    document["files"][0].update(bytes=raw.stat().st_size,
                                sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(document), encoding="utf-8")
    store = FakeStore()
    args = (store, FakeLLM(), (SceneNode("订单"),),
            ({"key": "kaggle:shop/orders", "approved": False},))
    kwargs = {"kaggle_root": root, "hk_root": tmp_path / "hk", "max_datasets": 1}
    assert build_local_candidate_links(*args, **kwargs)["unusable"] == 1
    assert build_local_candidate_links(*args, **kwargs)["unchanged"] == 1


def test_requested_local_dataset_must_exist(tmp_path: Path) -> None:
    with pytest.raises(TaskGenerationError, match="not downloaded"):
        build_local_candidate_links(
            FakeStore(), FakeLLM(), (SceneNode("订单"),), (),
            only_keys=("kaggle:missing/data",),
            kaggle_root=tmp_path / "kaggle", hk_root=tmp_path / "hk",
        )


def test_sales_values_do_not_support_cost_or_profit_scenes() -> None:
    assert not _numeric_semantics_match("家具商品成本核对", "Sales")
    assert not _numeric_semantics_match("订单利润核对", "Revenue")
    assert _numeric_semantics_match("家具商品成本核对", "Product Cost")


def test_grouped_candidate_keeps_original_group_field() -> None:
    class GroupLLM(FakeLLM):
        def complete(self, prompt, **kwargs):
            if "accepted_ids" in kwargs["system_prompt"]:
                return SimpleNamespace(content='{"accepted_ids":[0]}')
            return SimpleNamespace(content=json.dumps({"links": [{
                "scene": "网店订单", "group_value": "web", "business_label": "网店订单",
                "reason": "channel字段标明网店订单，amount提供金额",
            }]}))

    rows = [{"order_id": str(i), "channel": "web" if i <= 3 else "store",
             "amount": str(i * 10)} for i in range(1, 7)]
    selected = propose_local_candidates(GroupLLM(), (SceneNode("网店订单"),),
                                        "订单", ["order_id", "channel", "amount"], rows)
    assert selected[0]["group_field"] == "channel"
    assert selected[0]["group_value"] == "web"


def test_promotion_requires_explicit_existing_scene_and_source_approval(monkeypatch) -> None:
    candidate = {"scene": "订单", "source_sha256": "a" * 64,
                 "evidence": "原始字段支持订单查询", "business_label": "零售订单",
                 "group_field": None, "group_value": None}
    calls = []
    monkeypatch.setattr(promote_module, "verify_link_source", lambda link, **kwargs: calls.append(link))
    rows = promote_module.promotion_rows("kaggle:shop/orders", (candidate,), ["订单"])
    assert len(rows) == len(calls) == 1
    assert rows[0]["review_method"] == "llm_source_grounded_v2"
    with pytest.raises(ValueError):
        promote_module.promotion_rows("kaggle:shop/orders", (candidate,), ["不存在"])
