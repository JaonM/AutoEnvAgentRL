import csv
import json
import random
from pathlib import Path

import pytest

from env_factory.generation.dataset_task_generator import (
    DatasetTaskGenerator, TaskGenerationError, _choose_rows, _column_name, _columns, _description,
    _project, _sample_source, _validate_scene, _validate_voice,
)


def test_csv_source_supports_all_three_training_routes(tmp_path: Path) -> None:
    source = tmp_path / "orders.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "Order ID", "Customer Name", "Product Category", "Total Amount",
        ])
        writer.writeheader()
        for number in range(1, 31):
            writer.writerow({
                "Order ID": f"O{number}", "Customer Name": f"Private {number}",
                "Product Category": "Books" if number <= 20 else "Food",
                "Total Amount": number * 10,
            })
    headers, source_rows = _sample_source(source)
    key, group, numeric = _columns(headers, source_rows)
    assert (key, group, numeric) == ("Order ID", "Product Category", "Total Amount")
    for category, operations in (("direct_response", 0), ("simple_agentic", 1),
                                 ("multi_step_agentic", 2)):
        chosen = _choose_rows(source_rows, key, group, numeric, category, random.Random(4))
        rows = _project(chosen, key, group, numeric)
        description, facts = _description(
            category, rows, *map(_column_name, (key, group, numeric)), "Orders"
        )
        assert len(description["route_plan"]["environment_operations"]) == operations
        assert "Private" not in str(rows)
        assert facts
        public = description["public_input"]
        assert public["initial_user_message"] == description["task"]
        assert "Orders" not in description["task"]
        assert "编号" not in description["task"]
        assert "记录" not in description["task"]
        assert rows[0]["order_id"] not in description["task"]
        assert public["materials"]
        assert "凭证" not in description["task"]
        material = json.loads(public["materials"][0]["content"])
        if category == "direct_response":
            assert {item["标签"] for item in material} == {"A", "B"}
            assert all("order_id" not in item for item in material)
            assert facts["lower_label"] in {"A", "B"}
        else:
            assert rows[0]["order_id"] in material.values()
        if category == "multi_step_agentic":
            assert facts["starting_record_id"] != facts["lowest_record_id"]
            assert facts["lowest_value"] == min(
                row["total_amount"] for row in rows if row["product_category"] == facts["group_value"]
            )


def test_kaggle_source_selects_non_csv_manifest_file(tmp_path: Path, monkeypatch) -> None:
    import env_factory.generation.dataset_task_generator as module

    root = tmp_path / "kaggle"
    version = root / "owner" / "orders" / "v1"
    raw = version / "raw"
    raw.mkdir(parents=True)
    source = raw / "orders.jsonl"
    source.write_text("\n".join(json.dumps({
        "order_id": f"O{index}", "category": "Books", "amount": index * 10,
    }) for index in range(5)) + "\n", encoding="utf-8")
    version.joinpath("source_manifest.json").write_text(json.dumps({
        "title": "Orders", "source": "https://www.kaggle.com/datasets/owner/orders",
        "license": "CC0: Public Domain", "files": [{"path": source.name}],
    }), encoding="utf-8")
    monkeypatch.setattr(module, "KAGGLE_ROOT", root)
    generator = DatasetTaskGenerator.__new__(DatasetTaskGenerator)
    generator.dataset_ref = "owner/orders"
    generator.dataset_file = None
    selected, title, url, license_name = generator._source(random.Random(1))
    assert selected == source
    assert title == "Orders"
    assert url.endswith("owner/orders")
    assert license_name == "CC0: Public Domain"
    assert len(_sample_source(selected)[1]) == 5


def test_projected_contact_details_are_rejected_before_model_use() -> None:
    with pytest.raises(TaskGenerationError, match="private or unsafe"):
        _project([{"Order ID": "A1", "Product Category": "contact@example.com",
                   "Total Amount": "10"}],
                 "Order ID", "Product Category", "Total Amount")


def test_voice_rejects_dataset_ids_and_invented_vouchers() -> None:
    rows = [{"transaction_id": "79", "product_category": "Beauty", "total_amount": 150.0}]
    facts = {"starting_record_id": "79", "group_value": "Beauty",
             "lowest_record_id": "23", "lowest_value": 25.0}
    for message, reason in (
        ("帮我查一下编号79的交易属于什么类别，最低是多少钱？", "source-oriented"),
        ("我这两笔消费分别属于什么类别？最低多少钱？", "second starting voucher"),
        ("这笔消费 79 属于哪个类别？最低金额是多少？", "source record ID"),
        ("帮我查一下这笔消费类别，再看看同类最低金额。", "lookup procedure"),
        ("我在核对这笔消费，想先确认类别然后比较同类最低金额。", "lookup procedure"),
        ("我在核对这笔消费，想知道同类最低金额是多少。", "minimum item's identity"),
        ("我在评估营销活动的贡献，想知道这笔消费同类最低金额是多少？", "unsupported business context"),
    ):
        with pytest.raises(TaskGenerationError, match=reason):
            _validate_voice(message, "原始请求", "multi_step_agentic", rows,
                            "transaction_id", facts, "Retail Sales Dataset")
    _validate_voice("我在核对这笔消费的品类。它算哪个商品类别，同类中最低的是哪单、多少钱？",
                    "原始请求", "multi_step_agentic", rows, "transaction_id",
                    facts, "Retail Sales Dataset")


def test_business_scenario_contract() -> None:
    scene = {"entity_name": "零售交易", "identifier_label": "交易号",
             "group_label": "商品类别", "value_label": "成交金额",
             "user_role": "顾客", "business_situation": "核对一笔消费"}
    assert _validate_scene(scene) == scene
    with pytest.raises(TaskGenerationError, match="business scenario"):
        _validate_scene({**scene, "entity_name": "数据集记录"})
