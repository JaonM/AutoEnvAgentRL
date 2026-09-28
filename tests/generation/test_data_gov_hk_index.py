import pytest
import json

from env_factory.generation.dataset_source_registry import HK_INDEX, eligible_hk_ids, hk_catalog
from scripts.diagnostics.index_data_gov_hk_datasets import build_index


def test_full_catalog_join_preserves_only_reviewed_approval() -> None:
    old = {"datasets": [{"id": "old", "approved": True, "resource_url": "https://data.gov.hk/old.csv",
                         "resource_format": ".csv", "license": "reviewed"}]}
    bulk = {"As of Date": "2026-07-03", "Data": [
        {"Dataset ID": "old", "Dataset Name": "旧数据", "Data Provider": "机构",
         "Category": "商业", "Resource Name": "明细", "Data Format": "CSV"},
        {"Dataset ID": "old", "Dataset Name": "旧数据", "Data Provider": "机构",
         "Category": "商业", "Resource Name": "字典", "Data Format": "PDF"},
    ]}
    fresh = {"new": {"id": "new", "title": "新增", "provider": "机构", "category": "运输",
                     "resources": [{"name": "路线", "format": "JSON"}]}}
    result = build_index(old, ["old", "new"], bulk, fresh)
    assert result["count"] == 2
    assert result["approved_count"] == 1
    entries = {item["id"]: item for item in result["datasets"]}
    assert entries["old"]["resource_formats"] == ["CSV", "PDF"]
    assert entries["old"]["resource_count"] == 2
    assert entries["old"]["resource_url"] == "https://data.gov.hk/old.csv"
    assert entries["new"]["approved"] is False
    assert entries["new"]["metadata_source"] == "package_show"


def test_full_catalog_join_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError, match="missing package_show"):
        build_index({}, ["old", "new"], {"Data": [{"Dataset ID": "old"}]}, {})


def test_committed_catalog_is_complete_and_keeps_approval_explicit() -> None:
    document = json.loads(HK_INDEX.read_text(encoding="utf-8"))
    catalog = hk_catalog()
    assert document["count"] == len(document["datasets"]) == len(catalog) == 3822
    assert document["approved_count"] == 1
    assert eligible_hk_ids() == ["cc-pricewatch-pricewatch"]
    assert all(item["resource_count"] == len(item["resources"])
               for item in document["datasets"])
