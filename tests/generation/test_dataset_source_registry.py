import hashlib
import io
import json
import random
from pathlib import Path

import pytest

from env_factory.generation.dataset_source_registry import (
    balanced_platforms, eligible_hk_ids, verified_hk_source,
)
from scripts.diagnostics.download_data_gov_hk_dataset import download_dataset


DATASET_ID = "cc-pricewatch-pricewatch"
RESOURCE_URL = "https://online-price-watch.consumer.org.hk/opw/opendata/pricewatch_zh-Hans.csv"


def _index(path: Path) -> None:
    path.write_text(json.dumps({"datasets": [{
        "id": DATASET_ID, "url": f"https://data.gov.hk/sc-data/dataset/{DATASET_ID}",
        "title": "格价资讯通", "approved": True, "resource_format": ".csv",
        "resource_url": RESOURCE_URL, "license": "DATA.GOV.HK terms",
    }]}), encoding="utf-8")


def test_balanced_platform_allocation_is_exact_and_seeded() -> None:
    first = balanced_platforms(20, rng=random.Random(7))
    assert first == balanced_platforms(20, rng=random.Random(7))
    assert first.count("kaggle") == first.count("data_gov_hk") == 10
    with pytest.raises(ValueError):
        balanced_platforms(3, rng=random.Random(7))


def test_downloaded_source_is_verified_against_catalog_and_manifest(tmp_path: Path, monkeypatch) -> None:
    payload = "货品编号,货品分类,价格\nA,面包,10\nB,蛋糕,20\n".encode()
    index = tmp_path / "index.json"
    root = tmp_path / "sources"
    _index(index)
    assert eligible_hk_ids(index_path=index) == [DATASET_ID]
    detail = {"success": True, "result": {"name": DATASET_ID,
              "resources": [{"url": RESOURCE_URL, "format": "CSV"}]}}

    def fake_urlopen(request, timeout):
        if "package_show" in request.full_url:
            return io.BytesIO(json.dumps(detail).encode())
        assert request.full_url == RESOURCE_URL
        return io.BytesIO(payload)

    monkeypatch.setattr("scripts.diagnostics.download_data_gov_hk_dataset.urllib.request.urlopen",
                        fake_urlopen)
    manifest = download_dataset(DATASET_ID, root=root, index_path=index, max_bytes=100)
    assert manifest["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    source = verified_hk_source(DATASET_ID, root=root, index_path=index,
                                bulk_root=tmp_path / "bulk")
    assert source is not None and source[0].read_bytes() == payload
    source[0].write_bytes(b"tampered")
    assert verified_hk_source(DATASET_ID, root=root, index_path=index,
                              bulk_root=tmp_path / "bulk") is None


def test_approved_bulk_download_can_be_verified_without_redownload(tmp_path: Path) -> None:
    payload = b"id,category,price\n1,A,10\n2,A,20\n3,A,30\n"
    index = tmp_path / "index.json"
    _index(index)
    bulk = tmp_path / "bulk" / DATASET_ID
    (bulk / "raw").mkdir(parents=True)
    (bulk / "raw/data.csv").write_bytes(payload)
    (bulk / "source_manifest.json").write_text(json.dumps({
        "id": DATASET_ID, "source": f"https://data.gov.hk/sc-data/dataset/{DATASET_ID}",
        "files": [{"path": "data.csv", "resource_url": RESOURCE_URL,
                   "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")
    source = verified_hk_source(DATASET_ID, root=tmp_path / "approved",
                                bulk_root=tmp_path / "bulk", index_path=index)
    assert source is not None and source[0].read_bytes() == payload


def test_download_rejects_changed_resource_metadata(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "index.json"
    _index(index)
    detail = {"success": True, "result": {"name": DATASET_ID,
              "resources": [{"url": "https://example.org/other.csv", "format": "CSV"}]}}
    monkeypatch.setattr("scripts.diagnostics.download_data_gov_hk_dataset.urllib.request.urlopen",
                        lambda request, timeout: io.BytesIO(json.dumps(detail).encode()))
    with pytest.raises(ValueError, match="changed"):
        download_dataset(DATASET_ID, root=tmp_path / "sources", index_path=index)
