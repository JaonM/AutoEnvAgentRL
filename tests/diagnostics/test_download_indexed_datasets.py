"""Focused checks for catalog-wide raw dataset downloads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnostics/download_indexed_datasets.py"
SPEC = importlib.util.spec_from_file_location("download_indexed_datasets", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_select_all_indexed_sources_and_approved_subset(tmp_path: Path) -> None:
    kaggle = _json(tmp_path / "kaggle.json", {"count": 2, "datasets": [
        {"ref": "a/one", "license": "MIT"}, {"ref": "b/two", "license": "CC0"}]})
    hk = _json(tmp_path / "hk.json", {"count": 2, "datasets": [
        {"id": "hk-one", "approved": False}, {"id": "hk-two", "approved": True}]})
    allow = _json(tmp_path / "allow.json", {"datasets": [{"ref": "a/one", "license": "MIT"}]})
    assert len(module.selected_items("both", "all", kaggle, hk, allow)) == 4
    assert [row["ref" if source == "kaggle" else "id"] for source, row in
            module.selected_items("both", "approved", kaggle, hk, allow)] == ["a/one", "hk-two"]
    assert module.requested_items(["data_gov_hk:hk-one", "kaggle:b/two"], "both", kaggle, hk) == [
        ("data_gov_hk", {"id": "hk-one", "approved": False}),
        ("kaggle", {"ref": "b/two", "license": "CC0"}),
    ]
    with pytest.raises(ValueError, match="absent from"):
        module.requested_items(["data_gov_hk:hk-one"], "kaggle", kaggle, hk)
    with pytest.raises(ValueError, match="duplicate"):
        module.requested_items(["kaggle:a/one", "kaggle:a/one"], "both", kaggle, hk)


def test_resume_checks_hash_and_indexed_license(tmp_path: Path) -> None:
    root = tmp_path / "kaggle"
    version = root / "a/one/v1"
    (version / "raw").mkdir(parents=True)
    payload = version / "raw/data.csv"
    payload.write_bytes(b"a,b\n1,2\n")
    _json(version / "source_manifest.json", {"ref": "a/one", "license": "MIT", "files": [
        {"path": "data.csv", "bytes": payload.stat().st_size, "sha256": module.sha256_file(payload)}]})
    kwargs = {"kaggle_root": root, "hk_root": tmp_path / "hk", "state_file": tmp_path / "state.jsonl"}
    row = ("kaggle", {"ref": "a/one", "license": "MIT"})
    assert module.download_selected([row], **kwargs)["skipped"] == 1
    assert module.download_selected([("kaggle", {"ref": "a/one", "license": "CC0"})], **kwargs)["failed"] == 1
    payload.write_bytes(b"changed")
    assert module.download_selected([row], **kwargs)["failed"] == 1
    assert module.read_status(kwargs["state_file"])["latest_status_counts"] == {"invalid_local": 1}


def test_hk_downloads_every_resource_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"id": "sample-data", "resource_count": 2, "url": "https://data.gov.hk/sample-data"}
    monkeypatch.setattr(module, "_request_json", lambda _: {"success": True, "result": {
        "name": "sample-data", "title": "Sample", "resources": [
            {"id": "first", "url": "https://example.com/one.csv", "format": "CSV"},
            {"id": "second", "url": "https://example.com/two.zip", "format": "ZIP"}]}})

    def fake_download(url: str, target: Path, max_bytes: int) -> tuple[int, str]:
        target.write_bytes(url.encode())
        return target.stat().st_size, module.sha256_file(target)

    monkeypatch.setattr(module, "_download_file", fake_download)
    manifest = module.download_hk_dataset(row, tmp_path / "hk", 1000)
    assert module.verify_manifest(manifest, source_key="data_gov_hk:sample-data") > 0
    assert {entry["format"] for entry in json.loads(manifest.read_text())["files"]} == {"CSV", "ZIP"}


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/data", "http://localhost/data"])
def test_rejects_non_public_resource_urls(url: str) -> None:
    with pytest.raises(ValueError):
        module._public_url(url)
