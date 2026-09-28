#!/usr/bin/env python3
"""Download an approved DATA.GOV.HK resource after checking official CKAN metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from env_factory.generation.dataset_source_registry import HK_INDEX, hk_catalog, verified_hk_source


DEFAULT_ROOT = HK_INDEX.parent
API = "https://data.gov.hk/sc-data/api/3/action/package_show"


def download_dataset(dataset_id: str, *, root: Path = DEFAULT_ROOT,
                     index_path: Path = HK_INDEX, max_bytes: int = 5_000_000_000) -> dict:
    if not re.fullmatch(r"[a-z0-9_-]+", dataset_id) or max_bytes <= 0:
        raise ValueError("invalid DATA.GOV.HK dataset ID or size limit")
    indexed = hk_catalog(index_path).get(dataset_id)
    if indexed is None or indexed.get("approved") is not True:
        raise ValueError("dataset is not in the approved DATA.GOV.HK catalog")
    source = verified_hk_source(dataset_id, root=root, index_path=index_path,
                                bulk_root=root.parent / "data_gov_hk_bulk")
    if source is not None:
        return json.loads((source[0].parent.parent / "source_manifest.json").read_text(encoding="utf-8"))
    request = urllib.request.Request(API + "?" + urllib.parse.urlencode({"id": dataset_id}),
                                     headers={"User-Agent": "EnvFactory/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        document = json.load(response)
    detail = document.get("result") if document.get("success") is True else None
    if not isinstance(detail, dict) or detail.get("name") != dataset_id:
        raise ValueError("official catalog did not confirm the dataset identity")
    resource_url = indexed.get("resource_url")
    resources = detail.get("resources") or []
    if not any(isinstance(item, dict) and item.get("url") == resource_url
               and str(item.get("format", "")).lower() == indexed.get("resource_format", "").lstrip(".")
               for item in resources):
        raise ValueError("approved resource URL or format changed in the official catalog")
    parsed = urllib.parse.urlparse(resource_url)
    if parsed.scheme != "https" or parsed.hostname != "online-price-watch.consumer.org.hk":
        raise ValueError("resource host is not approved for direct download")
    name = Path(parsed.path).name
    if name != "pricewatch_zh-Hans.csv":
        raise ValueError("unexpected resource filename")
    target = root / dataset_id / "raw" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".data-gov-hk-", dir=target.parent, delete=False) as staged:
        temporary = Path(staged.name)
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(resource_url, headers={"User-Agent": "EnvFactory/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("DATA.GOV.HK resource exceeds the configured size limit")
                    digest.update(block)
                    staged.write(block)
            if size == 0:
                raise ValueError("DATA.GOV.HK resource is empty")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        "source": indexed["url"], "id": dataset_id, "title": indexed["title"],
        "license": indexed["license"], "resource_url": resource_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": name, "bytes": size, "sha256": digest.hexdigest()}],
    }
    (root / dataset_id / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--max-bytes", type=int, default=5_000_000_000)
    args = parser.parse_args()
    try:
        manifest = download_dataset(args.dataset_id, max_bytes=args.max_bytes)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({key: manifest[key] for key in ("source", "id", "files")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
