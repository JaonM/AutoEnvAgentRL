#!/usr/bin/env python3
"""Build the complete DATA.GOV.HK metadata index; keep approval explicit."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from env_factory.generation.dataset_source_registry import HK_INDEX


PACKAGE_LIST = "https://data.gov.hk/sc-data/api/3/action/package_list"
PACKAGE_SHOW = "https://data.gov.hk/sc-data/api/3/action/package_show"
BULK_LIST = "https://resource.data.one.gov.hk/opendata/open-data-list/open-data-dataset-list-zh-hans.json"
DATASET_ID = re.compile(r"[a-z0-9_-]+")
HEADERS = {
    "User-Agent": "Mozilla/5.0 Chrome/124.0",
    "Referer": "https://data.gov.hk/",
    "Accept": "application/json,*/*",
}


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"official catalog returned invalid JSON: {url.split('?')[0]}")
    return value


def _new_metadata(dataset_id: str) -> dict[str, Any]:
    document = _get_json(PACKAGE_SHOW + "?" + urllib.parse.urlencode({"id": dataset_id}))
    detail = document.get("result") if document.get("success") is True else None
    if not isinstance(detail, dict) or detail.get("name") != dataset_id:
        raise ValueError(f"package_show failed for {dataset_id}")
    resources = [
        {"name": str(item.get("name") or ""), "format": str(item.get("format") or "").upper()}
        for item in detail.get("resources", []) if isinstance(item, dict)
    ]
    groups = detail.get("groups") or []
    category = next((str(item.get("display_name") or item.get("title") or "")
                     for item in groups if isinstance(item, dict)), "")
    organization = detail.get("organization")
    organization_title = organization.get("title") if isinstance(organization, dict) else None
    return {"id": dataset_id, "title": str(detail.get("title") or dataset_id),
            "provider": str(detail.get("author") or organization_title or ""),
            "category": category, "resources": resources}


def build_index(existing: dict[str, Any], live_ids: list[str], bulk: dict[str, Any],
                missing_details: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Join current IDs to the official bulk snapshot and preserve manual approval."""
    if not live_ids or len(live_ids) != len(set(live_ids)) or any(
        DATASET_ID.fullmatch(value) is None for value in live_ids
    ):
        raise ValueError("live package_list is empty, duplicated, or invalid")
    rows = bulk.get("Data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("official bulk catalog has no resource rows")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dataset_id = row.get("Dataset ID")
        if not isinstance(dataset_id, str) or DATASET_ID.fullmatch(dataset_id) is None:
            continue
        entry = grouped.setdefault(dataset_id, {
            "id": dataset_id, "title": str(row.get("Dataset Name") or dataset_id),
            "provider": str(row.get("Data Provider") or ""),
            "category": str(row.get("Category") or ""), "resources": [],
        })
        entry["resources"].append({"name": str(row.get("Resource Name") or ""),
                                   "format": str(row.get("Data Format") or "").upper()})
    missing = set(live_ids) - grouped.keys()
    if set(missing_details) != missing:
        raise ValueError(f"missing package_show metadata for {len(missing ^ set(missing_details))} datasets")
    approved = {row["id"]: row for row in existing.get("datasets", [])
                if isinstance(row, dict) and row.get("approved") is True
                and isinstance(row.get("id"), str)}
    datasets = []
    for dataset_id in sorted(live_ids):
        item = dict(grouped.get(dataset_id) or missing_details[dataset_id])
        item["url"] = f"https://data.gov.hk/sc-data/dataset/{dataset_id}"
        item["resource_count"] = len(item["resources"])
        item["resource_formats"] = sorted({resource["format"] for resource in item["resources"]
                                           if resource["format"]})
        item["metadata_source"] = "bulk" if dataset_id in grouped else "package_show"
        item["approved"] = False
        if dataset_id in approved:
            original = approved[dataset_id]
            for key in ("resource_url", "resource_format", "resource_language", "license", "approved"):
                if key in original:
                    item[key] = original[key]
        datasets.append(item)
    return {
        "version": "2.0", "source": PACKAGE_LIST, "bulk_source": BULK_LIST,
        "bulk_as_of": bulk.get("As of Date"),
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(datasets), "approved_count": sum(row["approved"] for row in datasets),
        "terms": "https://data.gov.hk/sc/terms-and-conditions", "datasets": datasets,
    }


def refresh(index_path: Path = HK_INDEX, *, workers: int = 4) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    live_document = _get_json(PACKAGE_LIST)
    live_ids = live_document.get("result") if live_document.get("success") is True else None
    if not isinstance(live_ids, list):
        raise ValueError("package_list failed")
    bulk = _get_json(BULK_LIST + "?download=1")
    bulk_ids = {row.get("Dataset ID") for row in bulk.get("Data", []) if isinstance(row, dict)}
    missing = sorted(set(live_ids) - bulk_ids)
    details: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_new_metadata, dataset_id): dataset_id for dataset_id in missing}
        for future in as_completed(futures):
            details[futures[future]] = future.result()
    result = build_index(existing, live_ids, bulk, details)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=index_path.parent,
                                     prefix=".data-gov-hk-index-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(index_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = refresh(workers=args.workers)
    print(json.dumps({key: result[key] for key in ("count", "approved_count", "bulk_as_of")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
