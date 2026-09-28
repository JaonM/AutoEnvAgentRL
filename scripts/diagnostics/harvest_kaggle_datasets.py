#!/usr/bin/env python3
"""Build a bounded Kaggle catalog index; download only when explicitly requested."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = Path(__file__).with_name("download_kaggle_dataset.py")
DEFAULT_QUERIES = ("retail sales", "travel booking", "inventory", "customer support", "ecommerce orders")
ACCEPTED_LICENSES = {
    "CC0: Public Domain",
    "Attribution 4.0 International (CC BY 4.0)",
    "Apache 2.0",
    "MIT",
}


def catalog_page(query: str, page: int, max_bytes: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "search": query, "sortBy": "hottest", "filetype": "csv",
        "page": page, "maxSize": max_bytes,
    })
    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/list?{params}",
        headers={"User-Agent": "EnvFactory/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, list):
        raise ValueError("Kaggle catalog did not return a list")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Repeat to select catalog searches")
    parser.add_argument("--pages", type=int, default=2, help="Pages per query (default: 2)")
    parser.add_argument("--limit", type=int, default=20, help="Maximum new datasets to download")
    parser.add_argument("--max-dataset-mb", type=int, default=25)
    parser.add_argument("--max-total-mb", type=int, default=250)
    parser.add_argument("--download", action="store_true", help="Download selected datasets after indexing")
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "sources" / "kaggle")
    args = parser.parse_args()
    if min(args.pages, args.limit, args.max_dataset_mb, args.max_total_mb) < 1:
        parser.error("all limits must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict] = {}
    for query in args.query or DEFAULT_QUERIES:
        for page in range(1, args.pages + 1):
            for item in catalog_page(query, page, args.max_dataset_mb * 1_000_000):
                ref = item.get("ref")
                size = item.get("totalBytes")
                if (not isinstance(ref, str) or not isinstance(size, int)
                    or size < 1 or size > args.max_dataset_mb * 1_000_000
                    or item.get("licenseName") not in ACCEPTED_LICENSES):
                    continue
                candidates.setdefault(ref, {
                    "ref": ref, "title": item.get("title"), "bytes": size,
                    "license": item.get("licenseName"), "matched_query": query,
                })
    catalog_path = args.root / "catalog_candidates.json"
    catalog_path.write_text(json.dumps({
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": args.query or list(DEFAULT_QUERIES), "pages_per_query": args.pages,
        "candidates": list(candidates.values()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(candidates)} candidates: {catalog_path}")
    if not args.download:
        return
    log_path = args.root / "harvest_results.jsonl"
    downloaded = 0
    reserved_bytes = 0
    with log_path.open("a", encoding="utf-8") as log:
        for item in candidates.values():
            if downloaded >= args.limit:
                break
            if reserved_bytes + item["bytes"] > args.max_total_mb * 1_000_000:
                continue
            result = subprocess.run(
                [sys.executable, str(DOWNLOAD_SCRIPT), item["ref"],
                 "--root", str(args.root), "--max-bytes", str(args.max_dataset_mb * 1_000_000)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0:
                status = "downloaded"
                downloaded += 1
                reserved_bytes += item["bytes"]
            elif "already downloaded:" in result.stderr:
                status = "already_present"
            else:
                status = "failed"
            record = {**item, "status": status,
                      "checked_at_utc": datetime.now(timezone.utc).isoformat()}
            if status == "failed":
                record["error"] = result.stderr.strip()[-500:]
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{status}: {item['ref']}", flush=True)
    print(f"New downloads: {downloaded}; candidate index: {catalog_path}; log: {log_path}")


if __name__ == "__main__":
    main()
