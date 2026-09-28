#!/usr/bin/env python3
"""Index public Heywhale dataset metadata without downloading dataset files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data" / "sources" / "heywhale"
API = "https://www.heywhale.com/api/datasets"
CHINESE = re.compile(r"[\u3400-\u9fff]")
DATASET_ID = re.compile(r"[a-f0-9]{24}")


def fetch_page(page: int) -> dict:
    params = urllib.parse.urlencode({
        "page": page, "perPage": 100, "Visible": "true",
        "needMeta": "true", "sort": "-UpdateDate",
    })
    request = urllib.request.Request(f"{API}?{params}", headers={
        "User-Agent": "EnvFactory dataset metadata indexer/1.0",
        "Accept": "application/json",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise ValueError(f"unexpected Heywhale response for page {page}")
            return result
        except urllib.error.HTTPError as exc:
            if attempt == 3 or exc.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def names_from_meta(item: dict, key: str) -> list[str]:
    names = []
    for entry in item.get("metaInformation") or []:
        if not isinstance(entry, dict) or entry.get("Key") != key:
            continue
        for value in entry.get("Value") or []:
            if isinstance(value, dict) and isinstance(value.get("Name"), str):
                names.append(value["Name"])
    return list(dict.fromkeys(names))


def simplify(item: dict) -> dict | None:
    dataset_id = item.get("_id")
    if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
        return None
    title = str(item.get("Title") or "").strip()
    description = str(item.get("ShortDescription") or "").strip()[:1000]
    files = item.get("Files") or []
    extensions = sorted({str(f.get("Ext") or "").lower() for f in files if isinstance(f, dict)})
    license_value = item.get("License")
    if isinstance(license_value, dict):
        license_value = license_value.get("Name") or license_value.get("Title") or license_value.get("_id")
    creator = item.get("Creator") or {}
    return {
        "id": dataset_id,
        "url": f"https://www.heywhale.com/home/dataset/{dataset_id}",
        "title": title,
        "description": description,
        "creator": creator.get("Name") if isinstance(creator, dict) else None,
        "topics": names_from_meta(item, "Topic"),
        "fields": names_from_meta(item, "Field"),
        "file_extensions": extensions,
        "file_count": len(files),
        "listed_bytes": item.get("DatasetUsage"),
        "license": license_value,
        "download_enabled": bool(item.get("EnableDownload")),
        "published_at": item.get("PublishDate"),
        "updated_at": item.get("UpdateDate"),
        "download_count": item.get("DownloadCount"),
        "chinese_title": bool(CHINESE.search(title)),
        "chinese_text": bool(CHINESE.search(title + " " + description)),
    }


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY, data_json TEXT NOT NULL, first_page INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pages (
            page INTEGER PRIMARY KEY, result_count INTEGER NOT NULL,
            reported_total INTEGER NOT NULL, fetched_at_utc TEXT NOT NULL
        );
    """)
    return db


def store_page(db: sqlite3.Connection, page: int, payload: dict) -> None:
    rows = payload["data"]
    total = payload.get("totalNum")
    if not isinstance(total, int):
        raise ValueError("Heywhale page has no numeric totalNum")
    with db:
        for item in rows:
            record = simplify(item)
            if record:
                db.execute("INSERT OR IGNORE INTO datasets (id,data_json,first_page) VALUES (?,?,?)",
                           (record["id"], json.dumps(record, ensure_ascii=False), page))
        db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?)",
                   (page, len(rows), total, datetime.now(timezone.utc).isoformat()))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def export(db: sqlite3.Connection, root: Path) -> tuple[int, int]:
    rows = [json.loads(row[0]) for row in db.execute(
        "SELECT data_json FROM datasets ORDER BY first_page,id")]
    chinese = [row for row in rows if row["chinese_title"]]
    latest_total = db.execute("SELECT reported_total FROM pages ORDER BY page DESC LIMIT 1").fetchone()
    metadata = {
        "source": "https://www.heywhale.com/home/dataset",
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only": True,
        "reported_total": latest_total[0] if latest_total else None,
        "pages_fetched": db.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
    }
    write_json(root / "dataset_index.json", {**metadata, "count": len(rows), "datasets": rows})
    write_json(root / "chinese_dataset_index.json", {
        **metadata, "selection": "title contains a Chinese character",
        "count": len(chinese), "datasets": chinese,
    })
    return len(rows), len(chinese)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between page requests")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-pages", type=int, help="Optional bound for a partial crawl")
    args = parser.parse_args()
    if args.delay < 0 or args.workers < 1 or args.max_pages is not None and args.max_pages < 1:
        parser.error("delay must be nonnegative and max-pages must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    db = connect(args.root / "dataset_index.sqlite3")
    first = db.execute("SELECT reported_total FROM pages WHERE page=1").fetchone()
    if first is None:
        first_payload = fetch_page(1)
        store_page(db, 1, first_payload)
        total = first_payload["totalNum"]
    else:
        total = first[0]
    page_count = math.ceil(total / 100)
    if args.max_pages:
        page_count = min(page_count, args.max_pages)
    missing = [page for page in range(2, page_count + 1)
               if not db.execute("SELECT 1 FROM pages WHERE page=?", (page,)).fetchone()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for page in missing:
            pending[pool.submit(fetch_page, page)] = page
            if args.delay:
                time.sleep(args.delay)
        for future in concurrent.futures.as_completed(pending):
            page = pending[future]
            store_page(db, page, future.result())
            completed = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            if completed % 20 == 0 or completed == page_count:
                count = db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
                print(f"pages={completed}/{page_count} indexed={count}", flush=True)
    total_indexed, chinese_indexed = export(db, args.root)
    print(f"Indexed {total_indexed} public datasets; {chinese_indexed} have Chinese titles", flush=True)


if __name__ == "__main__":
    main()
