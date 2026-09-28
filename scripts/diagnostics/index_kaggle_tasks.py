#!/usr/bin/env python3
"""Build a resumable, metadata-only Kaggle catalog for task generation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data" / "sources" / "kaggle"
LICENSES = {
    "CC0: Public Domain", "Attribution 4.0 International (CC BY 4.0)",
    "Apache 2.0", "MIT",
}
QUERIES = (
    "retail", "travel", "inventory", "customer support", "ecommerce",
    "orders", "booking", "hotel", "airline", "restaurant", "finance",
    "banking", "insurance", "logistics", "shipping", "healthcare",
    "education", "housing", "energy", "transport", "products", "tickets",
    "transactions", "supply chain", "telecom", "sales", "movies", "music",
    "sports", "weather", "jobs", "payments", "vehicles", "flights",
    "food", "events", "reviews", "public services", "real estate",
    "manufacturing", "agriculture", "books", "games", "customers",
    "stores", "vendors", "warehouses", "appointments", "reservations",
)


def fetch_page(query: str, sort: str, page: int, max_bytes: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "search": query, "sortBy": sort, "filetype": "csv",
        "page": page, "maxSize": max_bytes,
    })
    url = f"https://www.kaggle.com/api/v1/datasets/list?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "EnvFactory/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if not isinstance(result, list):
                raise ValueError("Kaggle returned a non-list catalog page")
            return result
        except urllib.error.HTTPError as exc:
            # Kaggle can return an HTML 404 for otherwise valid pages under load.
            # Do not mistake that response for the end of a catalog partition.
            if attempt == 4 or exc.code not in (404, 429, 500, 502, 503, 504):
                raise
            time.sleep(min(2 ** (attempt + 1), 20))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError("unreachable")


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            ref TEXT PRIMARY KEY, title TEXT, bytes INTEGER NOT NULL,
            license TEXT NOT NULL, matched_query TEXT NOT NULL,
            sort_by TEXT NOT NULL, first_seen_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pages (
            query TEXT NOT NULL, sort_by TEXT NOT NULL, page INTEGER NOT NULL,
            result_count INTEGER NOT NULL, fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (query, sort_by, page)
        );
    """)
    return db


def save_page(db: sqlite3.Connection, query: str, sort: str, page: int,
              rows: list[dict], max_bytes: int) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with db:
        for item in rows:
            ref, size, license_name = item.get("ref"), item.get("totalBytes"), item.get("licenseName")
            if (not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ref)
                or not isinstance(size, int) or size < 1 or size > max_bytes
                or license_name not in LICENSES):
                continue
            db.execute("""INSERT OR IGNORE INTO datasets
                (ref,title,bytes,license,matched_query,sort_by,first_seen_utc)
                VALUES (?,?,?,?,?,?,?)""",
                (ref, item.get("title"), size, license_name, query, sort, now))
        db.execute("""INSERT OR REPLACE INTO pages
            (query,sort_by,page,result_count,fetched_at_utc) VALUES (?,?,?,?,?)""",
            (query, sort, page, len(rows), now))
    return len(rows)


def export(db: sqlite3.Connection, root: Path, target: int, max_bytes: int) -> int:
    rows = db.execute("""SELECT ref,title,bytes,license,matched_query,sort_by
        FROM datasets ORDER BY rowid LIMIT ?""", (target,)).fetchall()
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": target, "count": len(rows), "metadata_only": True,
        "filters": {"filetype": "csv", "max_listed_bytes": max_bytes,
                    "licenses": sorted(LICENSES)},
        "datasets": [dict(zip(("ref", "title", "bytes", "license", "matched_query", "sort_by"), row))
                     for row in rows],
    }
    destination = root / "task_dataset_index.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=10_000)
    parser.add_argument("--max-dataset-mb", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if min(args.target, args.max_dataset_mb, args.workers) < 1 or args.request_delay < 0:
        parser.error("target, max-dataset-mb and workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    db = connect(args.root / "task_dataset_index.sqlite3")
    with db:
        db.execute("DELETE FROM pages WHERE result_count=0")
    max_bytes = args.max_dataset_mb * 1_000_000
    count = db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    partitions = [(query, "hottest") for query in QUERIES]
    partitions += [("", sort) for sort in ("hottest", "updated", "votes", "active")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for query, sort in partitions:
            for start in range(1, 501, args.workers):
                if count >= args.target:
                    break
                if db.execute("SELECT 1 FROM pages WHERE query=? AND sort_by=? AND result_count<20 LIMIT 1",
                              (query, sort)).fetchone():
                    break
                pages = range(start, min(start + args.workers, 501))
                pending = {}
                for page in pages:
                    if db.execute("SELECT 1 FROM pages WHERE query=? AND sort_by=? AND page=?",
                                  (query, sort, page)).fetchone():
                        continue
                    pending[pool.submit(fetch_page, query, sort, page, max_bytes)] = page
                    if args.request_delay:
                        time.sleep(args.request_delay)
                if not pending:
                    continue
                empty = False
                for future in concurrent.futures.as_completed(pending):
                    page = pending[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        export(db, args.root, args.target, max_bytes)
                        raise RuntimeError(f"catalog page failed: {query!r} {sort} {page}") from exc
                    save_page(db, query, sort, page, rows, max_bytes)
                    empty |= len(rows) < 20
                count = db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
                if start == 1 or start % 50 < args.workers:
                    print(f"query={query!r} sort={sort} page={start} indexed={count}", flush=True)
                if empty:
                    break
            if count >= args.target:
                break
    exported = export(db, args.root, args.target, max_bytes)
    pages = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    print(f"Exported {exported} unique dataset references from {pages} pages", flush=True)
    if exported < args.target:
        raise SystemExit(f"target not reached: {exported}/{args.target}")


if __name__ == "__main__":
    main()
