#!/usr/bin/env python3
"""Download complete raw datasets from the committed Kaggle and DATA.GOV.HK indexes.

The default scope is the approved sources. Use --scope all for the full
metadata catalogs; catalog membership alone never approves training use.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
KAGGLE_INDEX = PROJECT / "data/sources/kaggle/task_dataset_index.json"
HK_INDEX = PROJECT / "data/sources/data_gov_hk/dataset_index.json"
ALLOWLIST = PROJECT / "config/dataset_generation_allowlist.json"
KAGGLE_ROOT = PROJECT / "data/sources/kaggle"
HK_BULK_ROOT = PROJECT / "data/sources/data_gov_hk_bulk"
STATE_FILE = PROJECT / "data/sources/download_state.jsonl"
HK_API = "https://data.gov.hk/sc-data/api/3/action/package_show"
USER_AGENT = "EnvFactory/1.0"
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_index(path: Path, identity: str) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("datasets")
    if not isinstance(rows, list) or document.get("count") != len(rows):
        raise ValueError(f"index count mismatch: {path}")
    ids = [row.get(identity) for row in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"index has invalid or duplicate {identity}: {path}")
    return rows


def selected_items(platform: str, scope: str, kaggle_index: Path = KAGGLE_INDEX,
                   hk_index: Path = HK_INDEX, allowlist: Path = ALLOWLIST) -> list[tuple[str, dict]]:
    """Select indexed sources; approval controls the default download scope."""
    kaggle = load_index(kaggle_index, "ref") if platform in ("both", "kaggle") else []
    hk = load_index(hk_index, "id") if platform in ("both", "data_gov_hk") else []
    if scope == "approved":
        approved = {row["ref"]: row["license"] for row in
                    json.loads(allowlist.read_text(encoding="utf-8"))["datasets"]}
        kaggle = [row for row in kaggle if approved.get(row["ref"]) == row.get("license")]
        hk = [row for row in hk if row.get("approved") is True]
    return [("kaggle", row) for row in kaggle] + [("data_gov_hk", row) for row in hk]


def _safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts or relative.parts[0] == ".":
        raise ValueError(f"unsafe manifest path: {name}")
    path = root / relative
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
        raise ValueError(f"manifest path contains symlink: {name}")
    return path


def verify_manifest(manifest_path: Path, *, source_key: str) -> int:
    """Return complete raw byte count, failing on missing or changed members."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform, source_id = source_key.split(":", 1)
    if document.get("ref" if platform == "kaggle" else "id") != source_id:
        raise ValueError("manifest source identity changed")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest has no raw files")
    raw_root = manifest_path.parent / "raw"
    total = 0
    for entry in files:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0
                or not isinstance(entry.get("sha256"), str)
                or not SHA256.fullmatch(entry["sha256"])):
            raise ValueError("manifest has an invalid file entry")
        path = _safe_member(raw_root, entry["path"])
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or sha256_file(path) != entry["sha256"]):
            raise ValueError(f"raw file fails integrity check: {entry['path']}")
        total += entry["bytes"]
    if total <= 0:
        raise ValueError("manifest contains no raw data")
    return total


@lru_cache(maxsize=4096)
def _public_host(host: str) -> None:
    found = False
    for family, _, _, _, address in socket.getaddrinfo(host, None):
        if family in (socket.AF_INET, socket.AF_INET6):
            found = True
            if not ipaddress.ip_address(address[0]).is_global:
                raise ValueError(f"resource host resolves to a private address: {host}")
    if not found:
        raise ValueError(f"resource host has no public address: {host}")


def _public_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("resource URL is not public HTTP(S)")
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")):
        raise ValueError("resource URL has a local host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("resource URL has a private address")
    _public_host(host)
    return url


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _request_json(url: str) -> dict:
    request = urllib.request.Request(_public_url(url),
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.build_opener(_PublicRedirects()).open(request, timeout=30) as response:
        return json.load(response)


def _download_file(url: str, target: Path, max_bytes: int) -> tuple[int, str]:
    request = urllib.request.Request(_public_url(url), headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.build_opener(_PublicRedirects()).open(request, timeout=60) as response:
        with target.open("wb") as stream:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("resource exceeds remaining byte limit")
                digest.update(block)
                stream.write(block)
    if not size:
        raise ValueError("empty resource")
    return size, digest.hexdigest()


def _resource_name(resource: dict, index: int) -> str:
    resource_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(resource.get("id") or ""))[:48]
    stem = resource_id or f"resource-{index:03d}"
    basename = Path(urllib.parse.urlsplit(str(resource.get("url") or "")).path).name
    suffix = Path(basename).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
        suffix = ".bin"
    return stem + suffix


def download_hk_dataset(row: dict, root: Path, max_bytes: int) -> Path:
    """Download every official CKAN resource to an atomic local directory."""
    dataset_id = row["id"]
    if not re.fullmatch(r"[a-z0-9_-]+", dataset_id):
        raise ValueError("invalid DATA.GOV.HK dataset ID")
    document = _request_json(HK_API + "?" + urllib.parse.urlencode({"id": dataset_id}))
    detail = document.get("result") if document.get("success") is True else None
    if not isinstance(detail, dict) or detail.get("name") != dataset_id:
        raise ValueError("official CKAN API did not confirm dataset identity")
    resources = detail.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("official CKAN dataset has no resources")
    root.mkdir(parents=True, exist_ok=True)
    target = root / dataset_id
    if target.exists():
        raise FileExistsError(f"existing DATA.GOV.HK directory needs inspection: {target}")
    stage = Path(tempfile.mkdtemp(prefix=".hk-bulk-", dir=root))
    try:
        raw = stage / "raw"
        raw.mkdir()
        files = []
        remaining = max_bytes
        for index, resource in enumerate(resources):
            if not isinstance(resource, dict) or not isinstance(resource.get("url"), str):
                raise ValueError("official CKAN dataset contains a resource without URL")
            name = _resource_name(resource, index)
            if (raw / name).exists():
                name = f"{index:03d}-{name}"
            size, digest = _download_file(resource["url"], raw / name, remaining)
            remaining -= size
            files.append({"path": name, "bytes": size, "sha256": digest,
                          "resource_id": resource.get("id"),
                          "resource_url": resource["url"],
                          "format": resource.get("format")})
        manifest = {
            "id": dataset_id, "source": row.get("url"), "title": detail.get("title"),
            "license": detail.get("license_title"), "indexed_resource_count": row.get("resource_count"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        }
        (stage / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_manifest(stage / "source_manifest.json", source_key=f"data_gov_hk:{dataset_id}")
        os.replace(stage, target)
        return target / "source_manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _latest_kaggle_manifest(root: Path, ref: str) -> Path | None:
    manifests = [path for path in (root / ref).glob("v*/source_manifest.json")
                 if re.fullmatch(r"v[1-9][0-9]*", path.parent.name)]
    return max(manifests, key=lambda path: int(path.parent.name[1:]), default=None)


def download_kaggle_dataset(row: dict, root: Path, max_bytes: int) -> Path:
    ref = row["ref"]
    result = subprocess.run([
        sys.executable, str(PROJECT / "scripts/diagnostics/download_kaggle_dataset.py"),
        ref, "--root", str(root), "--max-bytes", str(max_bytes),
    ], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Kaggle download failed for {ref}: {result.stderr[-500:]}")
    manifest = _latest_kaggle_manifest(root, ref)
    if manifest is None:
        raise RuntimeError(f"Kaggle download has no manifest: {ref}")
    return manifest


def _record(state_file: Path, key: str, status: str, *, size: int = 0,
            detail: str = "") -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    row = {"at_utc": datetime.now(timezone.utc).isoformat(), "dataset_key": key,
           "status": status, "bytes": size, "detail": detail[:500]}
    with state_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_status(state_file: Path) -> dict:
    latest: dict[str, dict] = {}
    if state_file.is_file():
        with state_file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("dataset_key"), str):
                    latest[row["dataset_key"]] = row
    counts: dict[str, int] = {}
    for row in latest.values():
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {"processed_unique": len(latest), "latest_status_counts": counts,
            "latest_at_utc": max((row.get("at_utc", "") for row in latest.values()), default=None),
            "state_file": str(state_file)}


def download_selected(
    items: list[tuple[str, dict]], *, kaggle_root: Path = KAGGLE_ROOT,
    hk_root: Path = HK_BULK_ROOT, state_file: Path = STATE_FILE,
    max_dataset_bytes: int = 5_000_000_000,
    max_total_bytes: int = 20_000_000_000,
    reserve_bytes: int = 1_000_000_000,
    retries: int = 2,
) -> dict[str, int]:
    """Resume a bounded batch; one failed source does not discard completed ones."""
    if min(max_dataset_bytes, max_total_bytes) <= 0 or reserve_bytes < 0 or retries < 0:
        raise ValueError("download limits must be positive and disk reserve nonnegative")
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "budget_stopped": 0, "bytes": 0}
    for platform, row in items:
        source_id = row["ref" if platform == "kaggle" else "id"]
        key = f"{platform}:{source_id}"
        manifest = (_latest_kaggle_manifest(kaggle_root, source_id) if platform == "kaggle"
                    else hk_root / source_id / "source_manifest.json")
        if manifest is not None and manifest.is_file():
            try:
                size = verify_manifest(manifest, source_key=key)
                if platform == "kaggle" and json.loads(manifest.read_text(encoding="utf-8")).get("license") != row.get("license"):
                    raise ValueError("Kaggle license differs from indexed metadata")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                stats["failed"] += 1
                _record(state_file, key, "invalid_local", detail=str(exc))
                continue
            stats["skipped"] += 1
            _record(state_file, key, "verified_local", size=size)
            continue
        remaining = max_total_bytes - stats["bytes"]
        if remaining <= 0:
            _record(state_file, key, "budget_stop", detail="total byte budget exhausted")
            stats["budget_stopped"] = 1
            break
        if platform == "kaggle" and isinstance(row.get("bytes"), int) and row["bytes"] > remaining:
            _record(state_file, key, "budget_stop", detail="indexed size exceeds remaining budget")
            stats["budget_stopped"] = 1
            break
        output_root = kaggle_root if platform == "kaggle" else hk_root
        output_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(output_root).free
        budget = min(max_dataset_bytes, remaining, max(0, free - reserve_bytes))
        if budget <= 0:
            _record(state_file, key, "budget_stop", detail="disk reserve reached")
            stats["budget_stopped"] = 1
            break
        for attempt in range(retries + 1):
            try:
                manifest = (download_kaggle_dataset(row, kaggle_root, budget)
                            if platform == "kaggle" else download_hk_dataset(row, hk_root, budget))
                size = verify_manifest(manifest, source_key=key)
                if platform == "kaggle":
                    actual = json.loads(manifest.read_text(encoding="utf-8"))
                    if actual.get("license") != row.get("license"):
                        raise ValueError("Kaggle license differs from indexed metadata")
                stats["downloaded"] += 1
                stats["bytes"] += size
                _record(state_file, key, "downloaded", size=size, detail=str(manifest.parent))
                break
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                message = str(exc)
                transient = any(term in message.casefold() for term in (
                    "429", "500", "502", "503", "504", "timed out", "timeout",
                    "temporarily unavailable", "connection reset"))
                if transient and attempt < retries:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                stats["failed"] += 1
                _record(state_file, key, "failed", detail=message)
                break
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("both", "kaggle", "data_gov_hk"), default="both")
    parser.add_argument("--scope", choices=("approved", "all"), default="approved")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-dataset-gb", type=float, default=5.0)
    parser.add_argument("--max-total-gb", type=float, default=20.0)
    parser.add_argument("--reserve-gb", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--kaggle-index", type=Path, default=KAGGLE_INDEX)
    parser.add_argument("--hk-index", type=Path, default=HK_INDEX)
    parser.add_argument("--kaggle-root", type=Path, default=KAGGLE_ROOT)
    parser.add_argument("--hk-root", type=Path, default=HK_BULK_ROOT)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="summarize the latest result for each indexed dataset")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_status(args.state_file), ensure_ascii=False))
        return 0
    if (args.offset < 0 or args.limit is not None and args.limit <= 0
            or min(args.max_dataset_gb, args.max_total_gb) <= 0 or args.reserve_gb < 0
            or args.retries < 0):
        parser.error("offset, limit and byte budgets must be valid positive values")
    items = selected_items(args.platform, args.scope, args.kaggle_index, args.hk_index)
    items = items[args.offset:args.offset + args.limit if args.limit is not None else None]
    if args.dry_run:
        for platform, row in items:
            print(f"{platform}:{row['ref' if platform == 'kaggle' else 'id']}")
        print(f"Selected {len(items)} indexed datasets; no downloads performed")
        return 0
    stats = download_selected(
        items, kaggle_root=args.kaggle_root, hk_root=args.hk_root,
        state_file=args.state_file,
        max_dataset_bytes=int(args.max_dataset_gb * 1_000_000_000),
        max_total_bytes=int(args.max_total_gb * 1_000_000_000),
        reserve_bytes=int(args.reserve_gb * 1_000_000_000),
        retries=args.retries,
    )
    print(json.dumps({"selected": len(items), **stats, "state_file": str(args.state_file)},
                     ensure_ascii=False))
    return 1 if stats["failed"] or stats["budget_stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
