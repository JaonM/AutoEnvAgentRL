"""Balanced source allocation and verified DATA.GOV.HK raw files."""

from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from env_factory.generation.dataset_formats import SUPPORTED_SOURCE_EXTENSIONS, source_extension
from env_factory.generation.task_generator import TaskGenerationError


PROJECT = Path(__file__).resolve().parents[3]
HK_INDEX = PROJECT / "data/sources/data_gov_hk/dataset_index.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_platforms(count: int, *, rng: random.Random) -> list[str]:
    """Allocate exact 1:1 source quotas before concurrent task generation."""
    if count <= 0 or count % 2:
        raise ValueError("balanced Kaggle/DATA.GOV.HK generation requires a positive even count")
    platforms = ["kaggle"] * (count // 2) + ["data_gov_hk"] * (count // 2)
    rng.shuffle(platforms)
    return platforms


@lru_cache(maxsize=8)
def _cached_hk_catalog(index_path: Path, mtime_ns: int, size: int) -> dict[str, dict[str, Any]]:
    del mtime_ns, size
    try:
        document = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskGenerationError("DATA.GOV.HK catalog is unavailable") from exc
    rows = document.get("datasets")
    if not isinstance(rows, list):
        raise TaskGenerationError("DATA.GOV.HK catalog has no datasets")
    return {str(row["id"]): row for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)}


def hk_catalog(index_path: Path = HK_INDEX) -> dict[str, dict[str, Any]]:
    try:
        stat = index_path.stat()
    except OSError as exc:
        raise TaskGenerationError("DATA.GOV.HK catalog is unavailable") from exc
    return _cached_hk_catalog(index_path.resolve(), stat.st_mtime_ns, stat.st_size)


def eligible_hk_ids(*, index_path: Path = HK_INDEX) -> list[str]:
    entries = hk_catalog(index_path).values()
    return sorted(entry["id"] for entry in entries
                  if entry.get("approved") is True
                  and entry.get("resource_format") in SUPPORTED_SOURCE_EXTENSIONS
                  and isinstance(entry.get("resource_url"), str))


def verified_hk_source(
    dataset_id: str, *, root: Path = HK_INDEX.parent, index_path: Path = HK_INDEX,
) -> tuple[Path, str, str, str | None] | None:
    """Read only files whose recorded digest and official catalog entry match."""
    indexed = hk_catalog(index_path).get(dataset_id)
    if indexed is None or indexed.get("approved") is not True:
        return None
    manifest_path = root / dataset_id / "source_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (manifest.get("id") != dataset_id or manifest.get("source") != indexed.get("url")
            or manifest.get("resource_url") != indexed.get("resource_url")):
        return None
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        name = entry["path"]
        if Path(name).name != name or source_extension(name) not in SUPPORTED_SOURCE_EXTENSIONS:
            continue
        path = manifest_path.parent / "raw" / name
        if (path.is_file() and isinstance(entry.get("sha256"), str)
                and SHA256.fullmatch(entry["sha256"])
                and sha256_file(path) == entry["sha256"]):
            return path, str(indexed.get("title") or path.stem), str(indexed["url"]), indexed.get("license")
    return None
