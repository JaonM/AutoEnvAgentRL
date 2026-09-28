#!/usr/bin/env python3
"""Download a public Kaggle dataset into a versioned local source directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PROJECT = Path(__file__).resolve().parents[2]
API = "https://www.kaggle.com/api/v1/datasets"
MAX_BYTES = 5_000_000_000


def dataset_ref(value: str) -> str:
    if value.startswith("https://www.kaggle.com/"):
        parts = urlparse(value).path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "datasets":
            raise ValueError("expected a Kaggle dataset URL with owner and slug")
        value = "/".join(parts[1:3])
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError("dataset must be owner/slug or a Kaggle dataset URL")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, max_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "EnvFactory/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as stream:
        size = 0
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"download exceeds {max_bytes} bytes")
            stream.write(chunk)


def extract_zip(archive: Path, target: Path, max_bytes: int) -> list[dict]:
    files = []
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if sum(item.file_size for item in entries) > max_bytes:
            raise ValueError(f"uncompressed dataset exceeds {max_bytes} bytes")
        for item in entries:
            name = Path(item.filename)
            if item.is_dir():
                continue
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError(f"unsafe archive path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"archive contains symlink: {item.filename}")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            files.append({"path": name.as_posix(), "bytes": destination.stat().st_size,
                          "sha256": sha256(destination)})
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Kaggle owner/slug or dataset URL")
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "sources" / "kaggle")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    ref = dataset_ref(args.dataset)
    with urllib.request.urlopen(f"{API}/view/{ref}", timeout=30) as response:
        metadata = json.load(response)
    version = metadata.get("currentVersionNumber")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Kaggle metadata has no valid dataset version")
    target = args.root / ref / f"v{version}"
    if target.exists():
        raise FileExistsError(f"already downloaded: {target}")
    total_bytes = metadata.get("totalBytes")
    if isinstance(total_bytes, int) and total_bytes > args.max_bytes:
        raise ValueError(f"dataset exceeds {args.max_bytes} bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-", dir=target.parent) as temporary:
        stage = Path(temporary)
        archive = stage / "dataset.zip"
        download(f"{API}/download/{ref}", archive, args.max_bytes)
        archive_hash = sha256(archive)
        files = extract_zip(archive, stage / "raw", args.max_bytes)
        archive.unlink()
        manifest = {
            "source": f"https://www.kaggle.com/datasets/{ref}",
            "ref": ref,
            "version": version,
            "title": metadata.get("title"),
            "license": metadata.get("licenseName"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": archive_hash,
            "files": files,
        }
        (stage / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(stage, target)
    print(json.dumps({"path": str(target), "files": files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
