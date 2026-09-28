#!/usr/bin/env python3
"""Resolve a verbose Docker manifest to one platform-specific content address."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


def normalize_architecture(value: str) -> str:
    return {"aarch64": "arm64", "x86_64": "amd64"}.get(value, value)


def resolve_reference(
    document: Any, candidate: str, wanted_os: str, wanted_architecture: str
) -> str | None:
    wanted_architecture = normalize_architecture(wanted_architecture)
    if isinstance(document, Mapping) and isinstance(document.get("manifests"), list):
        items = document["manifests"]
    else:
        items = document if isinstance(document, list) else [document]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        descriptor = item.get("Descriptor", {})
        if not isinstance(descriptor, Mapping):
            descriptor = {}
        platform = item.get("Platform") or item.get("platform") or descriptor.get("platform")
        if not isinstance(platform, Mapping):
            continue
        digest = descriptor.get("digest") or item.get("digest")
        if not (
            isinstance(digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
            and platform.get("os") == wanted_os
            and normalize_architecture(str(platform.get("architecture", "")))
                == wanted_architecture
        ):
            continue
        repository = candidate.split("@", 1)[0]
        slash = repository.rfind("/")
        colon = repository.rfind(":")
        if colon > slash:
            repository = repository[:colon]
        return f"{repository}@{digest}"
    return None


def resolve_cached_reference(
    document: Any, candidate: str, wanted_os: str, wanted_architecture: str
) -> str | None:
    """Pin a locally cached image only when its platform and repository match."""
    items = document if isinstance(document, list) else [document]
    requested_digest = candidate.split("@", 1)[1] if "@" in candidate else None
    repository = candidate.split("@", 1)[0]
    if repository.rfind(":") > repository.rfind("/"):
        repository = repository[:repository.rfind(":")]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if item.get("Os") != wanted_os or normalize_architecture(str(item.get("Architecture", ""))) != normalize_architecture(wanted_architecture):
            continue
        for reference in item.get("RepoDigests", []):
            if (isinstance(reference, str)
                    and reference.startswith(repository + "@")
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", reference.split("@", 1)[1])
                    and (requested_digest is None or reference == repository + "@" + requested_digest)):
                return reference
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("candidate")
    parser.add_argument("os")
    parser.add_argument("architecture")
    parser.add_argument("--local-inspect", action="store_true")
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolver = resolve_cached_reference if args.local_inspect else resolve_reference
    resolved = resolver(value, args.candidate, args.os, args.architecture)
    if resolved is None:
        return 1
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
