"""Reproducible, non-secret identity for the EnvFactory execution environment."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import tomllib
from typing import Any, Mapping


TOP_LEVEL_FIELDS = {
    "version", "python", "platform", "packages", "pyproject_sha256",
    "uv_lock_sha256", "complete", "identity_sha256",
}


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise ValueError(f"invalid dependency declaration: {requirement!r}")
    return match.group(1)


def _declared_packages(project: Path) -> list[str]:
    document = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    project_table = document.get("project", {})
    requirements = list(project_table.get("dependencies", []))
    for values in document.get("dependency-groups", {}).values():
        if isinstance(values, list):
            requirements.extend(value for value in values if isinstance(value, str))
    names = [_requirement_name(value) for value in requirements]
    project_name = project_table.get("name")
    if isinstance(project_name, str) and project_name:
        names.append(project_name)
    return sorted(set(names), key=str.casefold)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def collect_execution_provenance(project: Path) -> dict[str, Any]:
    """Collect only stable runtime facts; never include paths, hosts or env vars."""
    project = project.resolve()
    packages = {}
    for name in _declared_packages(project):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "version": "1.0",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "pyproject_sha256": file_sha256(project / "pyproject.toml"),
        "uv_lock_sha256": file_sha256(project / "uv.lock"),
    }
    result["complete"] = (
        all(isinstance(value, str) and value for value in packages.values())
        and all(
            isinstance(result[name], str) and len(result[name]) == 64
            for name in ("pyproject_sha256", "uv_lock_sha256")
        )
    )
    result["identity_sha256"] = _digest(result)
    return result


def valid_execution_provenance(value: Mapping[str, Any] | Any) -> bool:
    """Validate a portable snapshot without requiring the consumer to match it."""
    if not isinstance(value, Mapping) or set(value) != TOP_LEVEL_FIELDS:
        return False
    python = value.get("python")
    target = value.get("platform")
    packages = value.get("packages")
    if not (
        value.get("version") == "1.0"
        and value.get("complete") is True
        and isinstance(python, Mapping)
        and set(python) == {"implementation", "version", "cache_tag"}
        and all(isinstance(item, str) and item for item in python.values())
        and isinstance(target, Mapping)
        and set(target) == {"system", "release", "machine"}
        and all(isinstance(item, str) and item for item in target.values())
        and isinstance(packages, Mapping)
        and bool(packages)
        and all(
            isinstance(name, str) and bool(name)
            and isinstance(version, str) and bool(version)
            for name, version in packages.items()
        )
        and all(
            isinstance(value.get(name), str)
            and re.fullmatch(r"[0-9a-f]{64}", value[name]) is not None
            for name in ("pyproject_sha256", "uv_lock_sha256", "identity_sha256")
        )
    ):
        return False
    unsigned = {name: value[name] for name in TOP_LEVEL_FIELDS - {"identity_sha256"}}
    return value["identity_sha256"] == _digest(unsigned)


def verify_execution_provenance(
    project: Path, recorded: Mapping[str, Any] | Any
) -> dict[str, Any]:
    current = collect_execution_provenance(project)
    if not isinstance(recorded, Mapping):
        return {
            "verified": False,
            "differences": ["missing_recorded_provenance"],
            "current": current,
        }
    fields = (
        "version", "python", "platform", "packages", "pyproject_sha256",
        "uv_lock_sha256", "complete", "identity_sha256",
    )
    differences = [name for name in fields if recorded.get(name) != current.get(name)]
    if not valid_execution_provenance(recorded):
        differences.append("invalid_recorded_provenance")
    return {
        "verified": not differences,
        "differences": sorted(set(differences)),
        "recorded_identity_sha256": recorded.get("identity_sha256"),
        "current_identity_sha256": current.get("identity_sha256"),
        "current": current,
    }
