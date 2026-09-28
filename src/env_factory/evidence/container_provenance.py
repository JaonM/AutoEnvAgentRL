"""Independent checks for reproducible, portable sandbox container evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from env_factory.evidence.material_artifacts import docker_build_context_digest, docker_context_errors


EXACT_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+(?:\s*;.*)?$"
)
PINNED_IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
CONTENT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIREMENT_PARTS = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^\s;]+)(?:\s*;(.*))?$"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_requirements(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    values = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return bool(values) and all(EXACT_REQUIREMENT.fullmatch(line) for line in values)


def _package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def verify_python_packages(path: Path, requirements: Path) -> dict[str, Any]:
    failures = []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"verified": False, "packages": 0, "failed_gates": ["inventory_file"]}
    packages = document.get("packages") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or set(document) != {"version", "packages"}
        or document.get("version") != "1.0"
        or not isinstance(packages, list) or not packages
    ):
        failures.append("inventory_schema")
        packages = []
    normalized = {}
    rendered_order = []
    for package in packages:
        if not (
            isinstance(package, Mapping)
            and set(package) == {"name", "version"}
            and isinstance(package.get("name"), str) and package["name"].strip()
            and isinstance(package.get("version"), str) and package["version"].strip()
        ):
            failures.append("inventory_schema")
            continue
        name = _package_name(package["name"])
        if name in normalized:
            failures.append("inventory_duplicates")
        normalized[name] = package["version"]
        rendered_order.append(package["name"])
    if rendered_order != sorted(rendered_order, key=str.casefold):
        failures.append("inventory_order")
    try:
        lines = requirements.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        match = REQUIREMENT_PARTS.fullmatch(value)
        if match is None:
            continue
        name, wanted, marker = match.groups()
        # Environment-marked pins may deliberately be absent on this platform.
        if marker is None and normalized.get(_package_name(name)) != wanted:
            failures.append("direct_dependency_version")
    return {
        "verified": not failures,
        "packages": len(normalized),
        "failed_gates": sorted(set(failures)),
    }


def verify_container_provenance(
    root: Path, *, expected_tag: str | None = None
) -> dict[str, Any]:
    failures = []
    failures.extend(docker_context_errors(root))
    metadata_path = root / "docker_image_metadata.json"
    dockerfile = root / "Dockerfile"
    requirements = root / "requirements-dev.txt"
    package_inventory = root / "python_packages.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
        failures.append("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("version") != "5.0":
        failures.append("metadata_schema")
        metadata = {}
    try:
        context_digest = docker_build_context_digest(root)
    except (OSError, ValueError):
        context_digest = ""
    if metadata.get("build_context_sha256") != context_digest:
        failures.append("build_context_digest")
    if PINNED_IMAGE.fullmatch(str(metadata.get("base_image", ""))) is None:
        failures.append("base_image_digest")
    if expected_tag is not None and metadata.get("tag") != expected_tag:
        failures.append("image_tag")
    if CONTENT_DIGEST.fullmatch(str(metadata.get("image_id", ""))) is None:
        failures.append("image_id")
    try:
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        dockerfile_digest = file_sha256(dockerfile)
    except OSError:
        dockerfile_text = ""
        dockerfile_digest = ""
        failures.append("dockerfile")
    from_images = re.findall(r"(?im)^\s*FROM\s+([^\s]+)", dockerfile_text)
    if not from_images or any(PINNED_IMAGE.fullmatch(image) is None for image in from_images):
        failures.append("dockerfile_base_image")
    elif any(image != metadata.get("base_image") for image in from_images):
        failures.append("base_image_mismatch")
    users = re.findall(r"(?im)^\s*USER\s+([^\s]+)", dockerfile_text)
    runtime_user = str(metadata.get("runtime_user", ""))
    if (
        not users
        or users[-1] in {"root", "0"}
        or runtime_user in {"", "root", "0"}
        or users[-1] != runtime_user
    ):
        failures.append("non_root_user")
    if metadata.get("dockerfile_sha256") != dockerfile_digest:
        failures.append("dockerfile_digest")
    if not exact_requirements(requirements):
        failures.append("dependency_pins")
    try:
        requirements_digest = file_sha256(requirements)
    except OSError:
        requirements_digest = ""
    if metadata.get("requirements_sha256") != requirements_digest:
        failures.append("requirements_digest")
    package_report = verify_python_packages(package_inventory, requirements)
    if package_report["verified"] is not True:
        failures.extend(package_report["failed_gates"])
    try:
        package_digest = file_sha256(package_inventory)
    except OSError:
        package_digest = ""
    if metadata.get("python_packages_sha256") != package_digest:
        failures.append("python_packages_digest")
    smoke = metadata.get("smoke_test", {})
    if not (
        isinstance(smoke, Mapping)
        and smoke.get("passed") is True
        and smoke.get("network") == "none"
        and smoke.get("read_only_root") is True
        and smoke.get("cap_drop") == "ALL"
        and smoke.get("no_new_privileges") is True
        and smoke.get("non_root_user") is True
        and smoke.get("service_health") is True
        and smoke.get("runtime_tmpfs") == {
            "path": "/app/.runtime",
            "uid": 10001,
            "gid": 10001,
            "mode": "0700",
        }
    ):
        failures.append("container_smoke_test")
    platform = metadata.get("platform", {})
    if not (
        isinstance(platform, Mapping)
        and isinstance(platform.get("os"), str) and platform["os"]
        and isinstance(platform.get("architecture"), str) and platform["architecture"]
    ):
        failures.append("platform")
    return {
        "verified": not failures,
        "failed_gates": sorted(set(failures)),
        "base_image": metadata.get("base_image"),
        "tag": metadata.get("tag"),
        "image_id": metadata.get("image_id"),
        "platform": metadata.get("platform"),
        "python_packages": package_report["packages"],
    }
