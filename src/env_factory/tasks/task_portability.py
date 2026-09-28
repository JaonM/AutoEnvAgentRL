"""Prepare one generated task as an immutable, portable sandbox input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from env_factory.evidence.material_artifacts import digest_json


MANIFEST_DESTINATIONS = {
    "data_manifest": "business_data",
    "user_simulation_manifest": "user_simulation",
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_task(value: Any, index: int) -> tuple[dict[str, Any], bool]:
    selected = value[index] if isinstance(value, list) else value
    if not isinstance(selected, dict):
        raise ValueError("task input must contain a JSON object")
    return selected, isinstance(value, list)


def prepare_sandbox_task(
    input_path: Path, output_root: Path, *, index: int = 0
) -> dict[str, Any]:
    """Copy task-owned artifacts without changing portable manifest roots.

    Legacy tasks with absolute roots remain buildable, but their relocation is
    recorded and they are not identity-preserving production inputs.
    """
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    source_value = json.loads(input_path.read_text(encoding="utf-8"))
    task, source_is_list = _selected_task(source_value, index)
    source_task = json.loads(json.dumps(task, ensure_ascii=False))
    artifacts = task.get("artifacts")
    legacy_relocations: list[str] = []

    if isinstance(artifacts, dict):
        for key, destination_name in MANIFEST_DESTINATIONS.items():
            manifest = artifacts.get(key)
            if not isinstance(manifest, dict) or not manifest.get("root"):
                continue
            declared_root = Path(str(manifest["root"]))
            canonical_root = Path("data") / destination_name
            if declared_root.is_absolute():
                source_root = declared_root
                runtime_root = canonical_root
                legacy_relocations.append(key)
            else:
                if not declared_root.parts or ".." in declared_root.parts:
                    raise ValueError(f"{key}.root is not a safe relative path")
                source_root = input_path.parent / declared_root
                runtime_root = canonical_root
                if declared_root != canonical_root:
                    legacy_relocations.append(key)
            if not source_root.is_dir():
                raise FileNotFoundError(f"{key} root does not exist: {source_root}")
            destination = output_root / runtime_root
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source_root.resolve() != destination.resolve():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source_root, destination)
            manifest["root"] = runtime_root.as_posix()

            for item in task.get("environment", []):
                if (
                    key == "data_manifest"
                    and isinstance(item, dict)
                    and item.get("type") == "business_data_manifest"
                ):
                    item["value"] = manifest

    output_root.mkdir(parents=True, exist_ok=True)
    runtime_path = output_root / "task.json"
    identity_preserved = task == source_task and not source_is_list
    if identity_preserved:
        shutil.copyfile(input_path, runtime_path)
    else:
        runtime_path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    runtime_task = json.loads(runtime_path.read_text(encoding="utf-8"))
    report = {
        "version": "1.0",
        "source_is_task_list": source_is_list,
        "source_task_content_sha256": digest_json(source_task),
        "runtime_task_content_sha256": digest_json(runtime_task),
        "source_file_sha256": _file_sha256(input_path),
        "runtime_task_sha256": _file_sha256(runtime_path),
        "identity_preserved": identity_preserved,
        "legacy_manifest_relocations": sorted(legacy_relocations),
    }
    (output_root / "task_lineage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def valid_task_lineage(
    report: Mapping[str, Any] | Any,
    source_task_path: Path,
    runtime_task_path: Path,
) -> bool:
    """Recompute the identity-preserving task preparation claim."""
    if not isinstance(report, Mapping):
        return False
    try:
        source_bytes = source_task_path.read_bytes()
        runtime_bytes = runtime_task_path.read_bytes()
        source_task = json.loads(source_bytes)
        runtime_task = json.loads(runtime_bytes)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(source_task, dict) or not isinstance(runtime_task, dict):
        return False
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    return (
        report.get("version") == "1.0"
        and report.get("source_is_task_list") is False
        and report.get("identity_preserved") is True
        and report.get("legacy_manifest_relocations") == []
        and report.get("source_task_content_sha256") == digest_json(source_task)
        and report.get("runtime_task_content_sha256") == digest_json(runtime_task)
        and report.get("source_file_sha256") == source_sha
        and report.get("runtime_task_sha256") == runtime_sha
        and source_bytes == runtime_bytes
    )
