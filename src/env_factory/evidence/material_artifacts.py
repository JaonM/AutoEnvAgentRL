"""Stable identities for portable sandbox inputs and certification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import fnmatch
import re
from typing import Any


MATERIAL_MANIFEST_VERSION = "6.0"
SUPPORTED_MATERIAL_MANIFEST_VERSIONS = {
    "1.0", "2.0", "3.0", "4.0", "5.0", "6.0",
}


EVIDENCE_NAMES = {
    "acceptance_result.json", "agentic_training_value.json",
    "agentic_training_value_live.json", "buildability.json",
    "data_governance.json",
    "offline_sandbox_score.json", "review_report.json", "score_summary.json",
    "training_readiness.json", "trajectory_privacy.json",
}
SKIPPED_SUFFIXES = {".pyc", ".pyo", ".sqlite", ".sqlite3", ".db", ".log"}

DOCKERIGNORE_SOURCE = """.git
.git/**
.env
.env.*
__pycache__
**/__pycache__
.pytest_cache
**/.pytest_cache
.runtime
**/.runtime
.outer_conformance
.outer_conformance/**
*.pyc
*.pyo
*.sqlite
*.sqlite3
*.db
*.log
*.stdout
*.stderr
live_rollout.json
acceptance_result.json
agentic_training_value.json
agentic_training_value_live.json
buildability.json
data_governance.json
offline_sandbox_score.json
review_report.json
score_summary.json
training_readiness.json
trajectory_privacy.json
docker_image_metadata.json
python_packages.json
status.json
runtime_trace.jsonl
mutation_report.json
runtime_state.json
last_delivery_error.txt
defects.json
failure.json
mutation_cases.json
defect_*_review.json
"""

IGNORED_CONTEXT_NAMES = EVIDENCE_NAMES | {
    "live_rollout.json", "status.json", "runtime_trace.jsonl",
    "mutation_report.json", "docker_image_metadata.json", "python_packages.json",
    "runtime_state.json", "last_delivery_error.txt", "defects.json",
    "failure.json", "mutation_cases.json",
}
IGNORED_CONTEXT_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".runtime", ".outer_conformance",
}
PORTABLE_POST_BUILD_NAMES = {"docker_image_metadata.json", "python_packages.json"}
IGNORED_CONTEXT_GLOBS = {"*.stdout", "*.stderr", "defect_*_review.json"}
RISKY_CONTEXT_FILE = re.compile(
    r"^(?:\.env(?:\..*)?|credentials?|secrets?|api[_-]?keys?)(?:\..*)?$",
    re.IGNORECASE,
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _digest_paths(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def docker_context_errors(root: Path) -> list[str]:
    """Validate the canonical, secret-resistant sandbox Docker context."""
    errors = []
    ignore = root / ".dockerignore"
    try:
        if ignore.read_text(encoding="utf-8") != DOCKERIGNORE_SOURCE:
            errors.append("dockerignore_contract")
    except OSError:
        errors.append("dockerignore_contract")
    try:
        paths = list(root.rglob("*"))
    except OSError:
        return sorted(set(errors + ["context_unreadable"]))
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"context_symlink:{relative}")
        if path.is_file() and RISKY_CONTEXT_FILE.fullmatch(path.name):
            errors.append(f"sensitive_context_file:{relative}")
        hidden_parts = [
            part for part in relative.parts
            if part.startswith(".") and part not in {
                ".dockerignore", ".git", ".pytest_cache", ".runtime",
                ".outer_conformance",
            }
        ]
        if hidden_parts:
            errors.append(f"unexpected_hidden_context_path:{relative}")
    return sorted(set(errors))


def _ignored_context_path(relative: Path) -> bool:
    return (
        any(part in IGNORED_CONTEXT_PARTS for part in relative.parts)
        or relative.name in IGNORED_CONTEXT_NAMES
        or relative.suffix in SKIPPED_SUFFIXES
        or any(fnmatch.fnmatch(relative.name, pattern) for pattern in IGNORED_CONTEXT_GLOBS)
        or RISKY_CONTEXT_FILE.fullmatch(relative.name) is not None
    )


def docker_build_context_digests(root: Path) -> dict[str, str]:
    """Hash every file allowed into the canonical Docker build context."""
    paths = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"portable sandbox contains symlink: {relative}")
        if not path.is_file():
            continue
        if _ignored_context_path(relative):
            continue
        paths.append(path)
    return _digest_paths(root, paths)


def docker_build_context_digest(root: Path) -> str:
    return digest_json(docker_build_context_digests(root))


def portable_artifact_digests(root: Path) -> dict[str, str]:
    """Hash the complete rebuild context plus post-build provenance artifacts."""
    values = docker_build_context_digests(root)
    for name in sorted(PORTABLE_POST_BUILD_NAMES):
        path = root / name
        if path.is_file():
            values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(values.items()))


def evidence_artifact_digests(root: Path) -> dict[str, str]:
    """Hash reports whose claims are consumed by production certification."""
    return _digest_paths(root, [root / name for name in EVIDENCE_NAMES if (root / name).is_file()])


def portable_artifact_digest(root: Path) -> str:
    return digest_json(portable_artifact_digests(root))
