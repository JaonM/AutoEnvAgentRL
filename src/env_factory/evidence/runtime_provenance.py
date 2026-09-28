"""Attestation checks for live rollouts executed by the validated container."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from env_factory.evidence.material_artifacts import docker_build_context_digest


FIELDS = {
    "version", "mode", "container_image_id", "transport", "read_only_root",
    "cap_drop", "no_new_privileges", "non_root_user",
}


def valid_container_rollout_execution(
    rollout: Mapping[str, Any] | Any, sandbox_root: Path, *,
    require_build_context: bool = True,
) -> bool:
    if not isinstance(rollout, Mapping):
        return False
    execution = rollout.get("runtime_execution")
    if not isinstance(execution, Mapping) or set(execution) != FIELDS:
        return False
    if not (
        execution.get("version") == "1.0"
        and execution.get("mode") == "docker_http"
        and execution.get("transport") == "loopback_http"
        and isinstance(execution.get("container_image_id"), str)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}", execution["container_image_id"]
        ) is not None
        and execution.get("read_only_root") is True
        and execution.get("cap_drop") == "ALL"
        and execution.get("no_new_privileges") is True
        and execution.get("non_root_user") is True
    ):
        return False
    try:
        metadata = json.loads(
            (sandbox_root / "docker_image_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    context_digest = None
    if require_build_context:
        try:
            context_digest = docker_build_context_digest(sandbox_root)
        except (OSError, ValueError):
            return False
    smoke = metadata.get("smoke_test", {}) if isinstance(metadata, Mapping) else {}
    return (
        isinstance(metadata, Mapping)
        and metadata.get("version") in ({"5.0"} if require_build_context else {"4.0", "5.0"})
        and (
            not require_build_context
            or metadata.get("build_context_sha256") == context_digest
        )
        and metadata.get("image_id") == execution["container_image_id"]
        and isinstance(metadata.get("runtime_user"), str)
        and metadata["runtime_user"] not in {"", "root", "0"}
        and isinstance(smoke, Mapping)
        and smoke.get("passed") is True
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
    )
