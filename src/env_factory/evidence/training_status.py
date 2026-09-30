"""Publish the final training eligibility without conflating it with build success."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def training_ready(result: Mapping[str, Any]) -> bool:
    """Only the complete live qualification path can certify a delivery."""
    return all(result.get(key) is True for key in (
        "passed", "live_rollout_verified", "live_reward_calibration_verified",
        "data_governance_verified", "trajectory_privacy_verified",
    )) and result.get("sandbox_score", {}).get("passed") is True


def write_training_status(output: Path, result: dict[str, Any] | None = None) -> None:
    """Reset on a new attempt; atomically publish the final result when available.

    Do not create a successful build status for generation failures or missing
    build artifacts. Old consumers can keep using the existing build fields.
    """
    ready = training_ready(result) if result is not None else False
    if result is not None:
        result["training_ready"] = ready
    path = Path(output) / "status.json"
    if not path.exists():
        if result is not None:
            result["training_ready"] = False
        return
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if ready:
            raise ValueError("Cannot publish training readiness without a valid build status")
        return
    if not isinstance(status, dict):
        raise ValueError("Build status must be an object")
    ready = ready and status.get("success") is True and status.get("status") == "success" and status.get("exit_code") == 0
    if result is not None:
        result["training_ready"] = ready
    status["training_ready"] = ready
    descriptor, temporary = tempfile.mkstemp(prefix=".training-status-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(status, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
