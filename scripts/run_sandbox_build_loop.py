#!/usr/bin/env python3
"""Compatibility entry point and legacy helpers; execution uses loop_experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from env_factory.sandbox_scoring import evidence_fingerprint


DEFAULT_TASK_IDS = (45, 78, 92, 175)
REQUIRED_MODEL = "gpt-6-luna"


def build_command(
    project: Path, task_id: int, output: Path, max_attempts: int, *, resume: bool = False
) -> list[str]:
    task_path = project / f"output/task/task-{task_id}/task.json"
    if not task_path.is_file():
        legacy = project / f"output/task_artifacts/task-{task_id}/task.json"
        if legacy.is_file():
            task_path = legacy
    command = [
        "bash", str(project / "scripts/develop_sandbox_with_agent.sh"),
        "--input", str(task_path),
        "--output", str(output), "--agent", "codex", "--review-agent", "codex",
        "--model", REQUIRED_MODEL, "--review-model", REQUIRED_MODEL,
        "--runtime", "none", "--max-attempts", str(max_attempts), "--foreground",
        "--skip-auto-score",
    ]
    if resume:
        command.append("--resume")
    return command


def reusable_result(history: dict[str, Any], task_id: int) -> dict[str, Any] | None:
    wanted = f"task-{task_id}"
    for round_report in reversed(history.get("rounds", [])):
        for item in round_report.get("tasks", []):
            score = item.get("score", {})
            # A sandbox may have been rescored after fixing the scorer or the
            # evidence lifecycle. Prefer that authoritative on-disk result to
            # the immutable historical snapshot.
            try:
                score_path = Path(item["output"]) / "sandbox_score.json"
                current_score = json.loads(score_path.read_text(encoding="utf-8"))
                if isinstance(current_score, dict):
                    score = current_score
            except (KeyError, OSError, json.JSONDecodeError, TypeError):
                pass
            if (
                item.get("task_id") == wanted
                and score.get("passed") is True
                and score.get("score", 0) >= 8
                and score.get("model") == REQUIRED_MODEL
                and score.get("review_model") == REQUIRED_MODEL
                and score.get("executed") is True
                and score.get("evidence_fingerprint") == evidence_fingerprint(Path(item["output"]), Path(__file__).resolve().parents[1])
            ):
                return {
                    **item,
                    "score": score,
                    "reused": True,
                    "reused_from": round_report.get("round"),
                }
    return None


def latest_seed(history: dict[str, Any], task_id: int) -> Path | None:
    """Return the newest prior implementation for incremental refinement."""
    wanted = f"task-{task_id}"
    for round_report in reversed(history.get("rounds", [])):
        for item in round_report.get("tasks", []):
            if item.get("task_id") != wanted or not item.get("output"):
                continue
            candidate = Path(item["output"])
            if candidate.is_dir() and (candidate / "task_impl.py").is_file():
                return candidate
    return None


def main() -> int:
    # Preserve helper imports for older callers; CLI experiments use durable state.
    from loop_experiment import main as experiment_main
    return experiment_main()


if __name__ == "__main__":
    raise SystemExit(main())
