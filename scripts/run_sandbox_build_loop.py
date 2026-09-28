#!/usr/bin/env python3
"""Iterative sandbox build loop for the selected high-value tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from env_factory.sandbox_scoring import evidence_fingerprint


DEFAULT_TASK_IDS = (45, 78, 92, 175)
REQUIRED_MODEL = "gpt-6-luna"


def next_round(root: Path) -> int:
    values = []
    for path in root.glob("round-*"):
        try:
            values.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0) + 1


def build_command(
    project: Path, task_id: int, output: Path, max_attempts: int, *, resume: bool = False
) -> list[str]:
    command = [
        "bash", str(project / "scripts/develop_sandbox_with_agent.sh"),
        "--input", str(project / f"output/task_artifacts/task-{task_id}/task.json"),
        "--output", str(output), "--agent", "codex", "--review-agent", "codex",
        "--model", REQUIRED_MODEL, "--review-model", REQUIRED_MODEL,
        "--runtime", "none", "--max-attempts", str(max_attempts), "--foreground",
        "--skip-auto-score",
    ]
    if resume:
        command.append("--resume")
    return command


def run_one(
    project: Path,
    round_root: Path,
    task_id: int,
    max_attempts: int,
    seed: Path | None = None,
) -> dict[str, Any]:
    output = round_root / f"task-{task_id}"
    if seed is not None and seed.is_dir():
        shutil.copytree(seed, output, dirs_exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "loop_build.log"
    command = build_command(project, task_id, output, max_attempts, resume=seed is not None)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT, text=True)
    score_path = output / "sandbox_score.json"
    score_command = [
        sys.executable, str(project / "scripts/sandbox/score_sandbox.py"), str(output),
        "--project", str(project), "--threshold", "8", "--output", str(score_path), "--execute",
    ]
    scored = subprocess.run(score_command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        score = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        score = {"score": 0.0, "passed": False, "error": scored.stdout[-4000:]}
    return {
        "task_id": f"task-{task_id}", "output": str(output),
        "build_exit_code": completed.returncode, "score_exit_code": scored.returncode,
        "score": score,
    }


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
