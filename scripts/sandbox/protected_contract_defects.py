#!/usr/bin/env python3
"""Identify sandbox review failures that require regenerating protected task contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PROTECTED = {"task.json", "BUILD_CONTRACT.json"}
PLATFORM_OWNED = {
    "app.py", "acceptance_runner.py", "sandbox_runtime.py", "runtime_llm.py",
    "Dockerfile", "docker_build.sh", "docker_run.sh", "tools.json",
}
CONTRACT_CHANGES = re.compile(
    r"metric|evaluator|rubric|reward|success scenario|success fixture|"
    r"acceptance contract|结果指标|奖励指标|成功轨迹|验收场景|评分规则",
    re.I,
)


def requires_task_regeneration(defects: Any) -> bool:
    if not isinstance(defects, list) or not defects:
        return False
    severe = [item for item in defects if isinstance(item, dict)
              and item.get("severity") in {"high", "critical"}]
    if not severe or any(Path(str(item.get("file") or "")).name not in PROTECTED
                         for item in severe):
        return False
    return any(CONTRACT_CHANGES.search(str(item.get("fix_required", "")))
               for item in severe)


def requires_platform_repair(defects: Any) -> bool:
    """Do not spend task-agent retries on EnvFactory-owned delivery assets."""
    if not isinstance(defects, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("severity") in {"high", "critical"}
        and Path(str(item.get("file") or "")).name in PLATFORM_OWNED
        for item in defects
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defects", required=True, type=Path)
    parser.add_argument("--owner", choices=("task_generation", "platform"),
                        default="task_generation")
    args = parser.parse_args()
    defects = json.loads(args.defects.read_text(encoding="utf-8"))
    check = requires_platform_repair if args.owner == "platform" else requires_task_regeneration
    return 0 if check(defects) else 1


if __name__ == "__main__":
    raise SystemExit(main())
