#!/usr/bin/env python3
"""Route known outer-validation failures before task-agent repair."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REWARD_ENDPOINT_TIMEOUT = re.compile(
    r"runtime endpoint unavailable\s+(?:GET|POST)\s+/v1/reward\s*:\s*.*timed out",
    re.IGNORECASE,
)


def zero_extension_reward_timeout(plan: object, error: str) -> bool:
    """A generated-only sandbox has no task code for the agent to repair."""
    return (
        isinstance(plan, dict)
        and plan.get("nodes") == []
        and REWARD_ENDPOINT_TIMEOUT.search(error) is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--error-log", required=True, type=Path)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        error = args.error_log.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return 1
    return 0 if zero_extension_reward_timeout(plan, error) else 1


if __name__ == "__main__":
    raise SystemExit(main())
