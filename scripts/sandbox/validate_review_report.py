#!/usr/bin/env python3
"""Validate the independent semantic review result used by sandbox builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_factory.sandbox_scoring import validate_semantic_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"review_report.json 无法解析：{exc}") from exc
    try:
        validate_semantic_review(report, args.report.parent)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print("semantic review: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
