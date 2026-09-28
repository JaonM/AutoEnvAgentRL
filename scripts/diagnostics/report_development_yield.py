#!/usr/bin/env python3
"""Summarize one frozen development experiment from per-request results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denominator)


def summarize(history: dict) -> dict:
    rounds = history.get("rounds", [])
    jobs = [job for round_ in rounds for job in round_.get("jobs", [])]
    expected = sum(round_.get("summary", {}).get("requested", 0) for round_ in rounds)
    if not rounds or not jobs or len(jobs) != expected:
        raise ValueError("history must contain a result for every requested job")

    by_category = defaultdict(lambda: Counter())
    failures = Counter()
    total = Counter()
    build_seconds = []
    for job in jobs:
        result = job.get("result") or {}
        category = result.get("category") or "unknown"
        bucket = by_category[category]
        task_score = result.get("task_score") or {}
        task_good = task_score.get("eligible") is True and task_score.get("score", 0) >= history.get("config", {}).get("threshold", 8)
        sandbox_score = result.get("sandbox_score") or {}
        sandbox_good = sandbox_score.get("passed") is True
        final_good = result.get("passed") is True
        if sandbox_good and not task_good or final_good and not sandbox_good:
            raise ValueError(f"non-monotonic eligibility in job {job.get('id')}")
        for counts in (total, bucket):
            counts["requested"] += 1
            counts["generated"] += bool(result.get("task_path"))
            counts["task_qualified"] += task_good
            counts["sandbox_qualified"] += sandbox_good
            counts["final_qualified"] += final_good
        if not final_good:
            failures[result.get("failure_code") or "UNKNOWN"] += 1
        build = result.get("build") or {}
        if isinstance(build.get("seconds"), (int, float)):
            build_seconds.append(float(build["seconds"]))

    def rates(counts: Counter) -> dict:
        values = dict(counts)
        requested = counts["requested"]
        task_good = counts["task_qualified"]
        values.update({
            "task_yield": task_good / requested,
            "task_yield_ci95_lower": wilson_lower(task_good, requested),
            "conditional_sandbox_yield": counts["sandbox_qualified"] / task_good if task_good else 0,
            "conditional_sandbox_yield_ci95_lower": wilson_lower(counts["sandbox_qualified"], task_good),
            "end_to_end_yield": counts["final_qualified"] / requested,
            "end_to_end_yield_ci95_lower": wilson_lower(counts["final_qualified"], requested),
        })
        return values

    return {
        "scope": "development_diagnostic_only",
        "stop_reason": history.get("stop_reason"),
        "rounds": len(rounds),
        "valid_quality_rounds": sum(
            round_.get("summary", {}).get("valid_quality_round") is True
            for round_ in rounds
        ),
        "overall": rates(total),
        "by_category": {name: rates(counts) for name, counts in sorted(by_category.items())},
        "failure_codes": dict(sorted(failures.items())),
        "build_seconds": {
            "count": len(build_seconds),
            "mean": sum(build_seconds) / len(build_seconds) if build_seconds else None,
            "max": max(build_seconds) if build_seconds else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(json.loads(args.history.read_text(encoding="utf-8")))
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
