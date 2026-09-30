#!/usr/bin/env python3
"""Summarize one frozen development experiment from per-request results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


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
    generation_seconds = []
    generation_recorded_llm_seconds = []
    generation_unattributed_seconds = []
    generation_build_seconds = []
    offline_qualified_build_within_300 = 0
    final_qualified_build_within_300 = 0
    final_qualified_without_build_timing = 0
    sample_latency_seconds = []
    qualified_within_300 = 0
    build_phase_seconds = defaultdict(list)
    score_bands = defaultdict(lambda: Counter())
    live_requested = history.get("config", {}).get("validation") == "live"
    for job in jobs:
        result = job.get("result") or {}
        category = result.get("category") or "unknown"
        bucket = by_category[category]
        task_score = result.get("task_score") or {}
        task_good = task_score.get("eligible") is True and task_score.get("score", 0) >= history.get("config", {}).get("threshold", 8)
        sandbox_score = result.get("sandbox_score") or {}
        sandbox_good = sandbox_score.get("passed") is True
        final_good = result.get("passed") is True
        if task_good:
            band = "9_to_10" if task_score["score"] >= 9 else "threshold_to_9"
            score_bands[band]["task_qualified"] += 1
            score_bands[band]["sandbox_qualified"] += sandbox_good
            score_bands[band]["final_qualified"] += final_good
        if sandbox_good and not task_good or final_good and not sandbox_good:
            raise ValueError(f"non-monotonic eligibility in job {job.get('id')}")
        for counts in (total, bucket):
            counts["requested"] += 1
            counts["generated"] += bool(result.get("task_path"))
            counts["task_qualified"] += task_good
            counts["sandbox_qualified"] += sandbox_good
            counts["final_qualified"] += final_good
            counts["post_score_live_survivors"] += (
                final_good and result.get("live_rollout_verified") is True
            )
        if not final_good:
            failures[result.get("failure_code") or "UNKNOWN"] += 1
        build_latency_observed = False
        manifest_path = result.get("sample_manifest") or job.get("sample_manifest")
        if isinstance(manifest_path, str) and manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                manifest = {}
            seconds = manifest.get("generation_seconds") if isinstance(manifest, dict) else None
            if (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                    and math.isfinite(float(seconds)) and seconds >= 0):
                generation_seconds.append(float(seconds))
                attempts = manifest.get("attempts")
                if isinstance(attempts, list) and attempts and all(
                    isinstance(attempt, dict)
                    and isinstance(attempt.get("llm_trace"), dict)
                    and attempt["llm_trace"].get("version") == "1.1"
                    and isinstance(attempt["llm_trace"].get("request_seconds"), (int, float))
                    and not isinstance(attempt["llm_trace"].get("request_seconds"), bool)
                    and math.isfinite(float(attempt["llm_trace"]["request_seconds"]))
                    and attempt["llm_trace"]["request_seconds"] >= 0
                    for attempt in attempts
                ):
                    llm_seconds = sum(float(attempt["llm_trace"]["request_seconds"])
                                       for attempt in attempts)
                    if 0 <= llm_seconds <= float(seconds) + 0.01:
                        generation_recorded_llm_seconds.append(llm_seconds)
                        generation_unattributed_seconds.append(max(0.0, float(seconds) - llm_seconds))
            reserved_at = manifest.get("reserved_at") if isinstance(manifest, dict) else None
            completed_at = job.get("completed_at")
            build_completed_at = result.get("build_completed_at")
            if isinstance(reserved_at, str) and isinstance(build_completed_at, str):
                try:
                    build_elapsed = (datetime.fromisoformat(build_completed_at) -
                                     datetime.fromisoformat(reserved_at)).total_seconds()
                except (TypeError, ValueError):
                    build_elapsed = -1
                if build_elapsed >= 0:
                    build_latency_observed = True
                    generation_build_seconds.append(build_elapsed)
                    if build_elapsed <= 300:
                        offline_qualified_build_within_300 += bool(sandbox_good)
                        final_qualified_build_within_300 += bool(final_good)
            if isinstance(reserved_at, str) and isinstance(completed_at, str):
                try:
                    elapsed = (datetime.fromisoformat(completed_at) -
                               datetime.fromisoformat(reserved_at)).total_seconds()
                except (TypeError, ValueError):
                    elapsed = -1
                if elapsed >= 0:
                    sample_latency_seconds.append(elapsed)
                    qualified_within_300 += bool(final_good and elapsed <= 300)
        if final_good and not build_latency_observed:
            final_qualified_without_build_timing += 1
        build = result.get("build") or {}
        if isinstance(build.get("seconds"), (int, float)):
            build_seconds.append(float(build["seconds"]))
        output = result.get("output")
        if isinstance(output, str) and output:
            try:
                status = json.loads((Path(output) / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                status = {}
            for phase in status.get("phase_timings", []) if isinstance(status, dict) else []:
                if (isinstance(phase, dict) and isinstance(phase.get("phase"), str)
                        and isinstance(phase.get("seconds"), (int, float))
                        and not isinstance(phase["seconds"], bool)):
                    build_phase_seconds[phase["phase"]].append(float(phase["seconds"]))

    def rates(counts: Counter) -> dict:
        values = dict(counts)
        requested = counts["requested"]
        task_good = counts["task_qualified"]
        values.update({
            "task_yield": task_good / requested,
            "task_yield_ci95_lower": wilson_lower(task_good, requested),
            "conditional_sandbox_yield": counts["sandbox_qualified"] / task_good if task_good else 0,
            "conditional_sandbox_yield_ci95_lower": wilson_lower(counts["sandbox_qualified"], task_good),
            "post_score_survival_rate": (
                counts["post_score_live_survivors"] / counts["sandbox_qualified"]
                if live_requested and counts["sandbox_qualified"] else None
            ),
            "post_score_survival_ci95_lower": (
                wilson_lower(counts["post_score_live_survivors"], counts["sandbox_qualified"])
                if live_requested and counts["sandbox_qualified"] else None
            ),
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
        "by_task_score_band": {
            band: {
                **dict(counts),
                "conditional_sandbox_yield": counts["sandbox_qualified"] / counts["task_qualified"],
                "conditional_final_yield": counts["final_qualified"] / counts["task_qualified"],
            }
            for band, counts in sorted(score_bands.items())
        },
        "build_seconds": {
            "count": len(build_seconds),
            "mean": sum(build_seconds) / len(build_seconds) if build_seconds else None,
            "median": percentile(build_seconds, 0.5),
            "p95": percentile(build_seconds, 0.95),
            "max": max(build_seconds) if build_seconds else None,
        },
        "generation_seconds": {
            "count": len(generation_seconds),
            "median": percentile(generation_seconds, 0.5),
            "p95": percentile(generation_seconds, 0.95),
            "max": max(generation_seconds) if generation_seconds else None,
        },
        "generation_recorded_llm_seconds": {
            "count": len(generation_recorded_llm_seconds),
            "median": percentile(generation_recorded_llm_seconds, 0.5),
            "p95": percentile(generation_recorded_llm_seconds, 0.95),
            "max": max(generation_recorded_llm_seconds) if generation_recorded_llm_seconds else None,
        },
        "generation_unattributed_seconds": {
            "count": len(generation_unattributed_seconds),
            "median": percentile(generation_unattributed_seconds, 0.5),
            "p95": percentile(generation_unattributed_seconds, 0.95),
            "max": max(generation_unattributed_seconds) if generation_unattributed_seconds else None,
        },
        "generation_build_seconds": {
            "count": len(generation_build_seconds),
            "median": percentile(generation_build_seconds, 0.5),
            "p95": percentile(generation_build_seconds, 0.95),
            "max": max(generation_build_seconds) if generation_build_seconds else None,
            "within_300_seconds": sum(value <= 300 for value in generation_build_seconds),
            "offline_qualified_within_300_seconds": offline_qualified_build_within_300,
            "final_qualified_within_300_seconds": final_qualified_build_within_300,
            "final_qualified_without_timing": final_qualified_without_build_timing,
            "final_qualified_within_300_seconds_rate": (
                final_qualified_build_within_300 / expected
                if not final_qualified_without_build_timing
                else None
            ),
            "within_300_seconds_rate": (
                sum(value <= 300 for value in generation_build_seconds) / total["generated"]
                if generation_build_seconds and len(generation_build_seconds) == total["generated"]
                else None
            ),
        },
        "sample_latency_seconds": {
            "count": len(sample_latency_seconds),
            "median": percentile(sample_latency_seconds, 0.5),
            "p95": percentile(sample_latency_seconds, 0.95),
            "max": max(sample_latency_seconds) if sample_latency_seconds else None,
            "within_300_seconds": sum(value <= 300 for value in sample_latency_seconds),
            "qualified_within_300_seconds": qualified_within_300,
            "within_300_seconds_rate": (
                sum(value <= 300 for value in sample_latency_seconds) / expected
                if len(sample_latency_seconds) == expected else None
            ),
        },
        "build_phase_seconds": {
            name: {
                "count": len(values), "total": sum(values),
                "median": percentile(values, 0.5), "p95": percentile(values, 0.95),
            }
            for name, values in sorted(build_phase_seconds.items())
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
