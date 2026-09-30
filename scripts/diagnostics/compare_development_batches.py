#!/usr/bin/env python3
"""Compare two completed, identically sampled development batches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def batch(root: Path) -> dict[str, Any]:
    summary = load(root / "task_yield_summary.json")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != summary.get("requested"):
        raise ValueError(f"{root} is not a completed generation batch")
    by_id = {item.get("task_id"): item for item in rows if isinstance(item, dict)}
    if len(by_id) != len(rows) or any(not isinstance(key, str) for key in by_id):
        raise ValueError(f"{root} has duplicate or missing task IDs")
    task_root = root / "task"
    if not task_root.is_dir() and (root / "task_artifacts").is_dir():
        task_root = root / "task_artifacts"
    manifests = {
        task_id: load(task_root / task_id / "sample_manifest.json")
        for task_id in by_id
    }
    selection = load(root / "build_probe_selection.json")
    probe_path = root / "build_probe_summary.json"
    probes = load(probe_path) if probe_path.is_file() else None
    return {
        "root": str(root), "summary": summary, "rows": by_id,
        "manifests": manifests, "selection": selection, "probes": probes,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ids = set(baseline["rows"])
    if ids != set(candidate["rows"]):
        raise ValueError("batches do not contain the same task IDs")
    for task_id in sorted(ids):
        before, after = baseline["manifests"][task_id], candidate["manifests"][task_id]
        identity = ("run_seed", "sample_seed", "training_category", "requested_task_intent",
                    "requested_task_style", "requested_task_type", "hops")
        if any(before.get(field) != after.get(field) for field in identity):
            raise ValueError(f"sampling identity differs for {task_id}")
    selection_fields = ("selection_seed", "selection_rule", "quotas", "selected")
    if any(
        baseline["selection"].get(field) != candidate["selection"].get(field)
        for field in selection_fields
    ):
        raise ValueError("build probe selection differs between batches")
    transitions: Counter[str] = Counter()
    failure_migrations: Counter[str] = Counter()
    for task_id in sorted(ids):
        before, after = baseline["rows"][task_id], candidate["rows"][task_id]
        before_good = before.get("status") == "completed" and before.get("buildable") is True
        after_good = after.get("status") == "completed" and after.get("buildable") is True
        transitions[f"{int(before_good)}->{int(after_good)}"] += 1
        if not before_good or not after_good:
            old_failure = before.get("failure_class") if not before_good else "PASS"
            new_failure = after.get("failure_class") if not after_good else "PASS"
            failure_migrations[f"{old_failure or 'UNKNOWN'}->{new_failure or 'UNKNOWN'}"] += 1

    def probe_rows(value: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
        summary = value["probes"]
        if summary is None:
            return None
        rows = summary.get("rows")
        if not isinstance(rows, list):
            raise ValueError("build probe summary has no rows")
        mapped = {item.get("task_id"): item for item in rows if isinstance(item, dict)}
        if len(mapped) != len(rows):
            raise ValueError("build probe summary has duplicate task IDs")
        return mapped

    before_probes, after_probes = probe_rows(baseline), probe_rows(candidate)
    probe_comparison = None
    if before_probes is not None and after_probes is not None:
        if set(before_probes) != set(after_probes):
            raise ValueError("build probe summaries cover different tasks")
        raw: Counter[str] = Counter()
        audited: Counter[str] = Counter()
        audit_coverage = 0
        for task_id in sorted(before_probes):
            before, after = before_probes[task_id], after_probes[task_id]
            raw[f"{int(before.get('build_success') is True)}->{int(after.get('build_success') is True)}"] += 1
            if isinstance(before.get("independent_reward_audit"), bool) and isinstance(
                after.get("independent_reward_audit"), bool
            ):
                audit_coverage += 1
                old_good = before.get("build_success") is True and before["independent_reward_audit"]
                new_good = after.get("build_success") is True and after["independent_reward_audit"]
                audited[f"{int(old_good)}->{int(new_good)}"] += 1
        probe_comparison = {
            "selected": len(before_probes), "raw_build_transitions": dict(sorted(raw.items())),
            "audited_coverage": audit_coverage,
            "audited_end_to_end_transitions": dict(sorted(audited.items())),
        }
    return {
        "scope": "paired_development_diagnostic_only",
        "baseline": baseline["root"], "candidate": candidate["root"],
        "requested": len(ids), "generation_transitions": dict(sorted(transitions.items())),
        "failure_migrations": dict(sorted(failure_migrations.items())),
        "build_probes": probe_comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(batch(args.baseline), batch(args.candidate))
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
