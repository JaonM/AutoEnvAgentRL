#!/usr/bin/env python3
"""Score and filter generated EnvFactory tasks without an LLM call."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from env_factory.tasks.task_quality import discover_task_files, error_report, score_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评分并过滤 Agentic-RL 任务样本")
    parser.add_argument("root", nargs="?", type=Path, default=Path("output/task"))
    parser.add_argument("--min-score", type=float, default=8.0, help="合格阈值，默认 8.0（含）")
    parser.add_argument("--report", type=Path, default=Path("output/task_quality_report.json"))
    parser.add_argument("--csv", type=Path, help="可选 CSV 汇总路径")
    parser.add_argument("--accepted-dir", type=Path, help="将合格任务复制到该目录")
    parser.add_argument("--high-value-dir", type=Path, help="仅将 high_value 任务复制到该目录")
    parser.add_argument("--rejected-dir", type=Path, help="将不合格任务复制到该目录")
    parser.add_argument("--fail-on-low-score", action="store_true", help="存在低分任务时返回退出码 1")
    args = parser.parse_args()
    if not 0 <= args.min_score <= 10:
        parser.error("--min-score 必须位于 [0, 10]")
    return args


def copy_selected(report, destination: Path) -> None:
    target = destination / report.task_id
    if target.exists():
        raise SystemExit(f"目标已存在，拒绝覆盖：{target}")
    shutil.copytree(Path(report.path).parent, target)


def main() -> int:
    args = parse_args()
    paths = discover_task_files(args.root)
    if not paths:
        raise SystemExit(f"未找到 task.json：{args.root}")
    reports = []
    for path in paths:
        try:
            reports.append(score_file(path, min_score=args.min_score))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reports.append(error_report(path, exc, min_score=args.min_score))
    category_summary = {}
    for category in sorted({item.training_category for item in reports}):
        selected = [item for item in reports if item.training_category == category]
        category_summary[category] = {
            "total": len(selected),
            "accepted": sum(item.passed for item in selected),
            "high_value": sum(item.tier == "high_value" for item in selected),
            "average_score": round(sum(item.score for item in selected) / len(selected), 2),
        }
    payload = {
        "rubric_version": "1.0",
        "threshold": args.min_score,
        "summary": {
            "total": len(reports),
            "accepted": sum(item.passed for item in reports),
            "rejected": sum(not item.passed for item in reports),
            "high_value": sum(item.tier == "high_value" for item in reports),
            "usable": sum(item.tier == "usable" for item in reports),
            "average_score": round(sum(item.score for item in reports) / len(reports), 2),
            "by_training_category": category_summary,
        },
        "tasks": [item.to_dict() for item in reports],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_id", "training_category", "score", "passed", "tier", "findings", "path"])
            for item in reports:
                writer.writerow([item.task_id, item.training_category, item.score, item.passed, item.tier, "；".join(item.findings), item.path])
    for item in reports:
        if item.passed and args.accepted_dir:
            copy_selected(item, args.accepted_dir)
        if item.tier == "high_value" and args.high_value_dir:
            copy_selected(item, args.high_value_dir)
        if not item.passed and args.rejected_dir:
            copy_selected(item, args.rejected_dir)
        print(f"{item.task_id}\t{item.training_category}\t{item.score:.2f}\t{item.tier.upper()}\t{'；'.join(item.findings)}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 1 if args.fail_on_low_score and payload["summary"]["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
