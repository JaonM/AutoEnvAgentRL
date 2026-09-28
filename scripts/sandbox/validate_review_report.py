#!/usr/bin/env python3
"""Validate the independent semantic review result used by sandbox builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"review_report.json 无法解析：{exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit("review_report.json 必须是 object")
    if not isinstance(report.get("review_run_id"), str) or not report["review_run_id"].strip():
        raise SystemExit("review_report.review_run_id 缺失，报告可能不是当前验收轮次生成")
    if not isinstance(report.get("reviewed_at"), str) or not report["reviewed_at"].strip():
        raise SystemExit("review_report.reviewed_at 缺失")
    if not isinstance(report.get("source_hashes"), dict) or not report["source_hashes"]:
        raise SystemExit("review_report.source_hashes 缺失，无法绑定当前产物")
    if report.get("status") not in {"pass", "fail"}:
        raise SystemExit("review_report.status 必须是 pass 或 fail")
    score = report.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise SystemExit("review_report.score 必须属于 [0,1]")
    checked = report.get("checked_modules")
    if not isinstance(checked, list) or not checked:
        raise SystemExit("review_report.checked_modules 不能为空")
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise SystemExit("review_report.findings 必须是 list")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise SystemExit(f"review_report.findings[{index}] 必须是 object")
        if finding.get("severity") not in {"critical", "high", "medium", "low"}:
            raise SystemExit(f"review_report.findings[{index}].severity 无效")
        if "tool_name" not in finding or finding.get("tool_name") is not None and not isinstance(finding.get("tool_name"), str):
            raise SystemExit(f"review_report.findings[{index}].tool_name 必须是 string 或 null")
        if finding.get("tool_category") not in {None, "task", "unrelated", "related_irrelevant"}:
            raise SystemExit(f"review_report.findings[{index}].tool_category 无效")
        for key in ("category", "evidence", "contract_reference"):
            if not isinstance(finding.get(key), str) or not finding[key].strip():
                raise SystemExit(f"review_report.findings[{index}] 缺少 {key}")
    repairs = report.get("required_repairs", [])
    if not isinstance(repairs, list) or any(not isinstance(item, str) or not item.strip() for item in repairs):
        raise SystemExit("review_report.required_repairs 必须是字符串列表")
    blocking = [item for item in findings if item.get("severity") in {"critical", "high"}]
    if report["status"] == "pass" and blocking:
        raise SystemExit("review_report.status=pass 但仍有 high/critical finding")
    if report["status"] == "fail":
        raise SystemExit("semantic review failed: " + json.dumps(blocking or findings, ensure_ascii=False))
    print("semantic review: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
