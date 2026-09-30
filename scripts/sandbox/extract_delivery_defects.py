#!/usr/bin/env python3
"""Select current, evidenced defects from a failed sandbox acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from env_factory.sandbox_scoring import review_source_hashes


def extract(root: Path, failed_phase: str) -> list[dict[str, Any]]:
    report_path = root / "review_report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            candidate = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                report = candidate
        except (OSError, json.JSONDecodeError):
            pass

    findings = report.get("findings")
    current_review = (
        report.get("status") == "fail"
        and isinstance(findings, list)
        and bool(findings)
        and report.get("source_hashes") == review_source_hashes(root)
    )
    if not current_review:
        error_path = root / "last_delivery_error.txt"
        error = error_path.read_text(encoding="utf-8", errors="replace") if error_path.is_file() else "验收失败"
        diagnostic = "\n".join(error.strip().splitlines()[-40:])[-4000:]
        findings = [{
            "severity": "critical", "category": "delivery_failure", "file": None,
            "line": None, "evidence": f"failed_phase={failed_phase}\n{diagnostic}",
            "contract_reference": f"outer workflow validation: {failed_phase}",
            "fix_required": "只修复当前失败门禁的首个根因；完整日志见 last_delivery_error.txt",
        }]
        report = {}

    adjudication = report.get("adjudication")
    if isinstance(adjudication, dict):
        confirmed = {v["finding_index"] for v in adjudication.get("findings", []) if v.get("status") == "confirmed"}
        findings = [finding for index, finding in enumerate(findings) if index in confirmed]
    normalized = []
    for index, finding in enumerate(findings, 1):
        if not isinstance(finding, dict):
            continue
        item = dict(finding)
        item["id"] = item.get("id") or f"DEF-{index:03d}"
        item["review_run_id"] = report.get("review_run_id")
        item["source_hashes"] = report.get("source_hashes", {})
        normalized.append(item)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--failed-phase", required=True)
    args = parser.parse_args()
    defects = extract(args.root, args.failed_phase)
    (args.root / "defects.json").write_text(
        json.dumps(defects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(len(defects))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
