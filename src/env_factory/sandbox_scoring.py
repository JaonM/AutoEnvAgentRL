"""Immutable identity and structural verification for sandbox score evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .evidence.material_artifacts import docker_build_context_digests


SCORE_RUBRIC = (
    ("delivery_integrity", 1.0, True),
    ("contract_and_tool_identity", 1.0, True),
    ("semantic_business_fidelity", 0.5, True),
    ("business_acceptance", 1.0, True),
    ("sandbox_pytest", 1.0, True),
    ("runtime_genericity", 0.5, True),
    ("outer_conformance", 1.0, True),
    ("mutation_resistance", 1.0, True),
    ("training_readiness", 1.0, True),
    ("declared_training_policy", 2.0, True),
)
REQUIRED_CHECKS = frozenset(name for name, _, _ in SCORE_RUBRIC)
RUBRIC_BY_NAME = {
    name: {"weight": weight, "critical": critical}
    for name, weight, critical in SCORE_RUBRIC
}


def sandbox_quality_factors(root: Path) -> dict[str, float]:
    """Measure quality within passing gates from bound offline evidence."""
    try:
        review = json.loads((root / "review_report.json").read_text(encoding="utf-8"))
        agentic = json.loads((root / "agentic_training_value.json").read_text(encoding="utf-8"))
        review_score = review.get("score")
        if (isinstance(review_score, bool) or not isinstance(review_score, (int, float))
                or not math.isfinite(review_score)):
            review_score = 0.0
        counterfactuals = agentic.get("evidence", {}).get("counterfactuals", {})
        success = counterfactuals.get("goal_success", {}).get("reward")
        negatives = [
            item["reward"] for name, item in counterfactuals.items()
            if name != "goal_success" and isinstance(item, dict)
            and item.get("status", "completed") == "completed"
            and isinstance(item.get("reward"), (int, float))
            and not isinstance(item["reward"], bool)
            and math.isfinite(item["reward"])
        ]
        margin = success - max(negatives) if (
            isinstance(success, (int, float)) and not isinstance(success, bool)
            and math.isfinite(success) and negatives
        ) else 0.0
        task_path = root / "task.json"
        if task_path.is_file() and isinstance(success, (int, float)):
            from .contracts.reward_contract import terminal_outcome_weight
            answer_weight = terminal_outcome_weight(json.loads(task_path.read_text()))
            normalized = []
            for name, item in counterfactuals.items():
                if name == "goal_success" or not isinstance(item, dict) or item.get("status", "completed") != "completed":
                    continue
                value = item.get("reward")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                expected_loss = answer_weight if name in {"wrong_final_answer", "wrong_numeric_answer"} and answer_weight > 0 else 1.0
                normalized.append((success - value) / expected_loss)
            if normalized:
                margin = min(normalized)
    except (OSError, ValueError, TypeError, AttributeError):
        review_score, margin = 0.0, 0.0
    return {
        "semantic_business_fidelity": round(max(0.0, min(1.0, review_score)), 4),
        "declared_training_policy": round(max(0.0, min(1.0, margin)), 4),
    }


def review_source_hashes(root: Path) -> dict[str, str]:
    """Bind review to every file shipped into the sandbox image."""
    return docker_build_context_digests(root)


def validate_semantic_review(report: Any, root: Path) -> float:
    """Validate one passing independent review against the current sandbox."""
    if not isinstance(report, dict):
        raise ValueError("review_report.json 必须是 object")
    if not isinstance(report.get("review_run_id"), str) or not report["review_run_id"].strip():
        raise ValueError("review_report.review_run_id 缺失")
    if not isinstance(report.get("reviewed_at"), str) or not report["reviewed_at"].strip():
        raise ValueError("review_report.reviewed_at 缺失")
    hashes = report.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("review_report.source_hashes 缺失")
    if hashes != review_source_hashes(root):
        raise ValueError("semantic review source hashes differ from current sandbox")
    if report.get("status") not in {"pass", "fail"}:
        raise ValueError("review_report.status 必须是 pass 或 fail")
    adjudication = report.get("adjudication")
    if adjudication is not None and (
        not isinstance(adjudication, dict) or adjudication.get("status") != "resolved"
        or adjudication.get("coverage_complete") is not True
        or adjudication.get("source_hashes") != hashes
        or adjudication.get("review_run_id") != report.get("review_run_id")
    ):
        raise ValueError("semantic review lacks resolved, bound adjudication")
    score = report.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("review_report.score 必须属于 [0,1]")
    checked = report.get("checked_modules")
    if (not isinstance(checked, list) or not checked
            or any(not isinstance(item, str) or not item.strip() for item in checked)):
        raise ValueError("review_report.checked_modules 不能为空且必须为字符串列表")
    if not {"business_tools", "reward", "user_simulator", "runtime_contract"} <= set(checked):
        raise ValueError("semantic review module coverage is incomplete")
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise ValueError("review_report.findings 必须是 list")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"review_report.findings[{index}] 必须是 object")
        if finding.get("severity") not in {"critical", "high", "medium", "low"}:
            raise ValueError(f"review_report.findings[{index}].severity 无效")
        if ("tool_name" not in finding
                or finding["tool_name"] is not None and not isinstance(finding["tool_name"], str)):
            raise ValueError(f"review_report.findings[{index}].tool_name 无效")
        if finding.get("tool_category") not in {None, "task", "unrelated", "related_irrelevant"}:
            raise ValueError(f"review_report.findings[{index}].tool_category 无效")
        for key in ("category", "evidence", "contract_reference"):
            if not isinstance(finding.get(key), str) or not finding[key].strip():
                raise ValueError(f"review_report.findings[{index}] 缺少 {key}")
    repairs = report.get("required_repairs", [])
    if (not isinstance(repairs, list)
            or any(not isinstance(item, str) or not item.strip() for item in repairs)):
        raise ValueError("review_report.required_repairs 必须是字符串列表")
    blocking = [item for item in findings if item["severity"] in {"critical", "high"}]
    if report["status"] == "pass" and blocking:
        raise ValueError("review_report.status=pass 但仍有 high/critical finding")
    if report["status"] == "fail":
        raise ValueError("semantic review failed")
    return float(score)


def _base_evidence_digest(root_value: str, project_value: str):
    root, project = Path(root_value), Path(project_value)
    paths = [
        *sorted((project / "src/env_factory").rglob("*.py")),
        *sorted((project / "scripts").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    digest.update(b"envfactory-sandbox-score-evidence-v3\0")
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(project)).encode())
            digest.update(path.read_bytes())
    for label, file_digest in docker_build_context_digests(root).items():
        digest.update(label.encode())
        digest.update(file_digest.encode())
    return digest


def evidence_fingerprint(
    root: Path, project: Path, *, task_path: Path | None = None
) -> str:
    """Bind scoring evidence to evaluator code and evaluated implementation."""
    digest = _base_evidence_digest(str(root.resolve()), str(project.resolve())).copy()
    task = task_path or root / "task.json"
    if task.is_file():
        digest.update(b"task.json")
        digest.update(task.read_bytes())
    return digest.hexdigest()


def valid_score_report(
    report: Any,
    *,
    root: Path,
    project: Path,
    threshold: float,
    task_path: Path | None = None,
) -> bool:
    """Recompute the deterministic score envelope without rerunning evidence."""
    if not isinstance(report, Mapping):
        return False
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    names: list[str] = []
    raw = 0.0
    failed_critical: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping) or set(check) != {
            "name", "weight", "passed", "evidence", "critical"
        }:
            return False
        name = check.get("name")
        weight = check.get("weight")
        if (
            not isinstance(name, str) or not name
            or isinstance(weight, bool) or not isinstance(weight, (int, float))
            or weight < 0
            or not isinstance(check.get("passed"), bool)
            or not isinstance(check.get("critical"), bool)
            or not isinstance(check.get("evidence"), str)
        ):
            return False
        names.append(name)
        if check["passed"]:
            raw += float(weight)
        elif check["critical"]:
            failed_critical.append(name)
    if names != [name for name, _, _ in SCORE_RUBRIC]:
        return False
    if any(
        check["weight"] != RUBRIC_BY_NAME[check["name"]]["weight"]
        or check["critical"] is not RUBRIC_BY_NAME[check["name"]]["critical"]
        for check in checks
    ):
        return False
    eligible = not failed_critical
    kind = report.get("score_kind", "hard_gate_binary")
    if kind == "gated_weighted_10_point":
        factors = sandbox_quality_factors(root)
        if report.get("quality_factors") != factors:
            return False
        weighted = sum(
            check["weight"] * factors.get(check["name"], 1.0)
            for check in checks if check["passed"]
        )
        score = round(weighted, 2) if eligible else 0.0
    elif kind == "hard_gate_binary":
        score = round(raw, 2) if eligible else 0.0
    else:
        return False
    try:
        return (
            report.get("mode") == "offline_executable"
            and report.get("score_scope", "offline_sandbox_qualification")
            == "offline_sandbox_qualification"
            and report.get("delivery_verified", False) is False
            and report.get("verification_status", "requires_live") == "requires_live"
            and report.get("network_used") is False
            and report.get("model_used") is False
            and report.get("live_rollout_verified") is False
            and report.get("threshold") == threshold
            and report.get("score") == score
            and report.get("raw_score", round(raw, 2)) == round(raw, 2)
            and report.get("eligible") is eligible
            and report.get("passed") is (eligible and score >= threshold)
            and report.get("failed_critical_gates") == failed_critical
            and report.get("evidence_fingerprint")
            == evidence_fingerprint(root, project, task_path=task_path)
            and isinstance(report.get("model"), str) and bool(report["model"])
            and isinstance(report.get("review_model"), str)
            and bool(report["review_model"])
        )
    except (OSError, ValueError):
        return False
