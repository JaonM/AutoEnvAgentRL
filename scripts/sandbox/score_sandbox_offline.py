#!/usr/bin/env python3
"""Fully offline, executable scoring for generated Agentic-RL sandboxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sandbox import Check, evaluate, rubric_check, score_checks
from env_factory.sandbox_scoring import sandbox_quality_factors


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def offline_semantic_check(root: Path, checks: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Use executable contract evidence instead of an LLM review report."""
    try:
        task = load(root / "task.json")
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"task contract unavailable: {exc}"
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    if not isinstance(scenarios, list):
        return False, "acceptance_contract.executable_scenarios is not a list"
    by_kind = {
        item.get("kind"): item for item in scenarios
        if isinstance(item, dict) and isinstance(item.get("kind"), str)
    }
    missing = sorted({"goal_success", "goal_failure"} - set(by_kind))
    if missing:
        return False, f"missing executable scenario kinds: {missing}"
    if task.get("noise_tools") and "noise_selection" not in by_kind:
        return False, "noise tools exist but noise_selection scenario is missing"

    success_steps = by_kind["goal_success"].get("steps", [])
    operations = [step.get("operation") for step in success_steps if isinstance(step, dict)]
    training = task.get("training_contract", {})
    category = training.get("category", task.get("training_category", "multi_step_agentic"))
    tool_required = category != "direct_response"
    required_operations = {"agent_response", "reward"}
    if tool_required:
        required_operations.add("tool_call")
    if not required_operations <= set(operations):
        return False, f"goal_success misses operations: {sorted(required_operations - set(operations))}"
    noise_names = {
        item.get("name") or item.get("tool_name")
        for item in task.get("noise_tools", []) if isinstance(item, dict)
    }
    task_tool_schemas = {
        item.get("function", {}).get("name"): item.get("function", {}).get("parameters", {})
        for item in task.get("tools", [])
        if isinstance(item, dict)
        and isinstance(item.get("function"), dict)
        and item.get("function", {}).get("name") not in noise_names
    }
    useful_calls = [
        step for step in success_steps
        if isinstance(step, dict)
        and step.get("operation") == "tool_call"
        and step.get("tool_name") in task_tool_schemas
        and isinstance(step.get("arguments"), dict)
        and set(task_tool_schemas[step["tool_name"]].get("required", []))
        <= set(step["arguments"])
    ]
    if tool_required and not useful_calls:
        return False, "goal_success has no schema-valid business tool call"
    if not tool_required and useful_calls:
        return False, "direct_response goal_success unexpectedly calls a business tool"

    policy_gate = "declared_training_policy" if "declared_training_policy" in checks else "agentic_training_value"
    required_gates = (
        "contract_and_tool_identity", "business_acceptance", "runtime_genericity",
        "mutation_resistance", "training_readiness", policy_gate,
    )
    failed = [name for name in required_gates if not checks.get(name, {}).get("passed")]
    if failed:
        return False, f"executable semantic evidence failed: {failed}"
    return True, (
        f"offline executable semantics passed; scenarios={len(scenarios)} "
        f"category={category} business_calls={len(useful_calls)}"
    )


def evaluate_offline(
    root: Path, *, project: Path, threshold: float, build_finalization: bool = False, reuse_evidence: bool = True,
) -> dict[str, Any]:
    result = evaluate(
        root, project=project, execute=True, threshold=threshold, offline=True,
        build_finalization=build_finalization, reuse_evidence=reuse_evidence,
    )
    by_name = {item["name"]: item for item in result["checks"]}
    executable_ok, executable_evidence = offline_semantic_check(root, by_name)
    review = by_name.get("semantic_business_fidelity", {})
    review_ok = review.get("passed") is True
    semantic_ok = review_ok and executable_ok
    semantic_evidence = (
        f"independent_review={review_ok}: {review.get('evidence', 'missing')}; "
        f"executable_semantics={executable_ok}: {executable_evidence}"
    )
    checks = []
    for item in result["checks"]:
        if item["name"] == "semantic_business_fidelity":
            checks.append(rubric_check(
                item["name"], semantic_ok, semantic_evidence,
            ))
        else:
            checks.append(Check(**item))
    scored = score_checks(
        checks, threshold=threshold, quality_factors=sandbox_quality_factors(root),
    )
    scored.update({
        "root": str(root),
        "mode": "offline_executable",
        "score_scope": "offline_sandbox_qualification",
        "delivery_verified": False,
        "verification_status": "requires_live",
        "live_rollout_verified": False,
        "evidence_fingerprint": result.get("evidence_fingerprint"),
        "evidence_reused": result.get("evidence_reused", False),
        "network_used": False,
        "model_used": False,
        "model": result.get("model"),
        "review_model": result.get("review_model"),
    })
    return scored


def discover(inputs: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for value in inputs:
        path = value.resolve()
        if (path / "task.json").is_file() and (path / "app.py").is_file():
            roots.append(path)
            continue
        if path.is_dir():
            roots.extend(
                candidate.parent for candidate in sorted(path.rglob("task.json"))
                if (candidate.parent / "app.py").is_file()
            )
    return list(dict.fromkeys(roots))


def main() -> int:
    parser = argparse.ArgumentParser(description="完全离线执行并评分 Agentic-RL 沙箱")
    parser.add_argument("roots", nargs="+", type=Path, help="沙箱目录或包含多个沙箱的目录")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--output", type=Path, help="批量汇总 JSON；默认打印到 stdout")
    parser.add_argument("--fresh", action="store_true", help="Force fresh executable qualification")
    parser.add_argument("--no-individual", action="store_true", help="不写入各沙箱 offline_sandbox_score.json")
    parser.add_argument("--build-finalization", action="store_true", help="仅构建收尾时接受 offline_scoring 待完成状态")
    args = parser.parse_args()
    roots = discover(args.roots)
    if not roots:
        parser.error("没有发现包含 task.json 和 app.py 的沙箱目录")
    reports = []
    for root in roots:
        report = evaluate_offline(
            root, project=args.project.resolve(), threshold=args.threshold,
            build_finalization=args.build_finalization, reuse_evidence=not args.fresh,
        )
        reports.append(report)
        if not args.no_individual:
            (root / "offline_sandbox_score.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    summary = {
        "mode": "offline_executable",
        "threshold": args.threshold,
        "total": len(reports),
        "passed": sum(item["passed"] for item in reports),
        "failed": sum(not item["passed"] for item in reports),
        "all_passed": all(item["passed"] for item in reports),
        "sandboxes": reports,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
