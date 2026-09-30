#!/usr/bin/env python3
"""Evidence-based 10-point scoring for generated Agentic-RL sandboxes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from env_factory.sandbox_scoring import (
    RUBRIC_BY_NAME, evidence_fingerprint, sandbox_quality_factors,
    validate_semantic_review,
)
from env_factory.tasks.task_quality import score_file


REQUIRED_FILES = (
    "task.json", "BUILD_CONTRACT.json", "app.py", "task_impl.py", "tools.json",
    "sandbox_runtime.py", "runtime_llm.py", "acceptance.sh", "Dockerfile",
    "requirements-dev.txt", "IMPLEMENTATION_REPORT.md",
)


class Check(NamedTuple):
    name: str
    weight: float
    passed: bool
    evidence: str
    critical: bool = False


def rubric_check(name: str, passed: bool, evidence: str) -> Check:
    spec = RUBRIC_BY_NAME[name]
    return Check(name, spec["weight"], passed, evidence, spec["critical"])


def score_checks(
    checks: list[Check], *, threshold: float = 8.0,
    quality_factors: dict[str, float] | None = None,
) -> dict[str, Any]:
    quality_factors = quality_factors or {}
    raw = round(sum(item.weight for item in checks if item.passed), 2)
    weighted = round(sum(
        item.weight * quality_factors.get(item.name, 1.0)
        for item in checks if item.passed
    ), 2)
    failed_critical = [item.name for item in checks if item.critical and not item.passed]
    eligible = not failed_critical
    score = weighted if eligible else 0.0
    return {
        "score": score,
        "raw_score": raw,
        "score_kind": "gated_weighted_10_point",
        "score_scope": "offline_sandbox_qualification",
        "qualification": "eligible" if eligible else "rejected",
        "score_interpretation": "Diagnostic quality within mandatory gates; not an agent success probability",
        "quality_factors": quality_factors,
        "eligible": eligible,
        "passed": eligible and score >= threshold,
        "threshold": threshold,
        "failed_critical_gates": failed_critical,
        "checks": [item._asdict() for item in checks],
    }


def run(command: list[str], *, cwd: Path, timeout: int = 240, env: dict[str, str] | None = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        )
        output = completed.stdout[-4000:].strip()
        return completed.returncode == 0, output or f"exit={completed.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_check(root: Path, *, threshold: float) -> tuple[bool, str]:
    try:
        task = json_file(root / "task.json")
        contract = json_file(root / "BUILD_CONTRACT.json")
        tools = json_file(root / "tools.json")
    except (OSError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not isinstance(task, dict):
        return False, "task.json must be a JSON object"
    expected = {key: value for key, value in task.items() if key != "actions"}
    if contract != expected:
        return False, "BUILD_CONTRACT.json is not the immutable task projection"
    if tools != task.get("tools"):
        return False, "tools.json differs from task.json.tools"
    try:
        quality = score_file(root / "task.json", min_score=threshold)
    except Exception as exc:
        return False, f"task quality could not be verified: {exc}"
    if not quality.passed:
        return False, (
            f"task quality gate failed: score={quality.score} "
            f"eligible={quality.eligible} findings={quality.eligibility_failures}"
        )
    return True, (
        "contract projection and tools schema are identical; "
        f"task quality score={quality.score} eligible=true"
    )


def review_check(root: Path) -> tuple[bool, str, float]:
    try:
        report = json_file(root / "review_report.json")
        score = validate_semantic_review(report, root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc), 0.0
    return True, f"status=pass score={score} blocking=0", score


def delivery_status_ok(status: Any, *, build_finalization: bool = False) -> bool:
    if not isinstance(status, dict):
        return False
    return (status.get("status") == "success" and status.get("success") is True) or (
        build_finalization
        and status.get("status") == "pending"
        and status.get("success") is False
        and status.get("phase") == "offline_scoring"
    )


def evaluate(
    root: Path, *, project: Path, execute: bool, threshold: float,
    offline: bool = False, build_finalization: bool = False, reuse_evidence: bool = True,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file() or not (root / name).stat().st_size]
    try:
        status = json_file(root / "status.json")
    except (OSError, json.JSONDecodeError):
        status = {}
    if not isinstance(status, dict):
        status = {}
    delivery_ok = not missing and delivery_status_ok(
        status, build_finalization=build_finalization,
    )
    checks = [rubric_check(
        "delivery_integrity", delivery_ok,
        f"missing={missing}; status={status.get('status')}",
    )]

    contract_ok, contract_evidence = contract_check(root, threshold=threshold)
    checks.append(rubric_check(
        "contract_and_tool_identity", contract_ok, contract_evidence,
    ))

    review_ok, review_evidence, review_score = review_check(root)
    checks.append(rubric_check(
        "semantic_business_fidelity", review_ok, review_evidence,
    ))

    # A failed preflight makes every later gate ineligible. Avoid launching
    # acceptance, pytest and mutation runs against an incomplete delivery.
    preflight_failures = [item.name for item in checks if not item.passed]
    if execute and preflight_failures:
        reason = "skipped: prerequisite failed: " + ", ".join(preflight_failures)
        checks.extend(rubric_check(name, False, reason) for name in (
            "business_acceptance", "sandbox_pytest", "runtime_genericity",
            "outer_conformance", "mutation_resistance", "training_readiness",
            "declared_training_policy",
        ))
        result = score_checks(
            checks, threshold=threshold, quality_factors=sandbox_quality_factors(root),
        )
        result.update({
            "root": str(root), "review_score": review_score,
            "model": status.get("model"), "review_model": status.get("review_model"),
            "executed": execute,
            "evidence_fingerprint": evidence_fingerprint(root, project),
            "skipped_checks": reason,
        })
        return result

    from env_factory.evidence.gate_cache import load as load_gates, save as save_gates
    cached = load_gates(root, project) if execute and reuse_evidence else None
    if cached:
        acceptance_ok, acceptance_out = cached['business_acceptance']
        pytest_ok, pytest_out = cached['sandbox_pytest']
        runtime_ok, runtime_out = cached['runtime_genericity']
        outer_ok, outer_out = cached['outer_conformance']
        mutation_ok, mutation_out = cached['mutation_resistance']
        readiness_ok, readiness_out = cached['training_readiness']
        agentic_ok, agentic_out = cached['declared_training_policy']
    elif execute:
        env = os.environ.copy()
        if offline:
            env["SANDBOX_EVALUATOR_MOCK"] = "1"
            for key in ("SANDBOX_LLM_API_KEY", "SANDBOX_LLM_BASE_URL", "SANDBOX_EXTERNAL_CAPABILITY_URL", "LLM_API_KEY", "LLM_BASE_URL"):
                env.pop(key, None)
        env.setdefault("SANDBOX_TRAINER_API_KEY", "envfactory-score-key")
        env.setdefault("SANDBOX_EVALUATOR_MOCK", "true")
        acceptance_ok, acceptance_out = run(["bash", "./acceptance.sh"], cwd=root, env=env)
        pytest_ok, pytest_out = run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env)
        runtime_ok, runtime_out = run([sys.executable, str(project / "scripts/sandbox/validate_sandbox_runtime.py"), "--root", str(root)], cwd=project)
        with tempfile.TemporaryDirectory(prefix="envfactory-score-outer-") as directory:
            outer_ok, outer_out = run([
                sys.executable, str(project / "scripts/sandbox/generate_outer_conformance.py"),
                "--root", str(root), "--output", directory, "--check",
            ], cwd=project)
        mutation_ok, mutation_out = run([sys.executable, str(project / "scripts/sandbox/run_mutation_tests.py"), "--root", str(root)], cwd=project, timeout=600, env=env)
        readiness_ok, readiness_out = run([sys.executable, str(project / "scripts/sandbox/validate_training_readiness.py"), "--root", str(root)], cwd=project, env=env)
        agentic_ok, agentic_out = run([sys.executable, str(project / "scripts/sandbox/validate_agentic_training_value.py"), "--root", str(root)], cwd=project, env=env)
        # Mutation probes intentionally rerun acceptance under broken modes and
        # may overwrite acceptance_result.json. Finish with a clean baseline
        # so automatic scoring never leaves the delivered sandbox corrupted.
        restored_ok, restored_out = run(["bash", "./acceptance.sh"], cwd=root, env=env)
        if not restored_ok:
            acceptance_ok = False
            acceptance_out += f"\nfinal baseline restoration failed: {restored_out}"
    else:
        result = {}
        try:
            result = json_file(root / "acceptance_result.json")
        except (OSError, json.JSONDecodeError):
            pass
        acceptance_ok = result.get("business_acceptance") == "passed"
        acceptance_out = f"acceptance_result={result}"
        pytest_logs = sorted(root.glob("pytest_*.log"))
        pytest_ok = bool(pytest_logs) and all("failed" not in path.read_text(encoding="utf-8", errors="replace").lower() for path in pytest_logs)
        pytest_out = f"pytest_logs={len(pytest_logs)}"
        runtime_ok = bool(status.get("success"))
        runtime_out = "inferred from successful completed workflow"
        outer_ok = (root / ".outer_conformance").is_dir()
        outer_out = "outer conformance artifacts present" if outer_ok else "missing .outer_conformance"
        mutation_ok = bool(status.get("success"))
        mutation_out = "inferred from successful completed workflow"
        try:
            readiness = json_file(root / "training_readiness.json")
        except (OSError, json.JSONDecodeError):
            readiness = {}
        readiness_ok = readiness.get("training_ready") is True
        readiness_out = json.dumps(readiness.get("failed_gates", []), ensure_ascii=False)
        try:
            agentic = json_file(root / "agentic_training_value.json")
        except (OSError, json.JSONDecodeError):
            agentic = {}
        agentic_ok = agentic.get("curriculum_training_ready", agentic.get("agentic_training_ready")) is True
        agentic_out = json.dumps({
            "failed_gates": agentic.get("failed_gates", []),
            "counterfactuals": agentic.get("evidence", {}).get("counterfactuals", {}),
        }, ensure_ascii=False)

    checks.extend([
        rubric_check("business_acceptance", acceptance_ok, acceptance_out),
        rubric_check("sandbox_pytest", pytest_ok, pytest_out),
        rubric_check("runtime_genericity", runtime_ok, runtime_out),
        rubric_check("outer_conformance", outer_ok, outer_out),
        rubric_check("mutation_resistance", mutation_ok, mutation_out),
        rubric_check("training_readiness", readiness_ok, readiness_out),
        rubric_check("declared_training_policy", agentic_ok, agentic_out),
    ])
    if execute and not cached:
        from env_factory.evidence.gate_cache import GATES
        save_gates(root, project, {item.name: [item.passed, item.evidence] for item in checks if item.name in GATES})
    result = score_checks(
        checks, threshold=threshold, quality_factors=sandbox_quality_factors(root),
    )
    result['evidence_reused'] = bool(cached)
    result.update({
        "root": str(root),
        "review_score": review_score,
        "model": status.get("model"),
        "review_model": status.get("review_model"),
        "executed": execute,
        "evidence_fingerprint": evidence_fingerprint(root, project),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="对生成沙箱执行 10 分制质量评分")
    parser.add_argument("root", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--threshold", type=float, default=8.0)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", dest="execute", action="store_true", default=True,
                           help="重新执行全部验收门禁（默认）")
    execution.add_argument("--reuse-evidence", dest="execute", action="store_false",
                           help="仅供诊断：复用已有验收证据，不作为新鲜评分")
    parser.add_argument("--fresh", action="store_true", help="Force fresh executable qualification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root.resolve(), project=args.project.resolve(), execute=args.execute, threshold=args.threshold, reuse_evidence=not args.fresh)
    output = args.output or args.root / "sandbox_score.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
