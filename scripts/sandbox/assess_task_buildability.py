#!/usr/bin/env python3
"""Fail fast when a task contract cannot be implemented by the shared sandbox."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from env_factory.sandbox_runtime import (
    DeclarativeMetricEvaluator, EpisodeStore, ManifestDataStore, SandboxError,
)
from env_factory.tasks.task_quality import score_file
from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline
from env_factory.tasks.task_spec import TaskSpecError, validate_task_spec
from env_factory.contracts.runtime_contract import missing_system_endpoints
from env_factory.contracts.tool_chain_contract import tool_chain_issues as _tool_chain_issues
from env_factory.contracts.reward_contract import ambiguous_metric_captures, reward_contract_issues


def _issue(code: str, owner: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {"code": code, "owner": owner, "message": message, "evidence": evidence}


def _manifest_root(task_root: Path, manifest: dict[str, Any]) -> Path:
    """Read business files from the task's declared root."""
    declared = Path(str(manifest["root"]))
    if declared.is_absolute():
        return declared
    return task_root / declared


def assess(root: Path, *, threshold: float = 8.0) -> dict[str, Any]:
    task_path = root / "task.json"
    issues: list[dict[str, Any]] = []
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"buildable": False, "failure_owner": "task_generation", "issues": [
            _issue("TASK_ARTIFACT_INVALID", "task_generation", str(exc))
        ]}
    quality = score_file(task_path, min_score=threshold).to_dict()
    if not quality.get("eligible") or quality.get("score", 0) < threshold:
        issues.append(_issue(
            "TASK_QUALITY_GATE", "task_generation", "task is not eligible for sandbox construction",
            score=quality.get("score"), findings=quality.get("findings", []),
        ))

    try:
        validate_task_spec(task.get("task_spec", {}))
    except (TaskSpecError, TypeError, ValueError) as exc:
        issues.append(_issue("TASK_SPEC_INVALID", "task_generation", str(exc)))

    missing_endpoints = missing_system_endpoints(
        task.get("requirements", {}).get("runtime_interface")
        if isinstance(task.get("requirements"), dict) else None
    )
    if missing_endpoints:
        issues.append(_issue(
            "TASK_RUNTIME_INTERFACE", "task_generation",
            "runtime interface is missing mandatory Trainer or system endpoints",
            missing=sorted(missing_endpoints),
        ))

    mode = task.get("environment_plan", {}).get("mode")
    if (mode == "stateful" and task.get("task_spec", {}).get("goal_contract", {}).get("allow_noop") is not True
            and TaskGenerationPipeline._has_conditional_noop_branch(
        task, task.get("public_input", {}) if isinstance(task.get("public_input"), dict) else {},
    )):
        issues.append(_issue(
            "STATEFUL_NOOP_BRANCH", "task_generation",
            "a valid no-write branch cannot satisfy a state-change-only goal contract",
        ))
    declared_modes = task.get("runtime_capabilities", {}).get("environment_modes", [])
    if isinstance(declared_modes, list) and declared_modes and mode not in declared_modes:
        issues.append(_issue(
            "ENVIRONMENT_MODE_UNDECLARED", "task_generation",
            "task environment mode is absent from its generation-time capability inventory",
            mode=mode, declared_modes=declared_modes,
        ))
    external_configured = bool(os.getenv("SANDBOX_EXTERNAL_CAPABILITY_URL", "").strip())
    fixture = os.getenv("SANDBOX_EXTERNAL_FIXTURES", "").strip()
    external_configured = external_configured or bool(fixture and Path(fixture).is_file())
    if mode == "external_capability" and not external_configured:
        issues.append(_issue(
            "EXTERNAL_CAPABILITY_UNAVAILABLE", "task_generation",
            "task requires an external capability but no provider or fixture is configured",
        ))

    manifest = task.get("artifacts", {}).get("data_manifest", {})
    preview_tables: list[dict[str, Any]] = []
    table_schemas: list[dict[str, Any]] = []
    if isinstance(manifest, dict) and manifest.get("root"):
        data_root = _manifest_root(root, manifest)
        if not data_root.is_dir():
            issues.append(_issue(
                "BUSINESS_DATA_MISSING", "task_generation", "declared business data root is missing",
                root=str(data_root),
            ))
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="envfactory-buildability-") as directory:
                    store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                    business_data = ManifestDataStore(manifest, data_root, store)
                    preview_tables = [
                        {**business_data.schemas.get(name, {}), "table_name": name, "rows": rows}
                        for name, rows in business_data.baseline.items()
                    ]
                    table_schemas = list(business_data.schemas.values())
            except (SandboxError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                issues.append(_issue(
                    "BUSINESS_DATA_INVALID", "task_generation",
                    "business data violates the shared persistence contract",
                    error=str(exc), root=str(data_root),
                ))

    if preview_tables and mode in {"reference_data", "stateful"}:
        try:
            preview = TaskGenerationPipeline._preview_success_tool_results(
                scenarios=task.get("acceptance_contract", {}).get("executable_scenarios", []),
                data_tables=preview_tables,
                tool_implementations=task.get("tool_implementations", []),
                environment_mode=mode,
                tools=task.get("tools", []),
                semantic_goal=task.get("task_spec", {}).get("goal_contract"),
            )
            issues.extend(_issue(
                issue["code"], "task_generation",
                "reward metric captures a positional row from a multirow tool result",
                **{key: value for key, value in issue.items() if key != "code"},
            ) for issue in ambiguous_metric_captures(task, preview))
        except (PipelineGenerationError, SandboxError, TypeError, ValueError) as exc:
            issues.append(_issue(
                "SUCCESS_TOOL_PREVIEW_INVALID", "task_generation",
                "declared success tools fail against persisted business data",
                error=str(exc),
            ))

    for index, spec in enumerate(task.get("metric_implementations", [])):
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            issues.append(_issue(
                "METRIC_IMPLEMENTATION_INVALID", "task_generation",
                f"metric implementation {index} has no valid path",
            ))
            continue
        try:
            DeclarativeMetricEvaluator.path_tokens(spec["path"])
        except (SandboxError, ValueError, SyntaxError) as exc:
            issues.append(_issue(
                "METRIC_PATH_UNSUPPORTED", "platform", str(exc),
                metric_id=spec.get("metric_id"), path=spec.get("path"),
            ))

    declared_tools = {
        item.get("function", {}).get("name") for item in task.get("tools", [])
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    implemented_tools = {
        item.get("tool_name") for item in task.get("tool_implementations", [])
        if isinstance(item, dict)
    }
    noise_tools = {
        item.get("name") for item in task.get("noise_tools", []) if isinstance(item, dict)
    }
    missing_handlers = sorted(declared_tools - implemented_tools - noise_tools)
    for spec in task.get("tool_implementations", []):
        if isinstance(spec, dict):
            missing_columns = TaskGenerationPipeline._missing_insert_columns(spec, table_schemas)
            if missing_columns:
                issues.append(_issue(
                    "INSERT_ROW_INCOMPLETE", "task_generation",
                    "declarative insert omits storage columns required by the shared data store",
                    tool_name=spec.get("tool_name"), table=spec.get("table"),
                    missing_columns=missing_columns,
                ))
            optional = TaskGenerationPipeline._optional_mutation_arguments(
                spec, task.get("tools", []),
            )
            if optional:
                issues.append(_issue(
                    "MUTATION_ARGUMENT_OPTIONAL", "task_generation",
                    "declarative mutation reads arguments not required by its tool schema",
                    tool_name=spec.get("tool_name"), optional_arguments=optional,
                ))
    business_tools = [
        item for item in task.get("tools", [])
        if isinstance(item, dict)
        and item.get("function", {}).get("name") not in noise_tools
    ]
    business_names = {
        item.get("function", {}).get("name") for item in business_tools
    }
    def contains_placeholder(value: Any) -> bool:
        if isinstance(value, str):
            return value.startswith("任务输入中的")
        if isinstance(value, list):
            return any(contains_placeholder(item) for item in value)
        if isinstance(value, dict):
            return any(contains_placeholder(item) for item in value.values())
        return False
    acceptance = task.get("acceptance_contract", {})
    if isinstance(acceptance, dict):
        for case in acceptance.get("tool_cases", []):
            if (isinstance(case, dict)
                    and case.get("tool_name") in business_names
                    and case.get("kind") == "schema_and_business_smoke"
                    and contains_placeholder(case.get("arguments_template"))):
                issues.append(_issue(
                    "TASK_ACCEPTANCE_FIXTURE", "task_generation",
                    "business tool smoke case contains unresolved placeholder values",
                    case_id=case.get("case_id"), tool_name=case.get("tool_name"),
                ))
        for probe in acceptance.get("argument_probes", []):
            if (isinstance(probe, dict)
                    and probe.get("tool_name") in business_names
                    and contains_placeholder(probe.get("arguments"))):
                issues.append(_issue(
                    "TASK_ACCEPTANCE_FIXTURE", "task_generation",
                    "business tool acceptance probe contains unresolved placeholder values",
                    probe_id=probe.get("probe_id"), tool_name=probe.get("tool_name"),
                ))
    issues.extend(_tool_chain_issues(task))
    issues.extend(reward_contract_issues(task))
    try:
        TaskGenerationPipeline._validate_business_tool_semantics(
            tools=business_tools,
            implementations=[
                item for item in task.get("tool_implementations", [])
                if isinstance(item, dict)
            ],
        )
    except PipelineGenerationError as exc:
        issues.append(_issue(
            "BUSINESS_TOOL_SEMANTICS_INVALID", "task_generation", str(exc),
        ))
    report = {
        "buildable": not issues,
        "failure_owner": issues[0]["owner"] if issues else None,
        "issues": issues,
        "task_score": quality,
        "environment_mode": mode,
        "custom_business_handlers": missing_handlers,
        "external_capability_configured": external_configured,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=8.0)
    args = parser.parse_args()
    root = args.root.resolve()
    report = assess(root, threshold=args.threshold)
    output = args.output or root / "buildability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["buildable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
