#!/usr/bin/env python3
"""Generate the stable module DAG used to build a task sandbox.

The platform architecture is owned by EnvFactory.  A code model should fill
task-specific behavior, not rediscover the same architecture for every task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("BUILD_CONTRACT.json must be an object")
    return value


def build_plan(contract: dict[str, Any]) -> dict[str, Any]:
    tools = contract.get("tools", [])
    tool_names = [
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]
    manifest = contract.get("artifacts", {}).get("data_manifest", {})
    tables = [
        item.get("table_name")
        for item in manifest.get("tables", [])
        if isinstance(item, dict) and item.get("table_name")
    ] if isinstance(manifest, dict) else []
    metrics = [
        item.get("id") for item in contract.get("metrics", [])
        if isinstance(item, dict) and item.get("id")
    ]
    task_spec = contract.get("task_spec", {})
    environment = task_spec.get("environment_contract", {}) if isinstance(task_spec, dict) else {}
    archetype = environment.get("archetype", "legacy") if isinstance(environment, dict) else "legacy"
    implementations = {
        item.get("tool_name") for item in contract.get("tool_implementations", [])
        if isinstance(item, dict) and item.get("tool_name")
    }
    noise = {
        item.get("name") for item in contract.get("noise_tools", [])
        if isinstance(item, dict) and item.get("name")
    }
    custom_tools = [name for name in tool_names if name not in noise and name not in implementations]
    implemented_metrics = {
        item.get("metric_id") for item in contract.get("metric_implementations", [])
        if isinstance(item, dict) and item.get("metric_id")
    }
    implemented_metrics.update(
        item.get("id") for item in contract.get("metrics", [])
        if isinstance(item, dict)
        and isinstance(item.get("evaluator"), dict)
        and item["evaluator"].get("kind") in {"external_llm_judge", "hybrid_outcome"}
    )
    custom_metrics = [name for name in metrics if name not in implemented_metrics]
    nodes: list[dict[str, Any]] = []
    if custom_tools:
        nodes.append({
            "id": "task_handlers",
            "goal": "Implement only business handlers not covered by the declarative tool compiler.",
            "depends_on": [],
            "inputs": ["BUILD_CONTRACT.json.task_spec.tool_contracts", "task_impl.py"],
            "outputs": ["task_impl.py business handlers", "tests/tools"],
            "validation": ["python3 -m pytest -q tests/tools"],
            "scope": {"tools": custom_tools, "tables": tables, "archetype": archetype},
        })
    if custom_metrics:
        nodes.append({
            "id": "metric_extensions",
            "goal": "Implement only metrics not covered by DeclarativeMetricEvaluator; do not alter aggregation or UserSimulator.",
            "depends_on": ["task_handlers"] if custom_tools else [],
            "inputs": ["BUILD_CONTRACT.json.task_spec.reward_contract", "task_impl.py"],
            "outputs": ["task_impl.py custom metric scores", "tests/reward"],
            "validation": ["python3 -m pytest -q tests/reward"],
            "scope": {"metrics": custom_metrics, "archetype": archetype},
        })
    return {
        "version": "2.0",
        "authority": "env_factory_outer_workflow",
        "environment_archetype": archetype,
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(_load(args.contract))
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
