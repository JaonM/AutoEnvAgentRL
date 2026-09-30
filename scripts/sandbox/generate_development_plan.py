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
    if contract.get("artifacts", {}).get("generation_pipeline", {}).get("backend") == "code_agent":
        nodes.append({
            "id": "business_integration",
            "goal": (
                "Validate the agent-authored business contract against the actual scaffold behavior. "
                "This node may edit only tests/business_integration. Implementation extensions belong "
                "to the preceding task_handlers/metric_extensions nodes when those nodes exist. "
                "Compiled tools and metric scores are already implemented and cannot be replaced. "
                "Add executable tests for this business scenario under tests/business_integration: "
                "For model-based metrics, SANDBOX_EVALUATOR_MOCK performs exact reference-field matching only; "
                "it cannot grade paraphrases. Offline tests must use the reference text in those fields. "
                "Do not claim semantic equivalence from mock tests; real semantic calibration/live evidence is separate. "
                "Use the existing dependency-free application API: create_app(db_path=tmp_path / 'episode.sqlite3'), "
                "then status, body, headers = app.handle(method, path, json_body, request_headers). "
                "Use pytest tmp_path for a fresh file-backed database per test; SQLite ':memory:' does not persist "
                "across this runtime's separate connections. app.handle exercises routing, authentication, tools "
                "and reward directly. Do not add Flask/Werkzeug or invent another HTTP client for these tests. "
                "Read the trainer token from os.environ['SANDBOX_TRAINER_API_KEY'] after setting a fallback only when absent; "
                "construct Authorization from that actual value. Validation injects a different token, as Docker does. "
                "a real successful episode, an incorrect business outcome, and a meaningful data/argument "
                "counterfactual when the contract has relevant business data or tool arguments. "
                "For a stateless direct-response task with no tools or tables, instead test a changed-meaning "
                "answer and its affected reward component. Absence of business state in that route is expected, "
                "not a contract defect; do not invent tables or tool calls for the test. "
                "For a stateful data-dependent write, test a changed UPSTREAM input which changes the correct "
                "write value or decision, not only the destination field which the same write overwrites. "
                "Construct this as a new initial fixture: copy BUILD_CONTRACT.json and data/ into tmp_path, "
                "change the relevant copied input row, monkeypatch app.ROOT to that copied root BEFORE "
                "create_app, then reset and execute the tool chain. Keep the contract and reward rules unchanged. "
                "Assert the upstream tool returns the new input, the correct new write earns full reward, "
                "and the old write or old answer loses its affected reward component. "
                "Never alter delivery fixture files. This fixture-before-reset method changes initial conditions; "
                "mutating unrelated upstream rows after reset would instead count as episode side effects. "
                "For tests of within-episode business mutations, a positive counterfactual preserves the public target (such as the requested "
                "order ID) while changing relevant business data. Reset the episode BEFORE applying a "
                "business-state mutation; reset restores baseline data. Assert the mutation affected a row "
                "and the subsequent tool result contains the new value before testing reward. Also submit "
                "the stale answer and verify the affected component loses credit. Another order is a negative case unless "
                "the public task authorizes choosing it. Tests must call actual tools/reward, not assert metadata or copy reference "
                "assertions. For an incorrect outcome, assert the affected reward components; independent "
                "correct fields may retain partial credit. Derive any total from the declared metric weights, "
                "never assume every incorrect answer scores 0 or 0.2. Use pytest.approx for fractional "
                "reward totals; binary floating-point sums need not equal decimal literals exactly. Include the complete reward object in "
                "assertion failures so diagnosis can distinguish a rejected decision from other earned credit. "
                "Keep task.json, BUILD_CONTRACT.json and platform runtime immutable. "
                "Do not replace the shared declarative compiler with duplicated hardcoded handlers. "
                "If the contract is inconsistent, report the exact defect; do not weaken it."
            ),
            "depends_on": [node["id"] for node in nodes],
            "inputs": ["BUILD_CONTRACT.json", "task_impl.py", "app.py"],
            "outputs": ["tests/business_integration"],
            "validation": ["python3 -m pytest -q tests/business_integration"],
            "scope": {"tools": tool_names, "tables": tables, "archetype": archetype,
                      "reward_components": [{"id": metric["id"], "category": metric["category"],
                          "weight": metric["weight"],
                          "answer_fields": [target.get("key", target.get("label"))
                              for rule in contract.get("metric_implementations", [])
                              if rule.get("metric_id") == metric["id"]
                              and rule.get("operator") in {"value_targets", "numeric_targets"}
                              for target in rule.get("expected", {}).get("targets", [])]}
                          for metric in contract.get("metrics", [])]},
        })
    return {
        "version": "2.0",
        "authority": "env_factory_outer_workflow",
        "environment_archetype": archetype,
        "task_implementation_editable": bool(custom_tools or custom_metrics),
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
