#!/usr/bin/env python3
"""Static guard against task-specific demo implementations.

This is intentionally a narrow structural gate.  Business handlers may be
task-specific, but evaluator selection, reward aggregation, and user-script
execution must be driven by BUILD_CONTRACT and runtime artifacts rather than
by fixed metric ids, one session, or a fixed turn count.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def trace_precondition_functions(source: str) -> list[str]:
    """Find task functions that turn historical tool order into a hard precondition."""
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        name: {
            call.func.id for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id in functions
        }
        for name, node in functions.items()
    }
    trace = {
        name: any(
            isinstance(node, ast.Attribute) and node.attr in {"replay", "events"}
            for node in ast.walk(function)
        )
        for name, function in functions.items()
    }
    precondition = {
        name: any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "PRECONDITION_FAILED"
            for node in ast.walk(function)
        )
        for name, function in functions.items()
    }
    changed = True
    while changed:
        changed = False
        for name, dependencies in calls.items():
            next_trace = trace[name] or any(trace[dependency] for dependency in dependencies)
            next_precondition = precondition[name] or any(
                precondition[dependency] for dependency in dependencies
            )
            if next_trace != trace[name] or next_precondition != precondition[name]:
                trace[name], precondition[name] = next_trace, next_precondition
                changed = True
    return sorted(
        name for name in functions if trace[name] and precondition[name]
    )


def hardcoded_metric_scores(source: str, metric_ids: set[str]) -> list[str]:
    """Flag explicit metric bindings inside scoring code, not business labels.

    This structural check supplements immutable runtime ownership and behavioral
    reward tests; it is not a general proof that authored code is correct.
    """
    found = set()
    def visit(node: ast.AST, scoring: bool = False) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scoring = scoring or bool(re.search(r"score|reward|evaluat", node.name, re.I))
        if scoring:
            keys = []
            if isinstance(node, ast.Dict):
                keys = node.keys
            elif isinstance(node, ast.Subscript):
                keys = [node.slice]
            elif isinstance(node, ast.Compare):
                keys = [node.left, *node.comparators]
            for key in keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in metric_ids:
                    found.add(key.value)
        for child in ast.iter_child_nodes(node):
            visit(child, scoring)
    visit(ast.parse(source))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = root / "app.py"
    contract_path = root / "BUILD_CONTRACT.json"
    if not source_path.is_file() or not contract_path.is_file():
        fail("runtime genericity check requires app.py and BUILD_CONTRACT.json")

    # Generated sandboxes are intentionally modular.  Inspect the production
    # runtime as a whole instead of requiring every implementation detail to
    # live in app.py (an earlier check incorrectly rejected good modular
    # implementations).
    production_names = (
        "app.py", "task_impl.py", "tool_registry.py", "reward_evaluator.py", "user_simulator.py"
    )
    sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in production_names if (root / name).is_file()
    }
    source = "\n".join(sources.values())
    task_source = sources.get("task_impl.py", "")
    try:
        trace_preconditions = trace_precondition_functions(task_source)
    except SyntaxError as exc:
        fail(f"task_impl.py cannot be parsed: {exc}")
    if trace_preconditions:
        fail(
            "task business handlers must not derive PRECONDITION_FAILED from replay order; "
            f"shared reward causality owns DAG sequencing: {trace_preconditions}"
        )
    shared_source_path = root / "sandbox_runtime.py"
    shared_source = shared_source_path.read_text(encoding="utf-8") if shared_source_path.is_file() else ""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    template = None
    if contract.get("task_spec"):
        import importlib.util
        template_path = Path(__file__).with_name("generate_sandbox_scaffold.py")
        module_spec = importlib.util.spec_from_file_location("scaffold_authority", template_path)
        template = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(template)
        if source_path.read_text(encoding="utf-8") != template.APP_SOURCE:
            fail("platform app.py differs from the EnvFactory scaffold; regenerate it before validation")
        platform = Path(__file__).resolve().parents[2] / "src" / "env_factory"
        for name in ("sandbox_runtime.py", "runtime_llm.py"):
            if not (root / name).is_file() or (root / name).read_bytes() != (platform / name).read_bytes():
                fail(f"platform runtime has changed: {name}")
    interface = contract.get("requirements", {}).get("runtime_interface", {})
    shared_runtime = interface.get("shared_runtime", {}) if isinstance(interface, dict) else {}
    required_components = shared_runtime.get("required_components", []) if isinstance(shared_runtime, dict) else []
    if isinstance(required_components, list):
        missing_components = [
            component for component in required_components
            if isinstance(component, str) and component not in source and component not in shared_source
        ]
        if missing_components:
            fail(f"runtime does not provide required shared components: {missing_components}")
    metrics = contract.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        fail("BUILD_CONTRACT.metrics must be a non-empty list")

    # The runtime must load the metric/evaluator definitions instead of
    # re-encoding the generated metric ids in Python.
    if not re.search(
        r"(?:CONTRACT|contract|self\.contract)\s*(?:\[\s*['\"]metrics['\"]\s*\]|\.get\(\s*['\"]metrics['\"])",
        source,
    ):
        fail("reward evaluator must iterate CONTRACT.metrics")
    if "DeclarativeMetricEvaluator" not in source and not re.search(r"metric\s*(?:\.|\[\s*['\"]evaluator['\"]\s*\])", source):
        fail("reward evaluator must consume metric.evaluator from the contract")
    evaluator_boundary = (
        ("RuntimeLLMClient" in source and "json_chat" in source)
        or ("ContractEvaluatorRuntime" in source and "RuntimeLLMClient" in shared_source and "json_chat" in shared_source)
    )
    if not evaluator_boundary:
        fail("runtime must use RuntimeLLMClient for external evaluator/user simulation")

    # Exact platform templates already passed the immutable-source check.
    # Their generic keys (e.g. "event") may coincide with any authored metric ID.
    trusted_templates = {"app.py": template.APP_SOURCE, "task_impl.py": template.TASK_IMPL_SOURCE} if template else {}
    metric_sources = {name: text for name, text in sources.items()
                      if text != trusted_templates.get(name)}
    metric_ids = [item.get("id") for item in metrics if isinstance(item, dict)]
    for text in metric_sources.values():
        for metric_id in hardcoded_metric_scores(text, set(metric_ids)):
            fail(f"runtime hard-codes generated metric id: {metric_id}")

    # The simulator chooses a deterministic seeded profile/FSM branch and may
    # not replace the FSM's termination decision with a fixed turn cutoff.
    if re.search(r"turns\s*>=\s*\d+", source) or re.search(r"turn_count\s*>=\s*\d+", source):
        fail("User Simulator termination must consume script/LLM should_end, not a fixed turn cutoff")
    if "ContractUserSimulator" not in source and ("class UserSimulator" not in source or not all(
        token in source for token in ("profiles", "scripts")
    )):
        fail("User Simulator must load profiles and scripts from its manifest")
    if re.search(r"ContractUserSimulator\s*\([^)]*renderer\s*=", source, re.S):
        fail("ContractUserSimulator must use its default external-LLM renderer; task-specific renderer injection is forbidden")

    # The generic runtime must expose a contract-driven tool registry and the
    # reward endpoint, while keeping reward calculation out of tool execution.
    shared_composition = all(token in source for token in ("ContractToolRegistry", "ContractRewardAggregator", "SandboxApplication"))
    legacy_composition = "class ToolRegistry" in source and "def execute" in source and "class RewardEvaluator" in source
    if not shared_composition and not legacy_composition:
        fail("runtime must contain a tool registry, tool executor, and reward endpoint")
    execute_start = source.find("def execute")
    reward_start = source.find("class RewardEvaluator")
    if execute_start >= 0 and reward_start > execute_start:
        execute_source = source[execute_start:reward_start]
        if "reward(" in execute_source or "_process_scores" in execute_source:
            fail("tool execution must not calculate reward")

    print("runtime genericity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
