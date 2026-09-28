#!/usr/bin/env python3
"""Validate the module-level development DAG produced for one sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("development_plan must be an object")
    return value


def validate(plan: dict) -> list[str]:
    errors: list[str] = []
    nodes = plan.get("nodes")
    if not isinstance(nodes, list):
        return ["development_plan.nodes must be a list"]
    ids: list[str] = []
    by_id: dict[str, dict] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            errors.append(f"nodes[{index}].id must be a non-empty string")
            continue
        if node_id in by_id:
            errors.append(f"duplicate node id: {node_id}")
        ids.append(node_id)
        by_id[node_id] = node
        for field in ("goal", "inputs", "outputs", "validation"):
            if field not in node:
                errors.append(f"node {node_id} requires {field}")
        if not isinstance(node.get("inputs"), list):
            errors.append(f"node {node_id}.inputs must be a list")
        if not isinstance(node.get("outputs"), list):
            errors.append(f"node {node_id}.outputs must be a list")
        if not isinstance(node.get("validation"), list) or not node.get("validation"):
            errors.append(f"node {node_id}.validation must be a non-empty list")
        depends = node.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(item, str) for item in depends):
            errors.append(f"node {node_id}.depends_on must be a list of strings")
        for dependency in depends if isinstance(depends, list) else []:
            if dependency == node_id:
                errors.append(f"node {node_id} cannot depend on itself")

    # Check dependencies against the complete ID set and detect cycles.
    known = set(ids)
    graph = {node_id: set(node.get("depends_on", [])) for node_id, node in by_id.items()}
    for node_id, deps in graph.items():
        for dependency in deps:
            if dependency not in known:
                errors.append(f"node {node_id} depends on unknown node {dependency}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            errors.append(f"development_plan contains a dependency cycle at {node_id}")
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in graph.get(node_id, set()):
            if dependency in graph:
                visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--ids", action="store_true")
    args = parser.parse_args()
    try:
        plan = load(args.plan)
        errors = validate(plan)
    except Exception as exc:  # noqa: BLE001 - CLI should report malformed JSON
        print(f"development_plan validation: {exc}")
        return 1
    if errors:
        print("development_plan validation: " + "; ".join(errors))
        return 1
    print("development_plan validation: ok")
    if args.ids:
        for node in plan["nodes"]:
            print(node["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
