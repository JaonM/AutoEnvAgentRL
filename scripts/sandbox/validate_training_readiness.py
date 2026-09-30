#!/usr/bin/env python3
"""Fast hard gate for generated Agentic-RL environments."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {"ground_truth", "expected_tool_call", "hidden_state", "api_key", "trainer_token", "password"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        if (isinstance(value.get("events"), list) and "trace_hash" in value
                and "episode_id" in value):
            events = []
            for event in value["events"]:
                if not isinstance(event, dict):
                    events.append(event)
                    continue
                normalized = {key: canonical(item) for key, item in event.items()
                              if key != "timestamp"}
                payload = normalized.get("payload")
                if event.get("event") == "tool_call" and isinstance(payload, dict):
                    normalized["payload"] = {
                        key: item for key, item in payload.items()
                        if key not in {"tool_call_id", "request_id", "timestamp", "duration_ms"}
                    }
                events.append(normalized)
            return {key: (events if key == "events" else canonical(item))
                    for key, item in value.items() if key != "trace_hash"}
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def forbidden_paths(value: Any, path: str = "$") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key.casefold() in FORBIDDEN_PUBLIC_KEYS or any(token in key.casefold() for token in ("secret", "credential")):
                found.append(child)
            found.extend(forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_paths(item, f"{path}[{index}]"))
    return found


def reward_from_run(run: dict[str, Any]) -> float | None:
    for item in reversed(run.get("history", [])):
        if item.get("operation") == "reward" and isinstance(item.get("body"), dict):
            value = item["body"].get("reward")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and -1 <= value <= 1 and math.isfinite(value)):
                return float(value)
            return None
    return None


def import_app(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("generated_sandbox_app", root / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import generated app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_app"):
        raise RuntimeError("generated app.py must expose create_app()")
    return module.create_app()


def valid_delta_precondition(delta: Any, tables: dict[str, Any]) -> bool:
    """Validate typed changes against their scoped baseline; never search prose."""
    if not isinstance(delta, dict) or "before" not in delta or "after" not in delta:
        return False
    before, after = delta["before"], delta["after"]
    if type(before) is type(after) and before == after:
        return False
    table, where, field = delta.get("table"), delta.get("where"), delta.get("field")
    if not isinstance(table, str) or not isinstance(where, dict) or not where or not isinstance(field, str):
        return False
    rows = [row for row in tables.get(table, []) if all(
        key in row and type(row[key]) is type(value) and row[key] == value for key, value in where.items())]
    return bool(rows) and all(field in row and type(row[field]) is type(before) and row[field] == before for row in rows)


def preservation_probes(app, runner, scenario, goal, headers):
    """Run final-sandbox counterexamples while keeping the requested goal true."""
    from sandbox_runtime import BusinessGoalEvaluator
    app.handle("POST", "/v1/reset", {"episode_id": "preservation-baseline", "seed": 17}, headers)
    baseline = copy.deepcopy(app.business_snapshot())
    runner.run(scenario)
    success_state = copy.deepcopy(app.business_snapshot())
    results = []
    for table, rows in success_state.items():
        keys = goal.get("table_primary_keys", {}).get(table, [])
        if not keys:
            continue
        found = False
        for row in rows:
            if found:
                break
            for field, value in row.items():
                if field in keys or value is None:
                    continue
                changed = (not value if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else value + "__preservation_probe__" if isinstance(value, str) else None)
                if changed is None:
                    continue
                runner.run(scenario)
                try:
                    app.mutate_business_state({"table": table, "selector": {k: row[k] for k in keys}, "changes": {field: changed}})
                except Exception:
                    continue  # schema/constraint rejection is not reward evidence
                current = app.business_snapshot()
                if (not BusinessGoalEvaluator.evaluate(goal["row_predicates"], current, baseline)
                        or BusinessGoalEvaluator.preserves_unrelated(goal, baseline, current)):
                    continue
                status, payload, _ = app.handle("GET", "/v1/reward", headers=headers)
                reward = payload.get("reward") if isinstance(payload, dict) else None
                passed = status == 200 and isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward) and reward <= 0.2
                results.append({"table": table, "field": field, "goal_still_satisfied": True,
                                "unrelated_state_changed": True, "reward": reward, "passed": passed})
                found = True
                break
    runner.run(scenario)
    return {"status": "completed" if results else "not_applicable", "cases": results,
            "passed": all(case["passed"] for case in results)}


def validate(root: Path) -> dict[str, Any]:
    task = load(root / "task.json")
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    failures: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    task_spec = task.get("task_spec")
    if not isinstance(task_spec, dict) or task_spec.get("version") != "1.0":
        failures.append({"gate": "task_spec_integrity", "message": "missing or invalid task_spec IR"})
    else:
        environment = task_spec.get("environment_contract", {})
        goal = task_spec.get("goal_contract", {})
        archetype = environment.get("archetype") if isinstance(environment, dict) else None
        if not isinstance(archetype, str) or not archetype:
            failures.append({"gate": "task_spec_integrity", "message": "environment archetype is missing"})
        deltas = goal.get("expected_delta", []) if isinstance(goal, dict) else []
        if environment.get("mode") == "stateful" and not isinstance(deltas, list):
            failures.append({"gate": "state_causality", "message": "stateful goal has no expected_delta list"})
        manifest = task.get("artifacts", {}).get("data_manifest", {})
        baseline_values: list[Any] = []
        for table in manifest.get("tables", []) if isinstance(manifest, dict) else []:
            rows_file = table.get("rows_file") if isinstance(table, dict) else None
            if not isinstance(rows_file, str):
                continue
            manifest_root = manifest.get("root", "") if isinstance(manifest, dict) else ""
            rows_path = root / str(manifest_root) / rows_file
            if rows_path.is_file():
                for line in rows_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        baseline_values.append(json.loads(line))
        invalid_deltas = []
        baseline_by_table = {}
        for table in manifest.get("tables", []) if isinstance(manifest, dict) else []:
            rows_file = table.get("rows_file") if isinstance(table, dict) else None
            if rows_file:
                rows_path = root / str(manifest.get("root", "")) / rows_file
                if rows_path.is_file():
                    baseline_by_table[table.get("table_name")] = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        for delta in deltas if isinstance(deltas, list) else []:
            if not valid_delta_precondition(delta, baseline_by_table):
                invalid_deltas.append(delta)
        evidence["state_causality"] = {"archetype": archetype, "expected_delta": deltas, "invalid": invalid_deltas}
        if environment.get("mode") == "stateful" and not goal.get("row_predicates"):
            failures.append({"gate": "state_causality", "message": "stateful task requires executable row_predicates"})
        if invalid_deltas:
            failures.append({"gate": "state_causality", "message": "initial fixture does not satisfy expected delta preconditions", "invalid": invalid_deltas})
    kinds = {item.get("kind") for item in scenarios if isinstance(item, dict)}
    for required in ("goal_success", "goal_failure"):
        if required not in kinds:
            failures.append({"gate": "scenario_coverage", "message": f"missing {required} scenario"})
    if task.get("noise_tools") and "noise_selection" not in kinds:
        failures.append({"gate": "noise_tool_behavior", "message": "missing noise_selection scenario"})
    # Training-readiness must be deterministic and runnable offline. Generated
    # sandboxes expose an explicit contract-aligned evaluator mock for this
    # purpose; production runs can override the variable to exercise a real
    # evaluator endpoint.
    os.environ.setdefault("SANDBOX_EVALUATOR_MOCK", "1")
    app = import_app(root)
    from sandbox_runtime import AcceptanceScenarioRunner  # imported from generated sandbox
    from sandbox_runtime import BusinessGoalEvaluator
    token = os.environ.get("SANDBOX_TRAINER_API_KEY", "envfactory-readiness-key")
    os.environ["SANDBOX_TRAINER_API_KEY"] = token
    runner = AcceptanceScenarioRunner(app.handle, trainer_headers={"Authorization": f"Bearer {token}"},
                                     business_snapshot=getattr(app, "business_snapshot", None),
                                     mutate_business_state=getattr(app, "mutate_business_state", None))
    runs: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("kind") not in {"goal_success", "goal_failure", "noise_selection", "counterfactual"}:
            continue
        try:
            result = runner.run(scenario)
            runs.setdefault(scenario["kind"], []).append(result)
        except Exception as exc:  # noqa: BLE001 - evidence gate
            failures.append({"gate": "scenario_execution", "scenario_id": scenario.get("scenario_id"), "message": str(exc)})
    success = reward_from_run(runs.get("goal_success", [{}])[0]) if runs.get("goal_success") else None
    failure = reward_from_run(runs.get("goal_failure", [{}])[0]) if runs.get("goal_failure") else None
    evidence["rewards"] = {"success": success, "failure": failure}
    if success is None or failure is None:
        failures.append({"gate": "reward_separation", "message": "success/failure scenarios must call reward"})
    elif success <= failure or success - failure < 0.1:
        failures.append({"gate": "reward_separation", "message": "success reward is not sufficiently above failure reward", "success": success, "failure": failure})
    auth = {"Authorization": f"Bearer {token}"}
    app.handle("POST", "/v1/reset", {"episode_id": "readiness-leak", "seed": 17}, auth)
    status, observation, _ = app.handle("GET", "/v1/observation", headers=auth)
    leaks = forbidden_paths(observation)
    evidence["observation_scan"] = {"status": status, "forbidden_paths": leaks}
    if status != 200 or leaks:
        failures.append({"gate": "hidden_state_leakage", "message": "public observation leaks forbidden fields", "paths": leaks})
    if runs.get("goal_success"):
        scenario = next(item for item in scenarios if item.get("kind") == "goal_success")
        try:
            first = canonical(runner.run(scenario))
            second = canonical(runner.run(scenario))
            evidence["determinism"] = first == second
            if first != second:
                failures.append({"gate": "determinism", "message": "same structured trajectory is not deterministic"})
            goal = (task_spec or {}).get("goal_contract", {})
            if goal.get("row_predicates"):
                snapshot = getattr(app, "business_snapshot", None)
                if snapshot is None or not BusinessGoalEvaluator.evaluate(goal["row_predicates"], snapshot(), baseline_by_table):
                    failures.append({"gate": "state_causality", "message": "success rollout did not satisfy business postconditions"})
        except Exception as exc:
            failures.append({"gate": "determinism", "message": str(exc)})
    if runs.get("goal_success") and (task_spec or {}).get("goal_contract", {}).get("row_predicates"):
        scenario = next(item for item in scenarios if item.get("kind") == "goal_success")
        try:
            preservation = preservation_probes(app, runner, scenario, task_spec["goal_contract"], auth)
            evidence["unrelated_state_preservation"] = preservation
            if not preservation["passed"]:
                failures.append({"gate": "unrelated_state_preservation", "message": "unrelated state mutation retained reward", "evidence": preservation})
        except Exception as exc:
            failures.append({"gate": "unrelated_state_preservation", "message": str(exc)})
    # Exercise the public episode protocol and the shared state layer directly.
    # These checks produce machine-readable evidence for factory-level
    # production-prepared certification instead of inferring isolation from
    # source code or a successful acceptance run.
    try:
        episode_a, episode_b = "readiness-episode-a", "readiness-episode-b"
        headers_a = {**auth, "X-Episode-ID": episode_a}
        headers_b = {**auth, "X-Episode-ID": episode_b}
        status_a, _, _ = app.handle(
            "POST", "/v1/reset", {"episode_id": episode_a, "seed": 1701}, headers_a
        )
        baseline_a = canonical(app.business_snapshot())
        marker = "envfactory episode isolation marker"
        marker_status, _, _ = app.handle(
            "POST", "/v1/agent_response", {"content": marker}, headers_a
        )
        replay_status_a, replay_a, _ = app.handle("GET", "/v1/replay", headers=headers_a)
        repeated_status_a, repeated_a, _ = app.handle("GET", "/v1/replay", headers=headers_a)

        status_b, _, _ = app.handle(
            "POST", "/v1/reset", {"episode_id": episode_b, "seed": 1701}, headers_b
        )
        baseline_b = canonical(app.business_snapshot())
        replay_status_b, replay_b, _ = app.handle("GET", "/v1/replay", headers=headers_b)
        _, replay_a_after_b, _ = app.handle("GET", "/v1/replay", headers=headers_a)

        reset_status, _, _ = app.handle(
            "POST", "/v1/reset", {"episode_id": episode_a, "seed": 1701}, headers_a
        )
        reset_a = canonical(app.business_snapshot())
        _, replay_a_after_reset, _ = app.handle("GET", "/v1/replay", headers=headers_a)
        reset_reproducible = (
            status_a == status_b == reset_status == 200
            and baseline_a == baseline_b == reset_a
            and replay_a_after_reset.get("events") == []
        )
        episode_isolation = (
            marker_status == replay_status_a == replay_status_b == 200
            and replay_a.get("episode_id") == episode_a
            and replay_b.get("episode_id") == episode_b
            and replay_b.get("events") == []
            and any(
                event.get("event") == "agent_response"
                and event.get("payload", {}).get("content") == marker
                for event in replay_a_after_b.get("events", [])
            )
        )
        replay_consistent = (
            repeated_status_a == 200
            and canonical(replay_a) == canonical(repeated_a)
            and replay_a.get("trace_hash") == replay_a_after_b.get("trace_hash")
        )
        evidence["runtime_state"] = {
            "reset_reproducible": reset_reproducible,
            "episode_isolation": episode_isolation,
            "replay_consistent": replay_consistent,
            "episode_a_event_count": len(replay_a.get("events", [])),
            "episode_b_event_count": len(replay_b.get("events", [])),
        }
        for gate, passed, message in (
            ("reset_reproducibility", reset_reproducible, "same seed reset did not restore the baseline"),
            ("episode_isolation", episode_isolation, "episode state or replay crossed episode boundaries"),
            ("replay_consistency", replay_consistent, "repeated replay was not stable"),
        ):
            if not passed:
                failures.append({"gate": gate, "message": message})
    except Exception as exc:
        evidence["runtime_state"] = {
            "reset_reproducible": False,
            "episode_isolation": False,
            "replay_consistent": False,
            "error_type": type(exc).__name__,
        }
        failures.append({
            "gate": "runtime_state_integrity",
            "message": f"{type(exc).__name__}: {exc}",
        })
    hard_gates = sorted({failure["gate"] for failure in failures})
    return {
        "training_ready": not failures,
        "validation_mode": "offline_mock" if os.environ.get("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"} else "live",
        "live_rollout_verified": False,
        "hard_gates_passed": not failures,
        "failed_gates": hard_gates,
        "evidence": evidence,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    output = args.output or args.root / "training_readiness.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
