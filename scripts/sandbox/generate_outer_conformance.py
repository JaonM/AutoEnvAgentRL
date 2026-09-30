#!/usr/bin/env python3
"""Generate and/or run EnvFactory-owned sandbox conformance checks.

The sandbox implementation is untrusted.  This checker is derived from the
task contract after the Code Agent has finished, so an implementation cannot
make its own acceptance.sh authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fail(message: str) -> None:
    raise SystemExit(message)


def load(root: Path, name: str) -> Any:
    path = root / name
    if not path.is_file():
        fail(f"missing {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid {name}: {exc}")


def check_schema(schema: Any, location: str, *, require_description: bool = True) -> None:
    if not isinstance(schema, dict):
        fail(f"{location} must be an object")
    if require_description and (not isinstance(schema.get("description"), str) or not schema["description"].strip()):
        fail(f"{location} requires description")
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            fail(f"{location}.properties must be an object")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, (bool, dict)):
            fail(f"{location}.additionalProperties must be boolean or a schema")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            fail(f"{location}.required is invalid")
        for name, child in properties.items():
            check_schema(child, f"{location}.properties.{name}")
        if isinstance(additional, dict):
            check_schema(additional, f"{location}.additionalProperties", require_description=False)
    elif schema.get("type") == "array":
        if "items" not in schema:
            fail(f"{location}.items is required for arrays")
        check_schema(schema["items"], f"{location}.items")


def check_tools(task: dict[str, Any], tools: Any) -> list[str]:
    declared = task.get("tools")
    if not isinstance(tools, list):
        fail("tools.json must be a top-level array")
    if isinstance(declared, list) and tools != declared:
        fail("tools.json differs from task.json.tools")
    names: list[str] = []
    for index, item in enumerate(tools):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(item, dict) or item.get("type") != "function" or not isinstance(function, dict):
            fail(f"tools[{index}] is not an OpenAI function tool")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            fail(f"invalid or duplicate tool name: {name!r}")
        if not isinstance(function.get("description"), str) or not function["description"].strip():
            fail(f"tool {name} requires description")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            fail(f"tool {name}.parameters must be an object schema")
        # JSON Schema/OpenAI does not require a description on the root
        # parameters object; nested properties and array item schemas do.
        check_schema(parameters, f"tool {name}.parameters", require_description=False)
        names.append(name)
    return names


def check_contract(task: dict[str, Any], contract: dict[str, Any], names: list[str]) -> None:
    expected = {key: value for key, value in task.items() if key != "actions"}
    if contract != expected:
        fail("BUILD_CONTRACT.json is not task.json minus actions")
    interface = contract.get("requirements", {}).get("runtime_interface")
    if not isinstance(interface, dict) or interface.get("protocol") != "http":
        fail("runtime_interface must declare http")
    endpoints = interface.get("endpoints")
    if not isinstance(endpoints, list):
        fail("runtime_interface.endpoints must be a list")
    endpoint_keys = {(e.get("name"), e.get("method"), e.get("path")) for e in endpoints if isinstance(e, dict)}
    required = {
        ("health", "GET", "/health"), ("reset", "POST", "/v1/reset"),
        ("observation", "GET", "/v1/observation"), ("tools", "GET", "/v1/tools"),
        ("state", "GET", "/v1/state"),
        ("user_simulator", "POST", "/v1/user_simulator"), ("reward", "GET", "/v1/reward"),
        ("replay", "GET", "/v1/replay"),
    }
    if not required <= endpoint_keys:
        fail("runtime_interface is missing required endpoints")
    declared = [e.get("name") for e in endpoints if isinstance(e, dict) and e.get("kind") == "llm_tool"]
    if declared != names or interface.get("llm_tools") != names:
        fail("runtime_interface tool list is inconsistent")
    for name in ("reward", "user_simulator"):
        matches = [e for e in endpoints if isinstance(e, dict) and e.get("name") == name]
        if len(matches) != 1 or matches[0].get("access") != "rl_trainer_only":
            fail(f"{name} must be trainer-only")
    replay = [e for e in endpoints if isinstance(e, dict) and e.get("name") == "replay"]
    if len(replay) != 1 or replay[0].get("access") != "rl_trainer_only":
        fail("replay must be trainer-only")
    security = interface.get("security", {})
    trainer = security.get("trainer", {}) if isinstance(security, dict) else {}
    if trainer.get("scheme") != "bearer" or trainer.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
        fail("runtime_interface trainer bearer authentication is missing")
    episode = interface.get("episode", {})
    if episode.get("isolation") != "per_episode" or episode.get("reset_accepts_seed") is not True or episode.get("deterministic_replay") is not True:
        fail("runtime_interface episode isolation/replay contract is missing")
    llm_runtime = interface.get("llm_runtime", {})
    if llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
        fail("runtime_interface external LLM credential boundary is missing")
    evaluator_runtime = interface.get("evaluator_runtime", {})
    if evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
        fail("runtime_interface evaluator mock contract is missing")
    mutation = interface.get("mutation_testing", {})
    if mutation.get("environment_variable") != "SANDBOX_MUTATION_MODE" \
            or not isinstance(mutation.get("modes"), list) or not mutation["modes"] \
            or mutation.get("production_default") != "disabled":
        fail("runtime_interface mutation testing contract is missing")
    launcher = interface.get("launcher", {})
    if launcher.get("command") != ["python", "app.py", "--port", "{port}"]:
        fail("runtime_interface launcher contract is missing")
    errors = interface.get("errors", {})
    if errors.get("content_type") != "application/json" or not isinstance(errors.get("schema"), dict):
        fail("runtime_interface error protocol is missing")
    observability = interface.get("observability", {})
    if observability.get("request_id_header") != "X-Request-ID" or observability.get("credential_redaction") is not True:
        fail("runtime_interface observability contract is missing")


def check_rewards(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = task.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        fail("metrics must be a non-empty list")
    key_steps = task.get("reward_key_steps")
    if not isinstance(key_steps, list):
        fail("reward_key_steps must be a list")
    actions = {a.get("name") for a in task.get("actions", []) if isinstance(a, dict)}
    step_actions = {s.get("action_name") for s in key_steps if isinstance(s, dict)}
    if not step_actions <= actions:
        fail("reward_key_steps contains unknown actions")
    step_ids = [s.get("step_id") for s in key_steps if isinstance(s, dict)]
    if len(step_ids) != len(key_steps) or len(set(step_ids)) != len(step_ids):
        fail("reward_key_steps contains duplicate or missing step IDs")
    for step in key_steps:
        if not step.get("step_id") or not isinstance(step.get("rationale"), str) or not isinstance(step.get("required_for_goal"), bool):
            fail("invalid reward_key_steps entry")
    positive = [m for m in metrics if m.get("category") in {"process", "outcome"}]
    negative = [m for m in metrics if m.get("category") == "penalty"]
    for group, expected in ((positive, 1.0), (negative, 1.0)):
        if group and not math.isclose(sum(float(m.get("weight", 0)) for m in group), expected, abs_tol=1e-6):
            fail("metric weights are not normalized by sign group")
    for metric in metrics:
        evaluator = metric.get("evaluator")
        if not isinstance(evaluator, dict) or not isinstance(evaluator.get("score_mapping"), dict):
            fail(f"metric {metric.get('id')} lacks executable evaluator")
        if metric.get("category") == "process":
            compiled = (
                metric.get("type") == "rule-based"
                and evaluator.get("kind") == "trajectory_rule"
                and evaluator.get("source") == "runtime_rule"
                and set(evaluator.get("score_mapping", {})) == {"pass", "fail"}
            )
            provisional = (
                metric.get("type") == "hybrid"
                and evaluator.get("kind") == "hybrid_tool_call"
                and evaluator.get("source") == "external_llm"
            )
            if not (compiled or provisional):
                fail(f"process metric {metric.get('id')} lacks a compiled or provisional evaluator")
            if metric.get("target_action") not in step_actions:
                fail(f"process metric {metric.get('id')} targets a non-key action")
        if metric.get("category") == "penalty" and any(float(v) > 0 for v in evaluator["score_mapping"].values()):
            fail(f"penalty metric {metric.get('id')} has positive score")
    formula = task.get("reward_formula")
    if not isinstance(formula, dict) or formula.get("score_range") != [-1, 1]:
        fail("reward_formula must be normalized to [-1,1]")
    return key_steps, metrics


def check_acceptance_contract(task: dict[str, Any], names: list[str], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    contract = task.get("acceptance_contract")
    if not isinstance(contract, dict) or contract.get("authority") != "env_factory_outer_workflow":
        fail("acceptance_contract is missing or not EnvFactory-owned")
    for key in ("fixtures", "scenarios", "tool_cases", "invariants", "mutations", "reward_cases", "mutation_tests"):
        if not isinstance(contract.get(key), (list, dict)):
            fail(f"acceptance_contract.{key} is invalid")
    if not contract["scenarios"] or (names and not contract["tool_cases"]) or not contract["reward_cases"]:
        fail("acceptance_contract requires scenarios, tool_cases and reward_cases")
    metric_ids = {metric.get("id") for metric in metrics}
    for case in contract["tool_cases"]:
        if not isinstance(case, dict) or not case.get("case_id") or case.get("tool_name") not in names or not isinstance(case.get("expected"), dict):
            fail("acceptance_contract references unknown tool")
    for case in contract["reward_cases"]:
        if not isinstance(case, dict) or not case.get("case_id") or case.get("metric_id") not in metric_ids or not isinstance(case.get("expected_score"), (int, float)):
            fail("acceptance_contract references unknown metric")
    for scenario in contract["scenarios"]:
        if not isinstance(scenario, dict) or not scenario.get("scenario_id") or not scenario.get("kind") or not isinstance(scenario.get("steps"), list):
            fail("acceptance_contract scenario is incomplete")
    if not contract["invariants"] or not contract["mutation_tests"]:
        fail("acceptance_contract requires invariants and mutation_tests")
    executable = contract.get("executable_scenarios", [])
    if executable:
        if not isinstance(executable, list):
            fail("acceptance_contract.executable_scenarios must be a list")
        allowed = {"reset", "tool_call", "agent_response", "observation", "reward", "replay", "business_snapshot", "mutate_business_state"}
        for scenario in executable:
            if not isinstance(scenario, dict) or not scenario.get("scenario_id") or not isinstance(scenario.get("steps"), list):
                fail("executable acceptance scenario is invalid")
            for step in scenario["steps"]:
                if not isinstance(step, dict) or step.get("operation") not in allowed:
                    fail("executable acceptance step is invalid")
                if step.get("operation") == "tool_call" and step.get("tool_name") not in names:
                    fail("executable acceptance step references unknown tool")
    return contract


def reference_reward(task: dict[str, Any], scores: dict[str, Any]) -> float:
    total = 0.0
    for metric in task.get("metrics", []):
        value = float(scores.get(metric["id"], 0.0))
        low, high = metric.get("score_range", [0, 1])
        if not low <= value <= high:
            fail(f"runtime reward component {metric['id']} outside score range")
        total += float(metric["weight"]) * value
    return max(-1.0, min(1.0, total))


def schema_example(schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if kind == "object":
        return {name: schema_example(schema["properties"][name]) for name in schema.get("required", [])}
    if kind == "array":
        return [schema_example(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind in {"integer", "number"}:
        return max(schema.get("minimum", 0), schema.get("exclusiveMinimum", -1) + 1)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return "x" * max(1, schema.get("minLength", 1))


def invalid_schema_cases(task: dict[str, Any]) -> list[dict[str, Any]]:
    from env_factory.sandbox_runtime import validate_json_schema, SandboxError
    cases = []
    acceptance = task.get("acceptance_contract", {})
    fixtures = acceptance.get("argument_probes", []) + acceptance.get("tool_cases", [])
    for tool in task.get("tools", []):
        function = tool["function"]
        name, schema = function["name"], function["parameters"]
        if schema.get("required"):
            cases.append({"tool": name, "case": "missing_required_arguments", "request": {}, "error_fragment": "missing required properties"})
        if schema.get("additionalProperties") is not False:
            continue
        candidates = [case.get("arguments", case.get("arguments_template")) for case in fixtures if case.get("tool_name") == name]
        candidates.append(schema_example(schema))
        valid = None
        for candidate in candidates:
            try:
                validate_json_schema(schema, candidate)
                valid = candidate
                break
            except SandboxError:
                continue
        if valid is None:
            fail(f"no schema-valid fixture for additionalProperties probe: {name}")
        key = "__outer_invalid_property__"
        while key in schema.get("properties", {}):
            key += "_"
        cases.append({"tool": name, "case": "unexpected_property", "request": {**valid, key: True}, "error_fragment": "unexpected properties"})
    return cases


def build_manifest(task: dict[str, Any], names: list[str], metrics: list[dict[str, Any]], key_steps: list[dict[str, Any]], acceptance: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority": "env_factory_outer_workflow",
        "checks": ["contract_projection", "openai_tool_schema", "runtime_http_contract", "key_steps", "metric_evaluators", "reward_formula", "business_scenarios", "counterfactual_rewards", "mutation_tests"],
        "tool_cases": invalid_schema_cases(task),
        "reward_cases": [
            {"metric_id": metric.get("id"), "case": "evaluator_positive_and_negative_paths_required", "category": metric.get("category"), "evaluator": metric.get("evaluator", {}).get("kind")}
            for metric in metrics
        ],
        "key_steps": [step.get("step_id") for step in key_steps],
        "business_scenarios": acceptance["scenarios"],
        "executable_scenarios": acceptance.get("executable_scenarios", []),
        "argument_probes": acceptance.get("argument_probes", []),
        "invariants": acceptance["invariants"],
        "mutations": acceptance["mutations"],
        "mutation_tests": acceptance["mutation_tests"],
        "runtime_required": True,
        "note": "The implementation acceptance.sh is supplementary; these checks are regenerated outside the sandbox.",
    }


def file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("task.json", "BUILD_CONTRACT.json", "tools.json", "runtime_llm.py", "sandbox_runtime.py"):
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def runtime_smoke(root: Path, names: list[str], base_url: str) -> None:
    """Run contract-level HTTP checks against an already running sandbox."""
    task = load(root, "task.json")
    key = __import__("os").environ.get("SANDBOX_TRAINER_API_KEY", "")
    base_url = base_url.rstrip("/")

    def call(method: str, path: str, body: Any = None, auth: bool = False) -> tuple[int, Any]:
        headers = {"Accept": "application/json"}
        if auth and key:
            headers["Authorization"] = f"Bearer {key}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            with urlopen(Request(base_url + path, data=data, headers=headers, method=method), timeout=10) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw) if raw else None
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except (ValueError, UnicodeError):
                payload = None
            return exc.code, payload
        except (URLError, TimeoutError, OSError) as exc:
            raise SystemExit(75) from exc

    status, _ = call("GET", "/health")
    if not 200 <= status < 300:
        fail(f"health check failed: HTTP {status}")
    status, payload = call("GET", "/v1/tools")
    if not 200 <= status < 300:
        fail(f"tool discovery failed: HTTP {status}")
    if isinstance(payload, dict):
        discovered = payload.get("tools", payload.get("data", []))
        if isinstance(discovered, list):
            discovered_names = [x.get("function", {}).get("name") for x in discovered if isinstance(x, dict)]
            if discovered_names and discovered_names != names:
                fail("/v1/tools differs from tools.json")
    if not key:
        fail("SANDBOX_TRAINER_API_KEY is required for dynamic trainer checks")
    status, _ = call("GET", "/v1/reward")
    if status not in {401, 403}:
        fail(f"unauthorized reward request must be rejected, got HTTP {status}")
    status, reset_payload = call("POST", "/v1/reset", {"episode_id": "outer-a", "seed": 17}, auth=True)
    if not 200 <= status < 300:
        fail(f"reset failed: HTTP {status}")
    status, _ = call("POST", "/v1/reset", {"episode_id": "outer-b", "seed": 17}, auth=True)
    if not 200 <= status < 300:
        fail(f"second reset failed: HTTP {status}")
    status, _ = call("GET", "/v1/observation", auth=True)
    if not 200 <= status < 300:
        fail(f"observation failed: HTTP {status}")
    status, state_payload = call("GET", "/v1/state", auth=True)
    if not 200 <= status < 300 or not isinstance(state_payload, dict) \
            or not isinstance(state_payload.get("business_state"), dict):
        fail(f"trainer state snapshot failed: HTTP {status}")
    status, replay_payload = call("GET", "/v1/replay", auth=True)
    if not 200 <= status < 300:
        fail(f"replay failed: HTTP {status}")
    if isinstance(replay_payload, dict) and replay_payload.get("episode_id") not in {None, "outer-b"}:
        fail("reset did not isolate the active episode")
    acceptance = task.get("acceptance_contract", {})
    for case in acceptance.get("tool_cases", []):
        body = case.get("arguments", case.get("arguments_template", {}))
        status, payload = call("POST", f"/v1/tools/{case['tool_name']}", body)
        expected = case.get("expected", {})
        if case.get("kind") == "invalid_input":
            if status not in set(expected.get("status_class", [400, 422])):
                fail(f"business invalid-input case failed: {case['case_id']} HTTP {status}")
        elif status >= 500:
            fail(f"business smoke case failed: {case['case_id']} HTTP {status}")
        if isinstance(payload, dict) and any(key in payload for key in expected.get("must_not_return", [])):
            fail(f"tool case {case['case_id']} returned observation/reward")
    status, reward_payload = call("GET", "/v1/reward", auth=True)
    if not 200 <= status < 300 or not isinstance(reward_payload, dict):
        fail(f"authorized reward failed: HTTP {status}")
    components = reward_payload.get("components")
    if isinstance(components, dict) and "reward" in reward_payload:
        expected = reference_reward(task, components)
        if abs(float(reward_payload["reward"]) - expected) > 1e-6:
            fail("runtime reward does not match independent reference formula")
    for case in invalid_schema_cases(task):
        status, payload = call("POST", f"/v1/tools/{case['tool']}", case["request"], auth=False)
        if status < 400 or status >= 500:
            fail(f"invalid request for {case['tool']} was not rejected with 4xx: HTTP {status}")
        message = payload.get("error", {}).get("message", "") if isinstance(payload, dict) else ""
        if case["error_fragment"] not in message:
            fail(f"schema probe {case['case']} failed for the wrong reason: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--base-url", help="对已运行沙箱执行动态 HTTP conformance smoke checks")
    args = parser.parse_args()
    task = load(args.root, "task.json")
    contract = load(args.root, "BUILD_CONTRACT.json")
    tools = load(args.root, "tools.json")
    if not isinstance(task, dict) or not isinstance(contract, dict):
        fail("task.json and BUILD_CONTRACT.json must be objects")
    names = check_tools(task, tools)
    check_contract(task, contract, names)
    key_steps, metrics = check_rewards(task)
    acceptance = check_acceptance_contract(task, names, metrics)
    manifest = build_manifest(task, names, metrics, key_steps, acceptance)
    manifest["artifact_hashes"] = file_hashes(args.root)
    if args.base_url:
        runtime_smoke(args.root, names, args.base_url)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "tool_cases.json").write_text(json.dumps(manifest["tool_cases"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "reward_cases.json").write_text(json.dumps(manifest["reward_cases"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "business_scenarios.json").write_text(json.dumps(manifest["business_scenarios"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "executable_scenarios.json").write_text(json.dumps(manifest["executable_scenarios"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "mutation_cases.json").write_text(json.dumps({"mutations": manifest["mutations"], "tests": manifest["mutation_tests"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "baseline.json").write_text(json.dumps({"artifact_hashes": manifest["artifact_hashes"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("outer conformance: ok" if args.check else f"outer conformance plan generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
