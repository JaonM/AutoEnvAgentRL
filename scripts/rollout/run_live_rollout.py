"""Bounded real-model trajectories; Agent sees public inputs, never acceptance answers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid
from env_factory.llm import LLMClient
from env_factory.evidence.data_governance import provider_identity
from env_factory.evidence.model_roles import resolve_model_roles
from env_factory.evidence.material_artifacts import portable_artifact_digest
from env_factory.runtime_llm import RuntimeLLMConfig
from env_factory.sandbox_http import HTTPSandboxClient, evaluator_http_timeout
from env_factory.sandbox_runtime import BusinessGoalEvaluator


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def episode(app, task, client, seed, max_steps):
    headers = {"Authorization": "Bearer " + os.environ["SANDBOX_TRAINER_API_KEY"],
               "X-Episode-ID": f"live-{seed}"}
    trace, transitions, usage = [], [], []

    def request(method, path, body=None):
        status, value, _ = app.handle(method, path, body, headers)
        trace.append({"method": method, "path": path, "status": status, "body": body, "result": value})
        if status >= 500 or (status >= 400 and not path.startswith("/v1/tools/")):
            raise RuntimeError(f"runtime endpoint failed: {path} status={status}")
        return status, value

    request("POST", "/v1/reset", {"episode_id": headers["X-Episode-ID"], "seed": seed})
    baseline_payload = request("GET", "/v1/state")[1]
    baseline = baseline_payload.get("business_state")
    if not isinstance(baseline, dict):
        raise RuntimeError("trainer state endpoint returned an invalid baseline")
    initial_reward = request("GET", "/v1/reward")[1]["reward"]
    current_reward = initial_reward
    tools = request("GET", "/v1/tools")[1]["tools"]
    observation = request("GET", "/v1/observation")[1]
    names = {tool["function"]["name"] for tool in tools}
    public_input = task.get("public_input") if isinstance(task.get("public_input"), dict) else {}
    initial_message = public_input.get("initial_user_message")
    if not isinstance(initial_message, str) or not initial_message.strip():
        initial_message = task["task"]
    materials = public_input.get("materials")
    if isinstance(materials, list) and materials:
        initial_message = initial_message.strip() + "\n\nPublic materials:\n" + json.dumps(
            materials, ensure_ascii=False
        )
    conversation = [{"role": "user", "content": initial_message}]
    messages = [{"role": "system", "content": (
        'Complete the user task using only available tools and public observations. Return JSON only: '
        '{"kind":"tool","name":"...","arguments":{...}} or '
        '{"kind":"respond","content":"..."}. Respond to ask the user a question or deliver your answer. '
        'Never invent a tool result. Tool schemas: ' + json.dumps(tools, ensure_ascii=False))}, *conversation,
        {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)}]
    termination = "step_budget"
    protocol_errors = 0
    for step in range(max_steps):
        agent_input = copy.deepcopy(messages)
        response = client.chat(messages, response_format={"type": "json_object"})
        usage.append({
            "response_model": response.model,
            "response_id_sha256": (
                hashlib.sha256(response.id.encode("utf-8")).hexdigest()
                if response.id else None
            ),
            "finish_reason": response.finish_reason,
            "token_usage": dict(response.usage or {}),
        })
        messages.append({"role": "assistant", "content": response.content})
        transition = {
            "step": step,
            "agent_input": agent_input,
            "assistant_output": response.content,
            "observation": copy.deepcopy(observation),
            "action": None,
            "result": None,
            "next_observation": None,
            "reward": None,
            "terminated": False,
            "truncated": False,
        }
        try:
            action = json.loads(response.content)
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            transition["action"] = copy.deepcopy(action)
            if action.get("kind") == "tool":
                if action.get("name") not in names or not isinstance(action.get("arguments"), dict):
                    raise ValueError("unknown tool or invalid arguments")
                status, result = request("POST", "/v1/tools/" + action["name"], action["arguments"])
                messages.append({"role": "user", "content": json.dumps({"tool_result": result, "status": status}, ensure_ascii=False)})
                transition["result"] = {"status": status, "tool_result": copy.deepcopy(result)}
            elif action.get("kind") == "respond" and isinstance(action.get("content"), str) and action["content"].strip():
                request("POST", "/v1/agent_response", {"content": action["content"]})
                conversation.append({"role": "assistant", "content": action["content"]})
                user = request("POST", "/v1/user_simulator", {"messages": conversation})[1]
                conversation.append({"role": "user", "content": user["user_query"]})
                messages.append({"role": "user", "content": user["user_query"]})
                # Store exactly what becomes visible to the policy.  Outcome
                # labels and FSM decisions remain trainer-only evidence.
                transition["result"] = {
                    "status": 200,
                    "user_query": user["user_query"],
                }
                transition["trainer_metadata"] = {
                    "user_simulator": copy.deepcopy(user),
                }
                if user.get("should_end"):
                    termination = user.get("termination_reason", "user_ended")
            else:
                raise ValueError("expected tool or respond action")
            observation = request("GET", "/v1/observation")[1]
            transition["next_observation"] = copy.deepcopy(observation)
            current_reward = request("GET", "/v1/reward")[1]["reward"]
            transition["reward"] = current_reward
            transition["terminated"] = bool(user.get("should_end")) if action.get("kind") == "respond" else False
            transitions.append(transition)
            if transition["terminated"]:
                break
        except (ValueError, TypeError) as exc:
            protocol_errors += 1
            transition["result"] = {
                "status": 422,
                "protocol_error": str(exc),
            }
            transition["next_observation"] = copy.deepcopy(observation)
            transition["reward"] = current_reward
            transitions.append(transition)
            messages.append({"role": "user", "content": f"Invalid action: {exc}. Return a corrected JSON action."})
            if protocol_errors >= 3:
                termination = "agent_protocol_error"
                transitions[-1]["terminated"] = True
                break
    if transitions and termination == "step_budget":
        transitions[-1]["truncated"] = True
    final_reward = request("GET", "/v1/reward")[1]["reward"]
    repeated_reward = request("GET", "/v1/reward")[1]["reward"]
    replay = request("GET", "/v1/replay")[1]
    state_payload = request("GET", "/v1/state")[1]
    state = state_payload.get("business_state")
    if not isinstance(state, dict):
        raise RuntimeError("trainer state endpoint returned an invalid final state")
    goal = task.get("task_spec", {}).get("goal_contract", {})
    state_success = None
    if goal.get("row_predicates"):
        state_success = (BusinessGoalEvaluator.evaluate(goal["row_predicates"], state, baseline)
                         and BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
                         and (not goal.get("requires_state_change") or baseline != state))
    fallbacks = [event for event in replay.get("events", []) if event.get("payload", {}).get("used_fallback")]
    issues = []
    if initial_reward >= 0.8:
        issues.append("high_reward_without_actions")
    if final_reward != repeated_reward:
        issues.append("unstable_reward_read")
    if state_success is False and final_reward >= 0.8:
        issues.append("reward_state_disagreement")
    if fallbacks:
        issues.append("runtime_llm_fallback")
    success = termination == "completed" and final_reward >= 0.8 and state_success is not False and not issues
    return {"schema_version": "2.0", "seed": seed, "agent_success": success, "termination": termination,
            "initial_reward": initial_reward, "final_reward": final_reward,
            "state_goal_satisfied": state_success, "issues": issues, "usage": usage,
            "protocol_errors": protocol_errors, "trajectory": trace,
            "transitions": transitions, "replay": replay,
            "initial_state": baseline, "final_state": state}


def summarize_episodes(episodes, minimum_success_rate):
    """Turn trajectory evidence into a strict, reusable rollout gate."""
    count = len(episodes)
    agent_success_rate = (
        sum(item.get("agent_success") is True for item in episodes) / count if count else 0
    )
    all_episodes_fallback_free = bool(episodes) and not any(
        "runtime_llm_fallback" in item.get("issues", []) for item in episodes
    )
    all_episodes_environment_clean = bool(episodes) and not any(
        item.get("issues") for item in episodes
    )
    live_rollout_verified = bool(episodes) and all(
        "execution_error" not in item.get("issues", [])
        and "runtime_llm_fallback" not in item.get("issues", [])
        for item in episodes
    )
    success_rate_gate_passed = (
        agent_success_rate > 0
        if minimum_success_rate == 0
        else agent_success_rate >= minimum_success_rate
    )
    passed = (
        live_rollout_verified
        and all_episodes_environment_clean
        and all_episodes_fallback_free
    )
    if any("execution_error" in item.get("issues", []) for item in episodes):
        failure_owner = "infrastructure"
    elif not all_episodes_environment_clean or not live_rollout_verified:
        failure_owner = "environment"
    else:
        failure_owner = None
    return {
        "live_rollout_verified": live_rollout_verified,
        "agent_success_rate": agent_success_rate,
        "qualification_policy_version": "3.0",
        "task_solvability_witness": agent_success_rate > 0,
        "environment_qualified": live_rollout_verified and all_episodes_environment_clean,
        "agent_policy_qualified": success_rate_gate_passed,
        "score_scope": "live_environment_integrity",
        "environment_checks_passed": all_episodes_environment_clean,
        "all_episodes_fallback_free": all_episodes_fallback_free,
        "all_episodes_environment_clean": all_episodes_environment_clean,
        "minimum_success_rate": minimum_success_rate,
        "success_rate_gate_passed": success_rate_gate_passed,
        "quality_score": round(
            (0.3 if live_rollout_verified else 0.0)
            + (0.3 if all_episodes_environment_clean else 0.0)
            + (0.4 if all_episodes_fallback_free else 0.0),
            2,
        ),
        "passed": passed,
        "failure_owner": failure_owner,
        "agent_failure_owner": None if success_rate_gate_passed else "agent",
        "conclusion": (
            "live_success_witness" if passed and agent_success_rate > 0 else
            "environment_clean_no_agent_success" if passed else "environment_or_execution_failure"
        ),
    }


def rollout_provider_attestation(client, runtime):
    """Use the governance identity algorithm for every rollout provider."""
    return {
        "agent_provider_sha256": provider_identity(
            client.base_url, client.model
        )["identity_sha256"],
        "runtime_provider_sha256": provider_identity(
            runtime.base_url, runtime.model
        )["identity_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--base-url")
    parser.add_argument("--container-image-id")
    parser.add_argument(
        "--min-success-rate", type=float, default=2 / 3,
        help="minimum successful episode ratio; 0 preserves the one-success witness gate",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.min_success_rate <= 1:
        parser.error("episodes and steps must be positive")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    roles = resolve_model_roles(os.environ)
    agent = roles["agent"]
    client = LLMClient(
        api_key=agent["api_key"],
        base_url=agent["base_url"],
        model=agent["model"],
        timeout=float(agent["timeout_seconds"]),
    )
    os.environ["SANDBOX_EVALUATOR_MOCK"] = "0"
    runtime = RuntimeLLMConfig.from_env()
    os.environ["SANDBOX_MUTATION_MODE"] = "disabled"
    os.environ.setdefault("SANDBOX_TRAINER_API_KEY", "local-rollout-trainer")
    root = args.root.resolve()
    task = json.loads((root / "task.json").read_text())
    if bool(args.base_url) != bool(args.container_image_id):
        parser.error("--base-url and --container-image-id must be provided together")
    if args.base_url:
        import re
        if re.fullmatch(r"sha256:[0-9a-f]{64}", args.container_image_id) is None:
            parser.error("--container-image-id must be a Docker sha256 image ID")
        app = HTTPSandboxClient(
            args.base_url,
            timeout=evaluator_http_timeout(runtime.timeout_seconds, runtime.max_retries),
        )
        module = None
        runtime_execution = {
            "version": "1.0",
            "mode": "docker_http",
            "container_image_id": args.container_image_id,
            "transport": "loopback_http",
            "read_only_root": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "non_root_user": True,
        }
    else:
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("rollout_app", root / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        app = None
        runtime_execution = {
            "version": "1.0",
            "mode": "in_process",
            "container_image_id": None,
        }
    provider_attestation = rollout_provider_attestation(client, runtime)
    report = {"schema_version": "2.0", "material_visibility_version": "1.0",
              "mode": "live_rollout", "agent_model": client.model,
              "runtime_model": runtime.model, "episodes": [],
              "task_sha256": hashlib.sha256((root / "task.json").read_bytes()).hexdigest(),
              "sandbox_artifacts_digest": portable_artifact_digest(root),
              **provider_attestation,
              "runtime_execution": runtime_execution,
              "same_model_bias_possible": client.model == runtime.model,
              "same_provider_identity": (
                  provider_attestation["agent_provider_sha256"]
                  == provider_attestation["runtime_provider_sha256"]
              ),
              "live_rollout_verified": False, "passed": False}
    with tempfile.TemporaryDirectory(prefix="envfactory-rollout-") as temporary:
        if app is None:
            app = module.create_app(db_path=Path(temporary) / "episodes.sqlite3")
        for seed in range(args.episodes):
            try:
                result = episode(app, task, client, seed, args.max_steps)
            except Exception as exc:
                # Error messages from providers may contain credentials; record type only.
                result = {"seed": seed, "agent_success": False, "issues": ["execution_error"], "error_type": type(exc).__name__}
            report["episodes"].append(result)
            write_json(args.output, report)
    report.update(summarize_episodes(report["episodes"], args.min_success_rate))
    write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
