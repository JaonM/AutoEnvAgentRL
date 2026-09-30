"""Canonical, trainer-facing schema checks for collected rollout trajectories."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping


POLICY_TRANSITION_FIELDS = (
    "step",
    "agent_input",
    "assistant_output",
    "observation",
    "action",
    "result",
    "next_observation",
    "reward",
    "terminated",
    "truncated",
)


def policy_transition(transition: Mapping[str, Any]) -> dict[str, Any]:
    """Project one step onto the stable trainer/policy interchange contract."""
    return {name: transition.get(name) for name in POLICY_TRANSITION_FIELDS}


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def transition_errors(transition: Any, *, expected_step: int) -> list[str]:
    prefix = f"transitions[{expected_step}]"
    if not isinstance(transition, Mapping):
        return [f"{prefix}:not_object"]
    errors = []
    if transition.get("step") != expected_step:
        errors.append(f"{prefix}:step")
    messages = transition.get("agent_input")
    if not (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(message, Mapping)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        errors.append(f"{prefix}:agent_input")
    raw = transition.get("assistant_output")
    if not isinstance(raw, str):
        errors.append(f"{prefix}:assistant_output")
    action = transition.get("action")
    parsed = None
    if isinstance(raw, str):
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, Mapping):
                parsed = dict(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    if action is not None and not isinstance(action, Mapping):
        errors.append(f"{prefix}:action")
    elif (dict(action) if isinstance(action, Mapping) else None) != parsed:
        errors.append(f"{prefix}:parsed_action_mismatch")
    for name in ("observation", "result", "next_observation"):
        if not isinstance(transition.get(name), Mapping):
            errors.append(f"{prefix}:{name}")
    if not _number(transition.get("reward")):
        errors.append(f"{prefix}:reward")
    if not isinstance(transition.get("terminated"), bool):
        errors.append(f"{prefix}:terminated")
    if not isinstance(transition.get("truncated"), bool):
        errors.append(f"{prefix}:truncated")
    if transition.get("terminated") is True and transition.get("truncated") is True:
        errors.append(f"{prefix}:dual_terminal")
    trainer_metadata = transition.get("trainer_metadata")
    if trainer_metadata is not None and not isinstance(trainer_metadata, Mapping):
        errors.append(f"{prefix}:trainer_metadata")
    elif isinstance(trainer_metadata, Mapping) and "user_simulator" in trainer_metadata:
        simulator = trainer_metadata.get("user_simulator")
        visible = transition.get("result")
        if not isinstance(simulator, Mapping):
            errors.append(f"{prefix}:user_simulator_metadata")
        else:
            if not isinstance(action, Mapping) or action.get("kind") != "respond":
                errors.append(f"{prefix}:user_simulator_action")
            if not isinstance(visible, Mapping) or (
                visible.get("user_query") != simulator.get("user_query")
            ):
                errors.append(f"{prefix}:user_simulator_visible_result")
            if transition.get("terminated") is not simulator.get("should_end"):
                errors.append(f"{prefix}:user_simulator_terminal")
            if (
                simulator.get("should_end") is False
                and simulator.get("termination_reason") is not None
            ):
                errors.append(f"{prefix}:premature_termination_reason")
    return errors


def episode_errors(episode: Any) -> list[str]:
    if not isinstance(episode, Mapping):
        return ["episode:not_object"]
    errors = []
    if episode.get("schema_version") != "2.0":
        errors.append("episode:schema_version")
    if not isinstance(episode.get("seed"), int) or isinstance(episode.get("seed"), bool):
        errors.append("episode:seed")
    if not isinstance(episode.get("agent_success"), bool):
        errors.append("episode:agent_success")
    termination = episode.get("termination")
    if not isinstance(termination, str) or not termination:
        errors.append("episode:termination")
    for name in ("initial_reward", "final_reward"):
        if not _number(episode.get(name)):
            errors.append(f"episode:{name}")
    transitions = episode.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        errors.append("episode:transitions")
        transitions = []
    for index, transition in enumerate(transitions):
        errors.extend(transition_errors(transition, expected_step=index))
    if transitions and all(isinstance(item, Mapping) for item in transitions):
        markers = [
            bool(item.get("terminated") or item.get("truncated"))
            for item in transitions
        ]
        if any(markers[:-1]) or not markers[-1]:
            errors.append("episode:terminal_position")
        last = transitions[-1]
        if termination == "step_budget":
            if last.get("truncated") is not True or last.get("terminated") is not False:
                errors.append("episode:step_budget_terminal")
        elif last.get("terminated") is not True or last.get("truncated") is not False:
            errors.append("episode:terminal_flag")
        if last.get("reward") != episode.get("final_reward"):
            errors.append("episode:final_reward_mismatch")
        trainer_metadata = last.get("trainer_metadata")
        simulator = (
            trainer_metadata.get("user_simulator")
            if isinstance(trainer_metadata, Mapping) else None
        )
        if (
            isinstance(simulator, Mapping)
            and simulator.get("should_end") is True
            and episode.get("termination") != simulator.get("termination_reason")
        ):
            errors.append("episode:user_simulator_termination_mismatch")
        for index in range(len(transitions) - 1):
            if transitions[index].get("next_observation") != transitions[index + 1].get(
                "observation"
            ):
                errors.append(f"episode:observation_chain[{index}]")
    trajectory = episode.get("trajectory")
    if not (
        isinstance(trajectory, list)
        and trajectory
        and all(
            isinstance(step, Mapping)
            and isinstance(step.get("method"), str)
            and isinstance(step.get("path"), str)
            and isinstance(step.get("status"), int)
            and "result" in step
            for step in trajectory
        )
    ):
        errors.append("episode:trajectory")
        trajectory = []
    prefix = [
        ("POST", "/v1/reset"), ("GET", "/v1/state"),
        ("GET", "/v1/reward"), ("GET", "/v1/tools"),
        ("GET", "/v1/observation"),
    ]
    suffix = [
        ("GET", "/v1/reward"), ("GET", "/v1/reward"),
        ("GET", "/v1/replay"), ("GET", "/v1/state"),
    ]
    if (
        len(trajectory) < len(prefix) + len(suffix)
        or [(step.get("method"), step.get("path")) for step in trajectory[:5]] != prefix
        or [(step.get("method"), step.get("path")) for step in trajectory[-4:]] != suffix
    ):
        errors.append("episode:http_skeleton")
    else:
        reset = trajectory[0]
        reset_body = reset.get("body")
        if (
            reset.get("status") != 200
            or not isinstance(reset_body, Mapping)
            or reset_body.get("seed") != episode.get("seed")
            or reset_body.get("episode_id") != f"live-{episode.get('seed')}"
        ):
            errors.append("episode:reset_http_mismatch")
        tools_step = trajectory[3]
        tools_result = tools_step.get("result")
        if (
            tools_step.get("status") != 200
            or not isinstance(tools_result, Mapping)
            or not isinstance(tools_result.get("tools"), list)
        ):
            errors.append("episode:tools_http_result")
        replay_step = trajectory[-2]
        if (
            replay_step.get("status") != 200
            or replay_step.get("result") != episode.get("replay")
        ):
            errors.append("episode:replay_http_mismatch")
    reward_positions = [
        index for index, step in enumerate(trajectory)
        if step.get("method") == "GET" and step.get("path") == "/v1/reward"
    ]
    reward_reads = [trajectory[index] for index in reward_positions]
    reward_values = []
    for step in reward_reads:
        result = step.get("result")
        reward = result.get("reward") if isinstance(result, Mapping) else None
        if step.get("status") != 200 or not _number(reward):
            errors.append("episode:reward_http_result")
        reward_values.append(reward)
    if len(reward_values) < 3:
        errors.append("episode:reward_http_coverage")
    else:
        initial_observations = [
            step for step in trajectory[reward_positions[0] + 1:reward_positions[1]]
            if step.get("method") == "GET" and step.get("path") == "/v1/observation"
        ]
        if (
            not initial_observations
            or initial_observations[0].get("status") != 200
            or not transitions
            or not isinstance(transitions[0], Mapping)
            or initial_observations[0].get("result") != transitions[0].get("observation")
        ):
            errors.append("episode:initial_observation_http_mismatch")
        previous_position = reward_positions[0]
        if reward_values[0] != episode.get("initial_reward"):
            errors.append("episode:initial_reward_http_mismatch")
        cursor = 1
        current = reward_values[0]
        first_executed = True
        for index, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                continue
            result = transition.get("result")
            protocol_error = isinstance(result, Mapping) and "protocol_error" in result
            if not protocol_error:
                if cursor >= len(reward_values) - 2:
                    errors.append("episode:transition_reward_http_coverage")
                    break
                reward_position = reward_positions[cursor]
                window = trajectory[previous_position + 1:reward_position]
                action = transition.get("action")
                action_requests = [
                    step for step in window
                    if step.get("method") == "POST"
                    and (step.get("path", "").startswith("/v1/tools/")
                         or step.get("path") in {"/v1/agent_response", "/v1/user_simulator"})
                ]
                if isinstance(action, Mapping) and action.get("kind") == "tool":
                    expected_path = "/v1/tools/" + str(action.get("name"))
                    expected_result = transition.get("result")
                    if (
                        len(action_requests) != 1
                        or action_requests[0].get("path") != expected_path
                        or action_requests[0].get("body") != action.get("arguments")
                        or not isinstance(expected_result, Mapping)
                        or action_requests[0].get("status") != expected_result.get("status")
                        or action_requests[0].get("result") != expected_result.get("tool_result")
                    ):
                        errors.append(f"episode:action_http_mismatch[{index}]")
                elif isinstance(action, Mapping) and action.get("kind") == "respond":
                    simulator = transition.get("trainer_metadata")
                    simulator = simulator.get("user_simulator") if isinstance(simulator, Mapping) else None
                    if (
                        len(action_requests) != 2
                        or [step.get("path") for step in action_requests]
                        != ["/v1/agent_response", "/v1/user_simulator"]
                        or action_requests[0].get("body") != {"content": action.get("content")}
                        or action_requests[0].get("status") != 200
                        or action_requests[1].get("status") != 200
                        or action_requests[1].get("result") != simulator
                    ):
                        errors.append(f"episode:action_http_mismatch[{index}]")
                else:
                    errors.append(f"episode:action_http_mismatch[{index}]")
                observations = [
                    step for step in window
                    if step.get("method") == "GET" and step.get("path") == "/v1/observation"
                ]
                if (
                    len(observations) != (2 if first_executed else 1)
                    or any(step.get("status") != 200 for step in observations)
                    or (first_executed and observations[0].get("result") != transition.get("observation"))
                    or observations[-1].get("result") != transition.get("next_observation")
                    or not window or window[-1] is not observations[-1]
                    or (first_executed and action_requests and window.index(observations[0]) > window.index(action_requests[0]))
                ):
                    errors.append(f"episode:observation_http_mismatch[{index}]")
                current = reward_values[cursor]
                cursor += 1
                previous_position = reward_position
                first_executed = False
            if transition.get("reward") != current:
                errors.append(f"episode:transition_reward_http_mismatch[{index}]")
        if cursor + 2 != len(reward_values):
            errors.append("episode:reward_http_count")
        if reward_values[-2:] != [episode.get("final_reward")] * 2:
            errors.append("episode:final_reward_http_mismatch")
    actual_actions = [
        step for step in trajectory
        if step.get("method") == "POST"
        and (step.get("path", "").startswith("/v1/tools/")
             or step.get("path") in {"/v1/agent_response", "/v1/user_simulator"})
    ]
    expected_actions = sum(
        0 if isinstance(item.get("result"), Mapping) and "protocol_error" in item["result"]
        else 2 if isinstance(item.get("action"), Mapping) and item["action"].get("kind") == "respond"
        else 1
        for item in transitions if isinstance(item, Mapping)
    )
    if len(actual_actions) != expected_actions:
        errors.append("episode:action_http_count")
    state_positions = [
        index for index, step in enumerate(trajectory)
        if step.get("method") == "GET" and step.get("path") == "/v1/state"
    ]
    state_reads = [trajectory[index] for index in state_positions]
    if len(state_reads) != 2 or any(step.get("status") != 200 for step in state_reads):
        errors.append("episode:state_http_coverage")
    else:
        if len(reward_positions) >= 3 and not (
            state_positions[0] < reward_positions[0]
            and reward_positions[-1] < state_positions[1]
        ):
            errors.append("episode:state_reward_http_order")
        states = [
            step.get("result", {}).get("business_state")
            if isinstance(step.get("result"), Mapping) else None
            for step in state_reads
        ]
        if states[0] != episode.get("initial_state"):
            errors.append("episode:initial_state_http_mismatch")
        if states[1] != episode.get("final_state"):
            errors.append("episode:final_state_http_mismatch")
    raw_user_results = [
        step.get("result") for step in trajectory
        if isinstance(step, Mapping) and step.get("path") == "/v1/user_simulator"
    ]
    metadata_user_results = [
        transition.get("trainer_metadata", {}).get("user_simulator")
        for transition in transitions if isinstance(transition, Mapping)
        and isinstance(transition.get("trainer_metadata"), Mapping)
        and "user_simulator" in transition["trainer_metadata"]
    ]
    if raw_user_results != metadata_user_results:
        errors.append("episode:user_simulator_evidence_mismatch")
    for name in ("replay", "initial_state", "final_state"):
        if not isinstance(episode.get(name), Mapping):
            errors.append(f"episode:{name}")
    usage = episode.get("usage")
    if not (
        isinstance(usage, list)
        and len(usage) == len(transitions)
        and all(isinstance(item, Mapping) for item in usage)
    ):
        errors.append("episode:usage")
    issues = episode.get("issues")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        errors.append("episode:issues")
    return errors


def complete_episode(episode: Any) -> bool:
    return not episode_errors(episode)
