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
