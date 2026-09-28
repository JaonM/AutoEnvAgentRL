"""Validate actual provider response identities captured during live rollout."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping


HEX64 = re.compile(r"[0-9a-f]{64}")


def _agent_usage(value: Any) -> tuple[bool, Counter[str]]:
    if not isinstance(value, Mapping):
        return False, Counter()
    model = value.get("response_model")
    response_id = value.get("response_id_sha256")
    finish = value.get("finish_reason")
    tokens = value.get("token_usage")
    valid = (
        isinstance(model, str) and bool(model.strip())
        and (response_id is None or (
            isinstance(response_id, str) and HEX64.fullmatch(response_id) is not None
        ))
        and (finish is None or isinstance(finish, str))
        and isinstance(tokens, Mapping)
        and all(
            isinstance(number, (int, float)) and not isinstance(number, bool)
            and number >= 0
            for number in tokens.values()
        )
    )
    return valid, Counter({model: 1}) if valid else Counter()


def _runtime_summary(value: Any) -> tuple[bool, Counter[str]]:
    if not isinstance(value, Mapping):
        return False, Counter()
    responses = value.get("responses")
    models = value.get("models")
    identifiers = value.get("response_id_sha256")
    usage = value.get("usage")
    valid_models = (
        isinstance(models, Mapping) and bool(models)
        and all(
            isinstance(model, str) and bool(model.strip())
            and isinstance(count, int) and not isinstance(count, bool) and count > 0
            for model, count in models.items()
        )
    )
    valid = (
        value.get("version") == "1.0"
        and isinstance(responses, int) and not isinstance(responses, bool)
        and responses > 0
        and value.get("mock_responses") == 0
        and valid_models
        and sum(models.values()) == responses
        and isinstance(identifiers, list)
        and all(
            isinstance(identifier, str) and HEX64.fullmatch(identifier) is not None
            for identifier in identifiers
        )
        and isinstance(usage, Mapping)
        and all(
            isinstance(number, (int, float)) and not isinstance(number, bool)
            and number >= 0
            for number in usage.values()
        )
    )
    return valid, Counter(models) if valid else Counter()


def response_provenance(rollout: Any) -> dict[str, Any]:
    """Recompute per-role actual model counts from immutable episode evidence."""
    episodes = rollout.get("episodes") if isinstance(rollout, Mapping) else None
    if not isinstance(episodes, list) or not episodes:
        return {
            "verified": False,
            "agent_response_models": {},
            "runtime_response_models": {},
            "failed_episodes": [],
        }
    agent_models: Counter[str] = Counter()
    runtime_models: Counter[str] = Counter()
    failed = []
    for index, episode in enumerate(episodes):
        episode_valid = isinstance(episode, Mapping)
        transitions = episode.get("transitions") if episode_valid else None
        usages = episode.get("usage") if episode_valid else None
        if not (
            isinstance(transitions, list) and transitions
            and isinstance(usages, list) and len(usages) == len(transitions)
        ):
            episode_valid = False
            usages = []
        for usage_item in usages:
            valid, counts = _agent_usage(usage_item)
            episode_valid = episode_valid and valid
            agent_models.update(counts)
        replay = episode.get("replay") if isinstance(episode, Mapping) else None
        events = replay.get("events") if isinstance(replay, Mapping) else None
        runtime_summaries = [
            event.get("payload", {}).get("summary")
            for event in events if isinstance(event, Mapping)
            and event.get("event") == "runtime_llm_call"
            and isinstance(event.get("payload"), Mapping)
        ] if isinstance(events, list) else []
        responded = any(
            isinstance(transition, Mapping)
            and isinstance(transition.get("action"), Mapping)
            and transition["action"].get("kind") == "respond"
            for transition in transitions
        ) if isinstance(transitions, list) else False
        if responded and not runtime_summaries:
            episode_valid = False
        for summary in runtime_summaries:
            valid, counts = _runtime_summary(summary)
            episode_valid = episode_valid and valid
            runtime_models.update(counts)
        if not episode_valid:
            failed.append(index)
    return {
        "verified": not failed and bool(agent_models) and bool(runtime_models),
        "agent_response_models": dict(sorted(agent_models.items())),
        "runtime_response_models": dict(sorted(runtime_models.items())),
        "failed_episodes": failed,
    }
