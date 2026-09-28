"""Canonical policy for production Agentic-RL material preparation."""

from __future__ import annotations

from typing import Any, Mapping


POLICY_VERSION = "1.0"
MINIMUM_SCORE_THRESHOLD = 8.0


def canonical_certification_policy(
    *, score_threshold: float = MINIMUM_SCORE_THRESHOLD,
) -> dict[str, Any]:
    if (
        not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not MINIMUM_SCORE_THRESHOLD <= float(score_threshold) <= 10.0
    ):
        raise ValueError("production score threshold must be between 8 and 10")
    return {
        "version": POLICY_VERSION,
        "score_threshold": float(score_threshold),
        "min_tasks": 300,
        "min_holdout_batches": 3,
        "min_task_yield": 0.90,
        "min_task_yield_ci95_lower": 0.85,
        "min_build_yield": 0.90,
        "min_build_yield_ci95_lower": 0.85,
        "min_end_to_end_rate": 0.85,
        "min_end_to_end_ci95_lower": 0.80,
        "min_category_rate": 0.75,
        "max_near_duplicate_rate": 0.05,
        "min_episodes_per_qualified_sandbox": 10,
        "min_agent_success_rate_per_sandbox": 2 / 3,
        "min_total_episodes": 7500,
        "max_environment_error_rate": 0.001,
        "max_reward_false_positive_rate": 0.005,
        "max_reward_false_negative_rate": 0.02,
        "max_same_provider_evaluator_rate": 0.0,
        "min_user_simulator_protocol_rate": 0.995,
        "min_user_outcome_categories": 3,
        "min_category_shares": {
            "direct_response": 0.10,
            "simple_agentic": 0.20,
            "multi_step_agentic": 0.35,
        },
        "max_category_shares": {"direct_response": 0.30},
    }


def policy_for_experiment(
    config: Mapping[str, Any] | Any,
) -> tuple[dict[str, Any], bool]:
    """Resolve the only policy allowed for a frozen production experiment."""
    if not isinstance(config, Mapping):
        return canonical_certification_policy(), False
    threshold = config.get("threshold", MINIMUM_SCORE_THRESHOLD)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return canonical_certification_policy(), False
    try:
        return canonical_certification_policy(
            score_threshold=threshold
        ), True
    except (TypeError, ValueError):
        return canonical_certification_policy(), False


def valid_certification_policy(
    value: Mapping[str, Any] | Any, *, expected_threshold: float | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        threshold = float(value.get("score_threshold"))
        expected = canonical_certification_policy(
            score_threshold=(threshold if expected_threshold is None else expected_threshold)
        )
    except (TypeError, ValueError):
        return False
    return dict(value) == expected
