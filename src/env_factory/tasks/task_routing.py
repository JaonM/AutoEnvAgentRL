"""Stratified routing for a diverse tool-use training curriculum."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import Any, Mapping


TRAINING_CATEGORIES = (
    "direct_response",
    "simple_agentic",
    "multi_step_agentic",
)

DEFAULT_TRAINING_MIX: dict[str, float] = {
    "direct_response": 0.20,
    "simple_agentic": 0.30,
    "multi_step_agentic": 0.50,
}


@dataclass(frozen=True)
class TrainingRoute:
    category: str
    agentic_level: int
    tool_policy_target: str
    sandbox_profile: str
    allowed_environment_modes: tuple[str, ...]
    business_tool_min: int
    business_tool_max: int | None
    dependency_required: bool
    required_scenarios: tuple[str, ...]
    required_counterfactuals: tuple[str, ...]
    allowed_intents: tuple[str, ...]


ROUTES = {
    "direct_response": TrainingRoute(
        "direct_response", 0, "do_not_call", "direct_response", ("stateless",), 0, 0, False,
        ("goal_success", "goal_failure"), ("unnecessary_tool",),
        ("explain", "summarize", "extract", "classify", "transform", "create", "calculate", "compare"),
    ),
    "simple_agentic": TrainingRoute(
        "simple_agentic", 1, "call_required_business_tool", "single_tool", 
        ("reference_data", "stateful", "external_capability"), 1, 1, False,
        ("goal_success", "goal_failure"), ("no_tool", "corrupted_arguments"),
        ("query", "validate", "calculate", "estimate", "modify", "monitor", "execute", "recommend"),
    ),
    "multi_step_agentic": TrainingRoute(
        "multi_step_agentic", 2, "follow_dependency_chain", "dependent_tool_chain",
        ("reference_data", "stateful", "external_capability"), 2, None, True,
        ("goal_success", "goal_failure"),
        ("no_tool", "skip_step", "wrong_order", "corrupted_arguments"),
        ("query", "execute", "plan", "diagnose", "modify", "audit", "schedule", "monitor", "troubleshoot", "simulate", "validate", "decide", "calculate"),
    ),
}


def training_contract(category: str) -> dict[str, Any]:
    """Return the single, serializable blueprint shared by generation and sandboxes."""
    try:
        route = ROUTES[category]
    except KeyError as exc:
        raise ValueError(f"unsupported training category: {category}") from exc
    return {
        "version": "1.0",
        "category": route.category,
        "agentic_level": route.agentic_level,
        "sandbox_profile": route.sandbox_profile,
        "allowed_environment_modes": list(route.allowed_environment_modes),
        "business_tools": {"min": route.business_tool_min, "max": route.business_tool_max},
        "dependency": {
            "required": route.dependency_required,
            "mechanism": "capture_ref" if route.dependency_required else "none",
        },
        "correct_tool_policy": route.tool_policy_target,
        "required_scenarios": list(route.required_scenarios),
        "required_counterfactuals": list(route.required_counterfactuals),
        "allowed_intents": list(route.allowed_intents),
    }


def select_training_intent(
    category: str, requested: str | None = None, *, rng: random.Random | None = None,
) -> str:
    """Select an intent compatible with the route, or reject an invalid override."""
    try:
        allowed = ROUTES[category].allowed_intents
    except KeyError as exc:
        raise ValueError(f"unsupported training category: {category}") from exc
    if requested is not None:
        if requested not in allowed:
            raise ValueError(
                f"task intent {requested!r} is incompatible with training category "
                f"{category!r}; allowed: {', '.join(allowed)}"
            )
        return requested
    return (rng or random).choice(allowed)


def compatible_training_categories(intent: str) -> tuple[str, ...]:
    categories = tuple(name for name, route in ROUTES.items() if intent in route.allowed_intents)
    if not categories:
        raise ValueError(f"unsupported task intent: {intent}")
    return categories


def parse_training_mix(value: str | None) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_TRAINING_MIX)
    result: dict[str, float] = {}
    for item in value.split(","):
        try:
            name, raw_weight = item.split("=", 1)
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError("training mix must use category=weight entries") from exc
        name = name.strip()
        if name not in TRAINING_CATEGORIES or weight < 0:
            raise ValueError(f"invalid training mix entry: {item}")
        result[name] = weight
    if set(result) != set(TRAINING_CATEGORIES) or sum(result.values()) <= 0:
        raise ValueError("training mix must contain every training category with a non-zero total")
    total = sum(result.values())
    return {name: result[name] / total for name in TRAINING_CATEGORIES}


def allocate_training_routes(
    count: int, mix: Mapping[str, float] | None = None, *, rng: random.Random | None = None,
    allowed_categories: tuple[str, ...] | None = None,
) -> list[str]:
    """Allocate exact batch quotas with the largest-remainder method."""
    if count <= 0:
        raise ValueError("count must be positive")
    normalized = parse_training_mix(None) if mix is None else parse_training_mix(
        ",".join(f"{name}={mix.get(name, 0)}" for name in TRAINING_CATEGORIES)
    )
    categories = allowed_categories or TRAINING_CATEGORIES
    if not categories or any(name not in TRAINING_CATEGORIES for name in categories):
        raise ValueError("allowed_categories must contain supported training categories")
    selected_total = sum(normalized[name] for name in categories)
    if selected_total <= 0:
        raise ValueError("training mix assigns zero weight to every compatible category")
    selected_mix = {name: normalized[name] / selected_total for name in categories}
    exact = {name: count * selected_mix[name] for name in categories}
    quotas = {name: floor(exact[name]) for name in categories}
    remaining = count - sum(quotas.values())
    order = sorted(categories, key=lambda name: (-(exact[name] - quotas[name]), TRAINING_CATEGORIES.index(name)))
    for name in order[:remaining]:
        quotas[name] += 1
    routes = [name for name in categories for _ in range(quotas[name])]
    (rng or random).shuffle(routes)
    return routes
