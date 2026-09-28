"""Resolve model-provider roles with explicit, field-wise fallbacks."""

from __future__ import annotations

import os
from typing import Any, Mapping


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _first(environment: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = environment.get(name, "").strip()
        if value:
            return value
    return default


def _allowlist(value: str, fallback_model: str) -> list[str]:
    values = list(dict.fromkeys(
        item.strip() for item in value.split(",") if item.strip()
    ))
    return values or ([fallback_model] if fallback_model else [])


def resolve_model_roles(
    environment: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return generation, rollout-agent and runtime model configurations.

    Rollout and sandbox settings fall back one field at a time to generation
    settings, so overriding a model never accidentally discards the base URL
    or credential fallback for that role.
    """
    env = os.environ if environment is None else environment
    generation = {
        "api_key": _first(env, "LLM_API_KEY"),
        "base_url": _first(env, "LLM_BASE_URL", default=DEFAULT_BASE_URL).rstrip("/"),
        "model": _first(env, "LLM_MODEL"),
        "timeout_seconds": _first(env, "LLM_TIMEOUT", default="60"),
    }
    generation["allowed_response_models"] = _allowlist(
        _first(env, "LLM_ALLOWED_RESPONSE_MODELS"), generation["model"]
    )
    agent = {
        "api_key": _first(env, "ROLLOUT_LLM_API_KEY", "LLM_API_KEY"),
        "base_url": _first(
            env, "ROLLOUT_LLM_BASE_URL", "LLM_BASE_URL",
            default=DEFAULT_BASE_URL,
        ).rstrip("/"),
        "model": _first(env, "ROLLOUT_LLM_MODEL", "LLM_MODEL"),
        "timeout_seconds": _first(
            env, "ROLLOUT_LLM_TIMEOUT_SECONDS", "ROLLOUT_LLM_TIMEOUT",
            "LLM_TIMEOUT", default="60",
        ),
    }
    agent_allowlist = _first(env, "ROLLOUT_LLM_ALLOWED_RESPONSE_MODELS")
    if not agent_allowlist and not _first(env, "ROLLOUT_LLM_MODEL"):
        agent_allowlist = _first(env, "LLM_ALLOWED_RESPONSE_MODELS")
    agent["allowed_response_models"] = _allowlist(
        agent_allowlist, agent["model"]
    )
    runtime = {
        "api_key": _first(env, "SANDBOX_LLM_API_KEY", "LLM_API_KEY"),
        "base_url": _first(
            env, "SANDBOX_LLM_BASE_URL", "LLM_BASE_URL",
            default=DEFAULT_BASE_URL,
        ).rstrip("/"),
        "model": _first(env, "SANDBOX_LLM_MODEL", "LLM_MODEL"),
        "timeout_seconds": _first(
            env, "SANDBOX_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT", default="60",
        ),
    }
    runtime_allowlist = _first(env, "SANDBOX_LLM_ALLOWED_RESPONSE_MODELS")
    if not runtime_allowlist and not _first(env, "SANDBOX_LLM_MODEL"):
        runtime_allowlist = _first(env, "LLM_ALLOWED_RESPONSE_MODELS")
    runtime["allowed_response_models"] = _allowlist(
        runtime_allowlist, runtime["model"]
    )
    return {"generation": generation, "agent": agent, "runtime": runtime}
