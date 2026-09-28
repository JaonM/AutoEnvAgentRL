"""Fail-fast checks for a production material certification experiment."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from env_factory.evidence.material_attestation import (
    private_key_public_identity,
    public_key_identity,
)
from env_factory.evidence.material_consumer import BUNDLE_VERSION
from env_factory.evidence.material_artifacts import digest_json
from env_factory.evidence.data_governance import provider_identity
from env_factory.evidence.model_roles import resolve_model_roles


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def run_production_preflight(
    project: Path,
    output_parent: Path,
    *,
    signing_private_key: Path | None,
    trusted_public_key: Path | None,
    environment: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    minimum_free_bytes: int = 10 * 1024**3,
    experiment_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-secret readiness evidence without contacting model providers."""
    env = dict(os.environ if environment is None else environment)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    required_files = (
        "pyproject.toml", "uv.lock", "scripts/loop_experiment.py",
        "scripts/develop_sandbox_with_agent.sh",
    )
    missing_files = [name for name in required_files if not (project / name).is_file()]
    record("project_contract", not missing_files, {"missing": missing_files})

    executables = {name: which(name) is not None for name in ("codex", "docker", "openssl")}
    record("required_executables", all(executables.values()), executables)

    docker_ready = False
    docker_identity: dict[str, str] = {}
    if executables["docker"]:
        try:
            completed = runner(
                ["docker", "info", "--format", "{{.OSType}} {{.Architecture}}"],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            parts = completed.stdout.strip().split()
            docker_ready = completed.returncode == 0 and len(parts) == 2
            if docker_ready:
                docker_identity = {"os": parts[0], "architecture": parts[1]}
        except (OSError, subprocess.TimeoutExpired):
            docker_ready = False
    record("docker_daemon", docker_ready, docker_identity)

    roles = resolve_model_roles(env)
    generation = roles["generation"]
    agent = roles["agent"]
    runtime = roles["runtime"]
    model_configured = all(
        role["model"] and role["api_key"]
        for role in (generation, agent, runtime)
    )
    response_allowlists_valid = all(
        isinstance(role["allowed_response_models"], list)
        and bool(role["allowed_response_models"])
        and len(role["allowed_response_models"])
            == len(set(role["allowed_response_models"]))
        and all(
            isinstance(model, str) and bool(model.strip())
            for model in role["allowed_response_models"]
        )
        for role in (generation, agent, runtime)
    )
    urls_valid = all(
        _valid_url(role["base_url"])
        for role in (generation, agent, runtime)
    )
    generation_provider = provider_identity(
        generation["base_url"], generation["model"]
    )
    agent_provider = provider_identity(agent["base_url"], agent["model"])
    runtime_provider = provider_identity(runtime["base_url"], runtime["model"])
    record(
        "model_configuration",
        model_configured and urls_valid and response_allowlists_valid,
        {
        "generation_model_configured": bool(generation["model"]),
        "generation_key_configured": bool(generation["api_key"]),
        "agent_model_configured": bool(agent["model"]),
        "agent_key_configured": bool(agent["api_key"]),
        "runtime_model_configured": bool(runtime["model"]),
        "runtime_key_configured": bool(runtime["api_key"]),
        "generation_host": urlparse(generation["base_url"]).hostname,
        "agent_host": urlparse(agent["base_url"]).hostname,
        "runtime_host": urlparse(runtime["base_url"]).hostname,
        "generation_provider": generation_provider,
        "agent_provider": agent_provider,
        "runtime_provider": runtime_provider,
        "generation_allowed_response_models": generation[
            "allowed_response_models"
        ],
        "agent_allowed_response_models": agent["allowed_response_models"],
        "runtime_allowed_response_models": runtime["allowed_response_models"],
        "response_allowlists_valid": response_allowlists_valid,
        "urls_valid": urls_valid,
        },
    )
    provider_hosts_distinct = agent_provider["host"] != runtime_provider["host"]
    evaluator_models_distinct = agent_provider["model"] != runtime_provider["model"]
    evaluator_separated = provider_hosts_distinct and evaluator_models_distinct
    record(
        "evaluator_role_separation",
        evaluator_separated,
        {
            "agent_and_evaluator_distinct": evaluator_separated,
            **(
                {
                    "provider_hosts_distinct": provider_hosts_distinct,
                    "models_distinct": evaluator_models_distinct,
                }
                if experiment_config is not None else {}
            ),
        },
    )

    timeout_checks: dict[str, bool] = {}
    for name, role in roles.items():
        try:
            timeout_checks[name] = float(role["timeout_seconds"]) > 0
        except ValueError:
            timeout_checks[name] = False
    try:
        retries = int(env.get("SANDBOX_LLM_MAX_RETRIES") or "3")
        retries_valid = 0 <= retries <= 10
    except ValueError:
        retries_valid = False
    record("model_runtime_limits", all(timeout_checks.values()) and retries_valid, {
        "timeout_valid": timeout_checks,
        "retries_valid": retries_valid,
    })

    signing_ready = False
    key_identity = None
    if executables["openssl"]:
        try:
            private_identity = private_key_public_identity(signing_private_key)
            public_identity = public_key_identity(trusted_public_key)
            signing_ready = private_identity == public_identity
            key_identity = public_identity if signing_ready else None
        except (OSError, ValueError, AttributeError, TypeError):
            signing_ready = False
    record("bundle_signing_identity", signing_ready, {
        "key_identity_sha256": key_identity,
        "bundle_version": BUNDLE_VERSION,
    })

    free_bytes = 0
    try:
        free_bytes = int(disk_usage(output_parent).free)
    except OSError:
        pass
    record("workspace_capacity", free_bytes >= minimum_free_bytes, {
        "free_gib": round(free_bytes / 1024**3, 2),
        "minimum_gib": round(minimum_free_bytes / 1024**3, 2),
    })

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "version": "1.3" if experiment_config is not None else "1.0",
        "scope": "production_pre_training_material_experiment",
        **(
            {"experiment_config_sha256": digest_json(experiment_config)}
            if experiment_config is not None else {}
        ),
        "network_probe_performed": False,
        "ready": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


REQUIRED_CHECKS = {
    "project_contract",
    "required_executables",
    "docker_daemon",
    "model_configuration",
    "evaluator_role_separation",
    "model_runtime_limits",
    "bundle_signing_identity",
    "workspace_capacity",
}


def valid_production_preflight(
    value: Mapping[str, Any] | Any,
    *,
    expected_agent_provider: Mapping[str, Any] | None = None,
    expected_runtime_provider: Mapping[str, Any] | None = None,
    expected_generation_provider: Mapping[str, Any] | None = None,
    expected_signing_key_identity: str | None = None,
    expected_experiment_config_sha256: str | None = None,
    expected_bundle_version: str = BUNDLE_VERSION,
    expected_generation_response_models: list[str] | None = None,
    expected_agent_response_models: list[str] | None = None,
    expected_runtime_response_models: list[str] | None = None,
) -> bool:
    """Validate evidence produced by a fresh, trusted preflight execution."""
    if not isinstance(value, Mapping) or value.get("version") not in {
        "1.0", "1.1", "1.2", "1.3",
    }:
        return False
    config_digest = value.get("experiment_config_sha256")
    if value.get("version") in {"1.1", "1.2", "1.3"} and not (
        isinstance(config_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", config_digest) is not None
    ):
        return False
    if expected_experiment_config_sha256 is not None and not (
        value.get("version") in {"1.1", "1.2", "1.3"}
        and config_digest == expected_experiment_config_sha256
    ):
        return False
    if (
        value.get("scope") != "production_pre_training_material_experiment"
        or value.get("network_probe_performed") is not False
        or value.get("ready") is not True
        or value.get("failed_checks") != []
    ):
        return False
    checks = value.get("checks")
    if not isinstance(checks, list):
        return False
    by_name: dict[str, Mapping[str, Any]] = {}
    for check in checks:
        if not isinstance(check, Mapping) or not isinstance(check.get("name"), str):
            return False
        name = check["name"]
        if name in by_name:
            return False
        by_name[name] = check
    if not (
        set(by_name) == REQUIRED_CHECKS
        and all(check.get("passed") is True for check in by_name.values())
    ):
        return False
    separation_evidence = by_name["evaluator_role_separation"].get("evidence")
    expected_separation = (
        {
            "agent_and_evaluator_distinct": True,
            "provider_hosts_distinct": True,
            "models_distinct": True,
        }
        if value.get("version") == "1.3"
        else {"agent_and_evaluator_distinct": True}
    )
    if separation_evidence != expected_separation:
        return False
    if (
        expected_bundle_version == BUNDLE_VERSION
        and expected_experiment_config_sha256 is not None
        and value.get("version") != "1.3"
    ):
        return False
    signing_evidence = by_name["bundle_signing_identity"].get("evidence")
    if not (
        isinstance(signing_evidence, Mapping)
        and signing_evidence.get("bundle_version") == expected_bundle_version
        and isinstance(signing_evidence.get("key_identity_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", signing_evidence["key_identity_sha256"]
        ) is not None
        and (
            expected_signing_key_identity is None
            or signing_evidence["key_identity_sha256"]
                == expected_signing_key_identity
        )
    ):
        return False
    model_evidence = by_name["model_configuration"].get("evidence")
    if not isinstance(model_evidence, Mapping):
        return False
    allowlists = {
        "generation": model_evidence.get("generation_allowed_response_models"),
        "agent": model_evidence.get("agent_allowed_response_models"),
        "runtime": model_evidence.get("runtime_allowed_response_models"),
    }
    require_allowlists = value.get("version") in {"1.2", "1.3"} or any(
        expected is not None for expected in (
            expected_generation_response_models,
            expected_agent_response_models,
            expected_runtime_response_models,
        )
    )
    if require_allowlists and not (
        value.get("version") in {"1.2", "1.3"}
        and model_evidence.get("response_allowlists_valid") is True
        and all(
            isinstance(values, list) and bool(values)
            and len(values) == len(set(values))
            and all(isinstance(model, str) and model.strip() for model in values)
            for values in allowlists.values()
        )
    ):
        return False
    for role, expected in (
        ("generation", expected_generation_response_models),
        ("agent", expected_agent_response_models),
        ("runtime", expected_runtime_response_models),
    ):
        if expected is not None and allowlists[role] != expected:
            return False
    if (
        expected_generation_provider is not None
        and model_evidence.get("generation_provider")
            != dict(expected_generation_provider)
    ):
        return False
    if (
        expected_agent_provider is not None
        and model_evidence.get("agent_provider") != dict(expected_agent_provider)
    ):
        return False
    if (
        expected_runtime_provider is not None
        and model_evidence.get("runtime_provider") != dict(expected_runtime_provider)
    ):
        return False

    def valid_provider(provider: Any) -> bool:
        return (
            isinstance(provider, Mapping)
            and isinstance(provider.get("host"), str)
            and bool(provider["host"].strip())
            and isinstance(provider.get("model"), str)
            and bool(provider["model"].strip())
            and isinstance(provider.get("identity_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", provider["identity_sha256"])
            is not None
        )

    providers_valid = all(
        valid_provider(model_evidence.get(name))
        for name in (
            "generation_provider", "agent_provider", "runtime_provider"
        )
    )
    return (
        providers_valid
        and model_evidence["agent_provider"]["host"]
            != model_evidence["runtime_provider"]["host"]
        and model_evidence["agent_provider"]["model"]
            != model_evidence["runtime_provider"]["model"]
    )
