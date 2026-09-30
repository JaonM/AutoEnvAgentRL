"""Portable, non-secret description of one frozen production experiment."""

from __future__ import annotations

import re
from typing import Any, Mapping

from env_factory.evidence.material_artifacts import digest_json
from env_factory.evidence.portable_metadata import audit_portable_metadata, sanitize_portable_metadata


EXPERIMENT_CONTRACT_VERSION = "1.0"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _provider(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"host", "model", "identity_sha256"}
        and isinstance(value.get("host"), str)
        and bool(value["host"].strip())
        and isinstance(value.get("model"), str)
        and bool(value["model"].strip())
        and HEX64.fullmatch(str(value.get("identity_sha256", ""))) is not None
    )


def build_experiment_contract(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    source = dict(config) if isinstance(config, Mapping) else {}
    portable = sanitize_portable_metadata(source)
    return {
        "version": EXPERIMENT_CONTRACT_VERSION,
        "source_config_sha256": digest_json(source),
        "configuration_sha256": digest_json(portable),
        "configuration": portable,
    }


def valid_experiment_contract(
    value: Mapping[str, Any] | Any,
    *,
    expected_source_config_sha256: str | None = None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "version", "source_config_sha256", "configuration_sha256", "configuration",
    }:
        return False
    configuration = value.get("configuration")
    source_digest = value.get("source_config_sha256")
    if (
        value.get("version") != EXPERIMENT_CONTRACT_VERSION
        or not isinstance(configuration, Mapping)
        or HEX64.fullmatch(str(source_digest or "")) is None
        or (
            expected_source_config_sha256 is not None
            and source_digest != expected_source_config_sha256
        )
        or value.get("configuration_sha256") != digest_json(configuration)
        or sanitize_portable_metadata(configuration) != dict(configuration)
        or audit_portable_metadata(value)["safe"] is not True
    ):
        return False
    threshold = configuration.get("threshold")
    if not (
        configuration.get("certification_profile") == "production"
        and configuration.get("validation") == "live"
        and configuration.get("sandbox_runtime") == "docker"
        and _number(threshold)
        and 8.0 <= float(threshold) <= 10.0
        and _positive_integer(configuration.get("holdout_count"))
        and configuration["holdout_count"] >= 30
        and _positive_integer(configuration.get("holdout_batches"))
        and configuration["holdout_batches"] >= 3
        and _positive_integer(configuration.get("holdout_rollout_episodes"))
        and configuration["holdout_rollout_episodes"] >= 10
    ):
        return False
    for role in ("generation", "rollout", "runtime"):
        provider = configuration.get(f"{role}_provider")
        model = configuration.get(f"{role}_model")
        allowlist = configuration.get(f"{role}_allowed_response_models")
        if not (
            _provider(provider)
            and model == provider["model"]
            and isinstance(allowlist, list)
            and bool(allowlist)
            and len(allowlist) == len(set(allowlist))
            and all(isinstance(item, str) and item.strip() for item in allowlist)
        ):
            return False
    agent_provider = configuration["rollout_provider"]
    evaluator_provider = configuration["runtime_provider"]
    return (
        agent_provider["host"] != evaluator_provider["host"]
        and agent_provider["model"] != evaluator_provider["model"]
        and HEX64.fullmatch(str(
            configuration.get("bundle_attestation_key_identity_sha256", "")
        )) is not None
        and HEX64.fullmatch(str(configuration.get("source_digest", ""))) is not None
        and isinstance(configuration.get("input_digests"), list)
        and all(
            HEX64.fullmatch(str(item)) is not None
            for item in configuration["input_digests"]
        )
    )
