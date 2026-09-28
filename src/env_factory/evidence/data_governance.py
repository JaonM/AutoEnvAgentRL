"""Data-origin and outbound-payload governance for generated RL materials."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from env_factory.generation.dataset_formats import (
    ARCHIVE_EXTENSIONS, SUPPORTED_EXTENSIONS, SUPPORTED_SOURCE_EXTENSIONS, source_extension,
)


SYNTHETIC_ORIGIN = {
    "origin": "model_generated_synthetic",
    "contains_real_user_data": False,
    "intended_use": "agentic_rl_training_material",
}
PUBLIC_DATASET_HOSTS = {
    "kaggle": {"kaggle.com", "www.kaggle.com"},
    "heywhale": {"heywhale.com", "www.heywhale.com"},
}


def valid_data_origin(governance: Any, task: Mapping[str, Any]) -> bool:
    """Accept synthetic fixtures or attributed Kaggle/Heywhale source rows."""
    if not isinstance(governance, Mapping):
        return False
    if dict(governance) == SYNTHETIC_ORIGIN:
        return True
    if governance.get("origin") != "public_dataset":
        return False
    provider = governance.get("provider")
    url = governance.get("source_url")
    digest = governance.get("source_sha256")
    source_format = governance.get("source_format")
    member = governance.get("source_member")
    if (
        provider not in PUBLIC_DATASET_HOSTS
        or not isinstance(url, str)
        or urlparse(url).scheme != "https"
        or urlparse(url).hostname not in PUBLIC_DATASET_HOSTS[provider]
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or source_format not in SUPPORTED_SOURCE_EXTENSIONS
        or (source_format in ARCHIVE_EXTENSIONS and (
            not isinstance(member, str)
            or source_extension(member) not in SUPPORTED_EXTENSIONS
            or Path(member).is_absolute()
            or ".." in Path(member).parts
            or "\\" in member
        ))
        or (source_format not in ARCHIVE_EXTENSIONS and member is not None)
        or governance.get("contains_real_user_data") != "undetermined"
        or governance.get("intended_use") != "agentic_rl_training_material"
    ):
        return False
    source = task.get("artifacts", {}).get("dataset_source")
    return isinstance(source, Mapping) and dict(source) == {
        key: governance.get(key)
        for key in ("provider", "source_url", "source_sha256", "source_format", "source_member", "license")
    }
OUTBOUND_SURFACES = {
    "agent": [
        "public_input",
        "public_observation",
        "tool_schemas",
        "tool_results",
        "visible_conversation",
    ],
    "user_simulator": [
        "visible_conversation",
        "synthetic_user_profile",
        "fsm_current_state",
        "fsm_legal_transitions",
        "fsm_variables",
        "recovery_policy",
    ],
    "reward_evaluator": [
        "public_task",
        "metric_rubric",
        "declared_evaluation_inputs",
    ],
}
FORBIDDEN_OUTBOUND = [
    "credentials",
    "acceptance_contract",
    "hidden_truth",
    "future_user_turns",
    "trainer_authentication",
]
REQUIRED_OUTBOUND_SURFACES = {
    name: set(values) for name, values in OUTBOUND_SURFACES.items()
}

CREDENTIAL_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.I),
    "assigned_secret": re.compile(
        r"\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{12,}",
        re.I,
    ),
}

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "mainland_phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "mainland_id": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
}
CONTENT_IDENTITY = re.compile(r"(?:sha256:)?[0-9a-f]{64}", re.I)


def provider_identity(base_url: str, model: str) -> dict[str, str]:
    normalized = (base_url or "https://api.openai.com/v1").rstrip("/")
    return {
        "host": urlparse(normalized).hostname or "unknown",
        "model": model,
        "identity_sha256": hashlib.sha256(
            json.dumps([normalized, model], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def valid_provider_binding(
    rollout: Mapping[str, Any] | Any,
    calibration: Mapping[str, Any] | Any,
    governance: Mapping[str, Any] | Any,
) -> bool:
    """Bind all live-model evidence to one authorized provider declaration."""
    if not all(
        isinstance(value, Mapping)
        for value in (rollout, calibration, governance)
    ):
        return False
    providers = governance.get("providers")
    if not isinstance(providers, Mapping):
        return False
    agent = providers.get("agent")
    runtime = providers.get("user_simulator_and_reward")

    def valid_provider(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and isinstance(value.get("host"), str)
            and bool(value["host"].strip())
            and isinstance(value.get("model"), str)
            and bool(value["model"].strip())
            and isinstance(value.get("identity_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["identity_sha256"])
            is not None
        )

    return (
        valid_provider(agent)
        and valid_provider(runtime)
        and rollout.get("agent_model") == agent.get("model")
        and rollout.get("agent_provider_sha256") == agent.get("identity_sha256")
        and rollout.get("runtime_model") == runtime.get("model")
        and rollout.get("runtime_provider_sha256")
            == runtime.get("identity_sha256")
        and calibration.get("evaluator_provider") == runtime
    )


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def scan_payloads(payloads: Mapping[str, Any]) -> dict[str, Any]:
    credentials = []
    pii = []
    for label, payload in payloads.items():
        for path, text in _walk(payload, f"$.{label}"):
            for name, pattern in CREDENTIAL_PATTERNS.items():
                if pattern.search(text):
                    credentials.append({"kind": name, "path": path})
            # A complete content identity is opaque machine metadata. Random
            # digit runs inside it can resemble a phone or national ID, but
            # exempting only the full digest does not weaken free-text scans.
            if CONTENT_IDENTITY.fullmatch(text) is None:
                for name, pattern in PII_PATTERNS.items():
                    if pattern.search(text):
                        pii.append({"kind": name, "path": path})
    return {"credential_findings": credentials, "pii_findings": pii}


def load_json_or_lines(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def sandbox_payloads(root: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    metrics = task.get("metrics", [])
    reward_contract = []
    if isinstance(metrics, list):
        reward_contract = [
            {
                key: metric.get(key)
                for key in ("rubric", "criteria", "evaluation_inputs")
                if key in metric
            }
            for metric in metrics
            if isinstance(metric, Mapping)
        ]
    payloads: dict[str, Any] = {
        "public_task": task.get("task", ""),
        "public_input": task.get("public_input", {}),
        "tools": task.get("tools", []),
        "reward_contract": reward_contract,
    }
    data_root = root / "data"
    if data_root.is_dir():
        for path in sorted(data_root.rglob("*")):
            # Business tools may expose any fixture file. Scan every UTF-8
            # payload instead of maintaining a bypass-prone suffix allowlist.
            if path.is_file():
                payloads[f"data/{path.relative_to(data_root)}"] = load_json_or_lines(path)
    return payloads


def audit(
    root: Path,
    *,
    agent_base_url: str,
    agent_model: str,
    runtime_base_url: str,
    runtime_model: str,
) -> dict[str, Any]:
    task = json.loads((root / "task.json").read_text(encoding="utf-8"))
    data_manifest = task.get("artifacts", {}).get("data_manifest", {})
    governance = data_manifest.get("data_governance", {}) if isinstance(data_manifest, Mapping) else {}
    scan = scan_payloads(sandbox_payloads(root, task))
    origin_valid = valid_data_origin(governance, task)
    agent_provider = provider_identity(agent_base_url, agent_model)
    runtime_provider = provider_identity(runtime_base_url, runtime_model)
    providers_complete = all(
        provider["host"] != "unknown" and bool(provider["model"].strip())
        for provider in (agent_provider, runtime_provider)
    )
    eligible = (
        origin_valid
        and providers_complete
        and not scan["credential_findings"]
        and not scan["pii_findings"]
    )
    return {
        "version": "1.0",
        "eligible_for_external_model_processing": eligible,
        "data_origin": dict(governance) if isinstance(governance, Mapping) else {},
        "providers": {
            "agent": agent_provider,
            "user_simulator_and_reward": runtime_provider,
        },
        "outbound_surfaces": OUTBOUND_SURFACES,
        "forbidden_outbound": FORBIDDEN_OUTBOUND,
        **scan,
    }


def valid_governance_report(
    report: Mapping[str, Any] | Any,
    root: Path,
    task: Mapping[str, Any] | Any,
) -> bool:
    """Recompute the complete external-model data boundary from portable files."""
    if not isinstance(report, Mapping) or not isinstance(task, Mapping):
        return False
    declared = task.get("artifacts", {}).get("data_manifest", {}).get(
        "data_governance", {}
    )
    try:
        rescanned = scan_payloads(sandbox_payloads(root, task))
    except (
        OSError, UnicodeError, json.JSONDecodeError, TypeError, AttributeError,
    ):
        return False
    providers = report.get("providers")
    surfaces = report.get("outbound_surfaces")

    def valid_provider(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and isinstance(value.get("host"), str)
            and bool(value["host"].strip())
            and value["host"] != "unknown"
            and isinstance(value.get("model"), str)
            and bool(value["model"].strip())
            and isinstance(value.get("identity_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["identity_sha256"])
                is not None
        )

    return (
        report.get("version") == "1.0"
        and report.get("eligible_for_external_model_processing") is True
        and report.get("data_origin") == declared
        and valid_data_origin(declared, task)
        and report.get("credential_findings")
            == rescanned["credential_findings"] == []
        and report.get("pii_findings") == rescanned["pii_findings"] == []
        and isinstance(providers, Mapping)
        and all(valid_provider(providers.get(name)) for name in (
            "agent", "user_simulator_and_reward",
        ))
        and isinstance(surfaces, Mapping)
        and dict(surfaces) == OUTBOUND_SURFACES
        and report.get("forbidden_outbound") == FORBIDDEN_OUTBOUND
    )
