"""Canonical machine-readable contract for downstream RL material consumers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from env_factory.evidence.trajectory_schema import POLICY_TRANSITION_FIELDS, transition_errors


BUNDLE_MANIFEST = "bundle_manifest.json"
TRANSITIONS_FILE = "transitions.jsonl"
CERTIFICATION_FILE = "certification.json"
DATASET_CARD_FILE = "dataset_card.json"
CONSUMER_CONTRACT_FILE = "consumer_contract.json"
EXPERIMENT_CONTRACT_FILE = "experiment_contract.json"
BUNDLE_SIGNATURE_FILE = "bundle_manifest.sig"
BUNDLE_VERSION = "17.0"
DATASET_SPLITS = ("train", "validation", "test")
FEATURE_VERSIONS = {
    "splits": {"6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "families": {"7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "generation": {"8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "container_rollout": {"9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "reward_calibration": {"10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "provider_binding": {"11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "task_lineage": {"11.0", "12.0", "13.0", "14.0", "15.0", "16.0"},
    "production_preflight": {"12.0", "13.0", "14.0", "15.0", "16.0"},
    "experiment_binding": {"13.0", "14.0", "15.0", "16.0"},
    "trajectory_purpose": {"14.0", "15.0", "16.0"},
    "model_response_provenance": {"15.0", "16.0"},
    "model_response_authorization": {"16.0"},
    "consumer_record_validation": {"16.0"},
    "portable_build_context": {"16.0"},
    "certification_policy_binding": {"16.0"},
    "portable_experiment_contract": {"17.0"},
}
for _feature_versions in FEATURE_VERSIONS.values():
    if _feature_versions != {BUNDLE_VERSION}:
        _feature_versions.add(BUNDLE_VERSION)


def supports_bundle_feature(bundle_version: str, feature: str) -> bool:
    return bundle_version in FEATURE_VERSIONS.get(feature, set())


def assign_dataset_splits(items: list[dict[str, Any]]) -> dict[str, str]:
    """Deterministically stratify task identities; never split one task's episodes."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get("category", "unknown")), []).append(item)
    assignments = {}
    for members in groups.values():
        families: dict[str, list[dict[str, Any]]] = {}
        for item in members:
            family = str(item.get("task_family_id") or item.get("item_id", ""))
            families.setdefault(family, []).append(item)
        ordered = sorted(families.items())
        count = len(ordered)
        validation_count = max(1, count // 10) if count >= 3 else 0
        test_count = max(1, count // 10) if count >= 3 else 0
        train_end = count - validation_count - test_count
        for index, (_, family_members) in enumerate(ordered):
            split = (
                "train" if index < train_end
                else "validation" if index < count - test_count
                else "test"
            )
            for item in family_members:
                assignments[str(item["item_id"])] = split
    return assignments


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def consumer_record_errors(
    record: Any, bundle_version: str = BUNDLE_VERSION
) -> list[str]:
    """Validate one portable record independently of its source rollout."""
    if not isinstance(record, Mapping):
        return ["record:not_object"]
    contract = consumer_contract(bundle_version)["records"]
    required = set(contract["required_fields"])
    errors: list[str] = []
    keys = set(record)
    if keys != required:
        errors.append("record:fields")

    def nonempty_string(name: str) -> None:
        if not isinstance(record.get(name), str) or not record[name]:
            errors.append(f"record:{name}")

    if record.get("schema_version") != "2.0":
        errors.append("record:schema_version")
    for name, pattern in (
        ("item_id", r"[0-9a-f]{16}-[0-9a-f]{16}"),
        ("task_sha256", r"[0-9a-f]{64}"),
    ):
        value = record.get(name)
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            errors.append(f"record:{name}")
    for name in ("category", "episode_termination", "agent_model", "runtime_model"):
        nonempty_string(name)
    for name in ("episode_index", "episode_seed"):
        value = record.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or (
            name == "episode_index" and value < 0
        ):
            errors.append(f"record:{name}")
    if not isinstance(record.get("episode_success"), bool):
        errors.append("record:episode_success")
    for name in ("episode_initial_reward", "episode_final_reward"):
        if not _finite_number(record.get(name)):
            errors.append(f"record:{name}")
    if not isinstance(record.get("agent_usage"), Mapping):
        errors.append("record:agent_usage")
    if supports_bundle_feature(bundle_version, "splits") and record.get(
        "split"
    ) not in DATASET_SPLITS:
        errors.append("record:split")
    if supports_bundle_feature(bundle_version, "families"):
        value = record.get("task_family_id")
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append("record:task_family_id")
    if supports_bundle_feature(bundle_version, "generation"):
        nonempty_string("generation_model")
        provider = record.get("generation_provider_identity_sha256")
        if not isinstance(provider, str) or re.fullmatch(
            r"[0-9a-f]{64}", provider
        ) is None:
            errors.append("record:generation_provider_identity_sha256")
        seed = record.get("generation_sample_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            errors.append("record:generation_sample_seed")
    if supports_bundle_feature(bundle_version, "trajectory_purpose"):
        if record.get("trajectory_role") != "certification_evidence":
            errors.append("record:trajectory_role")
        if record.get("direct_training_status") != "not_certified":
            errors.append("record:direct_training_status")

    transition = record.get("transition")
    if not isinstance(transition, Mapping):
        errors.append("record:transition")
        return errors
    if set(transition) != set(POLICY_TRANSITION_FIELDS):
        errors.append("transition:fields")
    step = transition.get("step")
    valid_step = (
        isinstance(step, int) and not isinstance(step, bool) and step >= 0
    )
    if not valid_step:
        errors.append("transition:step")
    expected_step = step if valid_step else -1
    errors.extend(transition_errors(transition, expected_step=expected_step))
    messages = transition.get("agent_input")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if isinstance(message, Mapping) and set(message) != {"role", "content"}:
                errors.append(f"transition:agent_input[{index}]:fields")
    return sorted(set(errors))


def consumer_contract(bundle_version: str = BUNDLE_VERSION) -> dict[str, Any]:
    """Return the exact portable handoff; consumers need no prompt conventions."""
    supports_splits = supports_bundle_feature(bundle_version, "splits")
    supports_families = supports_bundle_feature(bundle_version, "families")
    supports_generation = supports_bundle_feature(bundle_version, "generation")
    supports_lineage = supports_bundle_feature(bundle_version, "task_lineage")
    supports_trajectory_purpose = supports_bundle_feature(
        bundle_version, "trajectory_purpose"
    )
    supports_response_provenance = supports_bundle_feature(
        bundle_version, "model_response_provenance"
    )
    record_fields = [
        "schema_version", "item_id", "task_sha256", "category",
        "episode_index", "episode_seed", "episode_success",
        "episode_termination", "episode_initial_reward", "episode_final_reward",
        "agent_model", "runtime_model", "agent_usage",
        *(
            ["trajectory_role", "direct_training_status"]
            if supports_trajectory_purpose else []
        ),
        *(["split"] if supports_splits else []),
        *(["task_family_id"] if supports_families else []),
        *([] if not supports_generation else [
            "generation_model", "generation_provider_identity_sha256",
            "generation_sample_seed",
        ]), "transition",
    ]
    return {
        "version": "1.0",
        "kind": "agentic_rl_training_material_consumer_contract",
        "bundle_version": bundle_version,
        **({
            "experiment_contract": EXPERIMENT_CONTRACT_FILE,
        } if supports_bundle_feature(
            bundle_version, "portable_experiment_contract"
        ) else {}),
        "records": {
            "path": TRANSITIONS_FILE,
            "media_type": "application/x-ndjson",
            "encoding": "utf-8",
            "ordering": ["item_id", "episode_index", "transition.step"],
            "identity": ["item_id", "episode_index", "transition.step"],
            **({"split_unit": "item_id"} if supports_splits else {}),
            **({"near_duplicate_split_unit": "task_family_id"} if supports_families else {}),
            "required_fields": record_fields,
            "policy_transition_fields": list(POLICY_TRANSITION_FIELDS),
            "json_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": record_fields,
                "properties": {
                    "schema_version": {"const": "2.0"},
                    "item_id": {"type": "string", "pattern": "^[0-9a-f]{16}-[0-9a-f]{16}$"},
                    "task_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "category": {"type": "string", "minLength": 1},
                    **(
                        {"split": {"enum": list(DATASET_SPLITS)}}
                        if supports_splits else {}
                    ),
                    **(
                        {"task_family_id": {
                            "type": "string", "pattern": "^[0-9a-f]{64}$",
                        }} if supports_families else {}
                    ),
                    **({
                        "generation_model": {"type": "string", "minLength": 1},
                        "generation_provider_identity_sha256": {
                            "type": "string", "pattern": "^[0-9a-f]{64}$",
                        },
                        "generation_sample_seed": {"type": "integer"},
                    } if supports_generation else {}),
                    "episode_index": {"type": "integer", "minimum": 0},
                    "episode_seed": {"type": "integer"},
                    "episode_success": {"type": "boolean"},
                    "episode_termination": {"type": "string", "minLength": 1},
                    "episode_initial_reward": {"type": "number"},
                    "episode_final_reward": {"type": "number"},
                    "agent_model": {"type": "string", "minLength": 1},
                    "runtime_model": {"type": "string", "minLength": 1},
                    "agent_usage": {"type": "object"},
                    **({
                        "trajectory_role": {"const": "certification_evidence"},
                        "direct_training_status": {"const": "not_certified"},
                    } if supports_trajectory_purpose else {}),
                    "transition": {
                        "type": "object",
                        "required": list(POLICY_TRANSITION_FIELDS),
                        "properties": {
                            "step": {"type": "integer", "minimum": 0},
                            "agent_input": {
                                "type": "array", "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "required": ["role", "content"],
                                    "properties": {
                                        "role": {"enum": ["system", "user", "assistant"]},
                                        "content": {"type": "string"},
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "assistant_output": {"type": "string"},
                            "observation": {"type": "object"},
                            "action": {"type": ["object", "null"]},
                            "result": {"type": "object"},
                            "next_observation": {"type": "object"},
                            "reward": {"type": "number"},
                            "terminated": {"type": "boolean"},
                            "truncated": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "environments": {
            "index": f"{BUNDLE_MANIFEST}#items",
            "root_template": "environments/{item_id}",
            "task_contract": "environments/{item_id}/task.json",
            **({
                "task_lineage": "environments/{item_id}/task_lineage.json",
            } if supports_lineage else {}),
            "docker_build_context": "environments/{item_id}",
            "dockerfile": "environments/{item_id}/Dockerfile",
            **({
                "dockerignore": "environments/{item_id}/.dockerignore",
            } if supports_bundle_feature(
                bundle_version, "portable_build_context"
            ) else {}),
            "runtime_interface_source": "task.json#requirements.runtime_interface",
        },
        "visibility": {
            "policy_input": ["transition.agent_input", "transition.observation"],
            "policy_output": ["transition.assistant_output", "transition.action"],
            "environment_feedback": [
                "transition.result", "transition.next_observation", "transition.reward",
                "transition.terminated", "transition.truncated",
            ],
            "trainer_only": "environments/{item_id}/live_rollout.json",
            "forbid_policy_ingestion": [
                "trainer_metadata", "trajectory", "replay", "initial_state", "final_state",
            ],
        },
        **({
            "material_roles": {
                "environments": {
                    "status": "certified_for_fresh_rollout_collection",
                    "entrypoint": "environments/{item_id}/Dockerfile",
                },
                "exported_trajectories": {
                    "role": "certification_evidence",
                    "direct_policy_optimization": "not_certified",
                    "requires_downstream_approval": True,
                    **({
                        "actual_model_identity_source": (
                            "agent_usage.response_model_and_runtime_replay"
                        ),
                    } if supports_response_provenance else {}),
                },
            },
        } if supports_trajectory_purpose else {}),
        "invariants": [
            "records_are_ordered_and_unique_by_identity",
            "assistant_output_parses_to_action",
            "next_observation_equals_following_observation_within_episode",
            "only_final_transition_is_terminal_or_truncated",
            "final_transition_reward_equals_episode_final_reward",
            "trainer_only_fields_are_not_policy_inputs",
            *(
                ["every_record_validates_against_consumer_contract"]
                if supports_bundle_feature(
                    bundle_version, "consumer_record_validation"
                ) else []
            ),
            *(
                ["acceptance_rollouts_are_not_certified_training_targets"]
                if supports_trajectory_purpose else []
            ),
            *(["generated_task_equals_runtime_task"] if supports_lineage else []),
            *(
                ["portable_inventory_covers_complete_docker_build_context"]
                if supports_bundle_feature(
                    bundle_version, "portable_build_context"
                ) else []
            ),
            *(
                ["frozen_experiment_configuration_is_portable_and_bound"]
                if supports_bundle_feature(
                    bundle_version, "portable_experiment_contract"
                ) else []
            ),
        ],
    }
