#!/usr/bin/env python3
"""Export a certified EnvFactory result as a portable, verified RL-material bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from env_factory.evidence.material_artifacts import (
    MATERIAL_MANIFEST_VERSION,
    digest_json,
    docker_context_errors,
    portable_artifact_digests,
)
from env_factory.evidence.material_consumer import (
    BUNDLE_MANIFEST,
    BUNDLE_SIGNATURE_FILE,
    BUNDLE_VERSION,
    CERTIFICATION_FILE,
    CONSUMER_CONTRACT_FILE,
    DATASET_SPLITS,
    DATASET_CARD_FILE,
    EXPERIMENT_CONTRACT_FILE,
    TRANSITIONS_FILE,
    assign_dataset_splits,
    consumer_contract,
    consumer_record_errors,
    supports_bundle_feature,
)
from env_factory.evidence.material_attestation import (
    sign_file,
    signed_metadata,
    verify_file,
)
from env_factory.evidence.model_response_provenance import response_provenance
from env_factory.evidence.material_privacy import audit_rollout_privacy
from env_factory.evidence.trajectory_schema import episode_errors, policy_transition
from env_factory.evidence.execution_provenance import valid_execution_provenance
from env_factory.tasks.task_similarity import task_family_ids
from env_factory.evidence.generation_provenance import valid_generation_provenance
from env_factory.evidence.runtime_provenance import valid_container_rollout_execution
from env_factory.evidence.data_governance import (
    valid_governance_report,
    valid_provider_binding,
)
from env_factory.evidence.certification_policy import valid_certification_policy
from env_factory.evidence.experiment_contract import valid_experiment_contract
from env_factory.tasks.task_portability import valid_task_lineage
from env_factory.evidence.production_preflight import (
    REQUIRED_CHECKS,
    valid_production_preflight,
)
from env_factory.evidence.portable_metadata import (
    audit_portable_metadata,
    sanitize_portable_metadata,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe material path: {value}")
    return path


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    if file_sha256(source) != expected:
        raise ValueError(f"source digest changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected:
        raise ValueError(f"copied digest mismatch: {destination}")


def _portable_certification(certification: Mapping[str, Any]) -> dict[str, Any]:
    """Keep release claims and measurements without machine-local artifact paths."""
    source = certification.get("materials_manifest", {})
    return {
        "version": "1.0",
        "certification": certification.get("certification"),
        "scope": certification.get("scope"),
        "certified": certification.get("certified"),
        "does_not_certify": certification.get("does_not_certify", []),
        "policy": certification.get("policy", {}),
        "measurements": sanitize_portable_metadata(
            certification.get("measurements", {})
        ),
        "gates": certification.get("gates", {}),
        "failed_gates": certification.get("failed_gates", []),
        "source_dataset_sha256": source.get("dataset_sha256"),
        "evaluator_source_digest": source.get("evaluator_source_digest"),
        "experiment_config_sha256": source.get("experiment_config_sha256"),
        "experiment_contract_sha256": source.get(
            "experiment_contract_sha256"
        ),
        "execution_provenance": source.get("execution_provenance"),
    }


def _dataset_card(
    certification: Mapping[str, Any],
    *,
    source_dataset_sha256: Any,
    items: list[Mapping[str, Any]],
    transition_count: int,
    model_pairs: Counter[tuple[str, str]],
    successful_episodes: int,
    episode_count: int,
    data_origins: Counter[str] | None = None,
) -> dict[str, Any]:
    data_origins = data_origins or Counter({"model_generated_synthetic": len(items)})
    categories = Counter(str(item.get("category")) for item in items)
    splits = Counter(str(item.get("split")) for item in items)
    category_splits = Counter(
        (str(item.get("category")), str(item.get("split"))) for item in items
    )
    family_count = len({str(item.get("task_family_id")) for item in items})
    generation_configured_models = Counter(
        str(item.get("generation_provenance", {}).get("generator_provider", {}).get("model"))
        for item in items
    )
    generation_actual_models = Counter()
    for item in items:
        for attempt in item.get("generation_provenance", {}).get("attempts", []):
            generation_actual_models.update(attempt.get("llm_trace", {}).get("models", {}))
    generation_providers = Counter(
        str(item.get("generation_provenance", {}).get("generator_provider", {}).get(
            "identity_sha256"
        ))
        for item in items
    )
    generation_attempts = sum(
        len(item.get("generation_provenance", {}).get("attempts", [])) for item in items
    )
    generation_responses = sum(
        attempt.get("llm_trace", {}).get("responses", 0)
        for item in items
        for attempt in item.get("generation_provenance", {}).get("attempts", [])
    )
    agent_response_models: Counter[str] = Counter()
    runtime_response_models: Counter[str] = Counter()
    for item in items:
        provenance = item.get("model_response_provenance", {})
        agent_response_models.update(provenance.get("agent_response_models", {}))
        runtime_response_models.update(provenance.get("runtime_response_models", {}))
    runtime_modes = Counter(
        str(item.get("runtime_execution", {}).get("mode")) for item in items
    )
    container_images = {
        str(item.get("runtime_execution", {}).get("container_image_id"))
        for item in items
        if item.get("runtime_execution", {}).get("mode") == "docker_http"
    }
    reward_runtime_modes = Counter(
        str(item.get("reward_runtime_execution", {}).get("mode")) for item in items
    )
    reward_container_images = {
        str(item.get("reward_runtime_execution", {}).get("container_image_id"))
        for item in items
        if item.get("reward_runtime_execution", {}).get("mode") == "docker_http"
    }
    same_model_items = sum(
        count for (agent, runtime), count in model_pairs.items() if agent == runtime
    )
    limitations = list(certification.get("does_not_certify", []))
    limitations.extend([
        "offline_rl_algorithm_compatibility",
        "data_license_or_distribution_rights",
        "absence_of_same_model_evaluation_bias",
    ])
    return {
        "version": "2.3",
        "kind": "agentic_rl_pretraining_material_dataset_card",
        "source_dataset_sha256": source_dataset_sha256,
        "certification": {
            "name": certification.get("certification"),
            "scope": certification.get("scope"),
            "certified": certification.get("certified") is True,
            "policy_sha256": certification.get("materials_manifest", {}).get(
                "certification_policy_sha256"
            ),
            "experiment_contract_sha256": certification.get(
                "materials_manifest", {}
            ).get("experiment_contract_sha256"),
        },
        "intended_uses": [
            "reconstruct_agentic_sandbox_environments",
            "validate_rl_data_adapters",
            "collect_fresh_on_policy_rollouts",
            "prepare_policy_visible_transition_inputs",
        ],
        "prohibited_interpretations": sorted(set(limitations)),
        "distribution_status": "internal_only_until_legal_and_security_review",
        "license_status": "not_asserted_by_envfactory",
        "consumer_contract": CONSUMER_CONTRACT_FILE,
        "build_environment": certification.get("materials_manifest", {}).get(
            "execution_provenance"
        ),
        "composition": {
            "items": len(items),
            "transitions": transition_count,
            "episodes": episode_count,
            "successful_episodes": successful_episodes,
            "failed_episodes": episode_count - successful_episodes,
            "categories": dict(sorted(categories.items())),
            "model_pairs": [
                {"agent_model": pair[0], "runtime_model": pair[1], "items": count}
                for pair, count in sorted(model_pairs.items())
            ],
            "same_model_pair_items": same_model_items,
            "splits": {name: splits.get(name, 0) for name in DATASET_SPLITS},
            "category_splits": {
                category: {
                    split: category_splits.get((category, split), 0)
                    for split in DATASET_SPLITS
                }
                for category in sorted(categories)
            },
            "task_families": family_count,
            "near_duplicate_items": len(items) - family_count,
            "cross_split_family_overlap": 0,
            "task_generation": {
                "configured_models": dict(sorted(generation_configured_models.items())),
                "actual_response_models": dict(sorted(generation_actual_models.items())),
                "provider_identities": dict(sorted(generation_providers.items())),
                "attempts": generation_attempts,
                "responses": generation_responses,
            },
            "rollout_response_provenance": {
                "agent_models": dict(sorted(agent_response_models.items())),
                "runtime_models": dict(sorted(runtime_response_models.items())),
            },
            "model_response_authorization": certification.get(
                "measurements", {}
            ).get("model_response_authorization", {}),
            "runtime_execution": {
                "modes": dict(sorted(runtime_modes.items())),
                "validated_container_items": runtime_modes.get("docker_http", 0),
                "unique_container_images": len(container_images),
            },
            "reward_calibration_execution": {
                "modes": dict(sorted(reward_runtime_modes.items())),
                "validated_container_items": reward_runtime_modes.get(
                    "docker_http", 0
                ),
                "unique_container_images": len(reward_container_images),
            },
        },
        "data_boundary": {
            "origin": next(iter(data_origins)) if len(data_origins) == 1 else "mixed_verified_origins",
            "origins": dict(sorted(data_origins.items())),
            "contains_real_user_data": (
                False if set(data_origins) == {"model_generated_synthetic"}
                else "undetermined"
            ),
            "policy_transitions": TRANSITIONS_FILE,
            "trainer_only_evidence": "environments/*/live_rollout.json",
            "visibility_contract": "bundle_manifest.json#transition_visibility",
            "exported_trajectory_role": "certification_evidence",
            "direct_training_status": "not_certified",
        },
        "consumer_requirements": [
            "verify_bundle_before_use",
            "do_not_feed_trainer_only_evidence_to_the_policy",
            "use_fresh_rollouts_for_algorithms_requiring_on_policy_data",
            "perform_organization_specific_legal_security_and_model_risk_review",
        ],
    }


def _transition_records(
    item_id: str, item: Mapping[str, Any], rollout: Mapping[str, Any], *,
    split: str = "", task_family_id: str = "",
    generation_provenance: Mapping[str, Any] | None = None,
    bundle_version: str = BUNDLE_VERSION,
) -> list[dict[str, Any]]:
    if rollout.get("schema_version") != "2.0":
        raise ValueError("rollout schema_version must be 2.0")
    if not all(
        isinstance(rollout.get(name), str) and rollout[name]
        for name in ("agent_model", "runtime_model")
    ):
        raise ValueError("rollout model provenance is incomplete")
    privacy = audit_rollout_privacy(rollout)
    if privacy.get("eligible_for_policy_training_export") is not True:
        raise ValueError(f"rollout material payload is unsafe: {privacy}")
    episodes = rollout.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("rollout episodes must be non-empty")
    records = []
    if generation_provenance is not None and not valid_generation_provenance(
        generation_provenance
    ):
        raise ValueError("task generation provenance is invalid")
    for episode_index, episode in enumerate(episodes):
        errors = episode_errors(episode)
        if errors:
            raise ValueError(
                f"rollout episode {episode_index} violates trajectory schema: {errors}"
            )
        for transition_index, transition in enumerate(episode["transitions"]):
            records.append({
                "schema_version": "2.0",
                "item_id": item_id,
                "task_sha256": item["task_sha256"],
                "category": item["category"],
                "episode_index": episode_index,
                "episode_seed": episode.get("seed"),
                "episode_success": episode.get("agent_success"),
                "episode_termination": episode.get("termination"),
                "episode_initial_reward": episode.get("initial_reward"),
                "episode_final_reward": episode.get("final_reward"),
                "agent_model": rollout.get("agent_model"),
                "runtime_model": rollout.get("runtime_model"),
                "agent_usage": episode["usage"][transition_index],
                **({
                    "trajectory_role": "certification_evidence",
                    "direct_training_status": "not_certified",
                } if supports_bundle_feature(
                    bundle_version, "trajectory_purpose"
                ) else {}),
                **({"split": split} if split else {}),
                **({"task_family_id": task_family_id} if task_family_id else {}),
                **({
                    "generation_model": generation_provenance["generator_provider"]["model"],
                    "generation_provider_identity_sha256": generation_provenance[
                        "generator_provider"
                    ]["identity_sha256"],
                    "generation_sample_seed": generation_provenance["sample_seed"],
                } if generation_provenance is not None else {}),
                "transition": policy_transition(transition),
            })
    return records


def _export_bundle_uncommitted(
    certification: Mapping[str, Any], output: Path, project: Path,
    *, signing_private_key: Path | None = None, trusted_public_key: Path | None = None,
) -> dict[str, Any]:
    if (signing_private_key is None) != (trusted_public_key is None):
        raise ValueError("signing private key and trusted public key are both required")
    if certification.get("certified") is not True:
        raise ValueError("only a certified production report can be exported")
    if certification.get("material_verification", {}).get("verified") is not True:
        raise ValueError("source material verification is required before export")
    attestation = (
        signed_metadata(
            signing_private_key, trusted_public_key, BUNDLE_SIGNATURE_FILE
        )
        if signing_private_key is not None and trusted_public_key is not None
        else {"version": "1.0", "status": "unsigned"}
    )
    preflight = certification.get("measurements", {}).get(
        "production_preflight"
    )
    source_manifest = certification.get("materials_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("version") != MATERIAL_MANIFEST_VERSION
    ):
        raise ValueError(
            f"a v{MATERIAL_MANIFEST_VERSION.removesuffix('.0')} materials manifest is required"
        )
    experiment_config_sha256 = source_manifest.get("experiment_config_sha256")
    if not (
        isinstance(experiment_config_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", experiment_config_sha256) is not None
    ):
        raise ValueError("source manifest lacks a valid experiment configuration digest")
    experiment_contract = source_manifest.get("experiment_contract")
    experiment_contract_sha256 = source_manifest.get(
        "experiment_contract_sha256"
    )
    if not (
        valid_experiment_contract(
            experiment_contract,
            expected_source_config_sha256=experiment_config_sha256,
        )
        and experiment_contract_sha256 == digest_json(experiment_contract)
        and certification.get("gates", {}).get(
            "portable_experiment_contract"
        ) is True
        and certification.get("measurements", {}).get(
            "experiment_contract"
        ) == {
            "version": experiment_contract["version"],
            "sha256": experiment_contract_sha256,
            "verified": True,
        }
    ):
        raise ValueError("portable experiment contract is invalid or unbound")
    certification_policy = certification.get("policy")
    certification_policy_sha256 = source_manifest.get(
        "certification_policy_sha256"
    )
    if not (
        valid_certification_policy(certification_policy)
        and source_manifest.get("certification_policy") == certification_policy
        and certification_policy_sha256 == digest_json(certification_policy)
        and certification.get("gates", {}).get("certification_policy") is True
        and certification.get("measurements", {}).get(
            "certification_policy"
        ) == {
            "version": certification_policy["version"],
            "sha256": certification_policy_sha256,
            "verified": True,
        }
    ):
        raise ValueError("certification policy is not canonical or bound")
    expected_key_identity = (
        attestation.get("key_identity_sha256")
        if attestation.get("status") == "signed" else None
    )
    if not valid_production_preflight(
        preflight,
        expected_signing_key_identity=expected_key_identity,
        expected_experiment_config_sha256=experiment_config_sha256,
    ):
        raise ValueError(
            "production preflight does not match bundle signing identity"
        )
    try:
        from verify_training_materials import verify
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "verify_training_materials", Path(__file__).with_name("verify_training_materials.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load material verifier")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify = module.verify
    source_verification = verify(source_manifest, project)
    if source_verification.get("verified") is not True:
        raise ValueError(f"source material verification failed: {source_verification['failed_gates']}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bundle output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    transition_path = output / TRANSITIONS_FILE
    exported_items = []
    seen_item_ids = set()
    transition_count = 0
    episode_count = 0
    successful_episodes = 0
    model_pairs: Counter[tuple[str, str]] = Counter()
    source_items = source_manifest.get("items", [])
    if not isinstance(source_items, list):
        raise ValueError("material items must be a list")
    ordered_source_items = sorted(
        source_items,
        key=lambda value: (
            str(value.get("task_sha256", "")),
            str(value.get("sandbox_evidence_fingerprint", "")),
        ) if isinstance(value, Mapping) else ("", ""),
    )
    family_inputs = []
    for item in ordered_source_items:
        if not isinstance(item, Mapping):
            continue
        item_id = (
            f"{item['task_sha256'][:16]}-"
            f"{item['sandbox_evidence_fingerprint'][:16]}"
        )
        task_document = json.loads(
            Path(str(item["task_path"])).read_text(encoding="utf-8")
        )
        family_inputs.append({
            "item_id": item_id,
            "task": task_document.get("task") if isinstance(task_document, Mapping) else None,
        })
    family_assignments = task_family_ids(family_inputs)
    split_assignments = assign_dataset_splits([
        {
            "item_id": (
                f"{item['task_sha256'][:16]}-"
                f"{item['sandbox_evidence_fingerprint'][:16]}"
            ),
            "category": str(item.get("category", "unknown")),
            "task_family_id": family_assignments[
                f"{item['task_sha256'][:16]}-"
                f"{item['sandbox_evidence_fingerprint'][:16]}"
            ],
        }
        for item in ordered_source_items if isinstance(item, Mapping)
    ])
    with transition_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in ordered_source_items:
            if not isinstance(item, Mapping):
                raise ValueError("material item is not an object")
            item_id = f"{item['task_sha256'][:16]}-{item['sandbox_evidence_fingerprint'][:16]}"
            if item_id in seen_item_ids:
                raise ValueError(f"duplicate portable item identity: {item_id}")
            seen_item_ids.add(item_id)
            task_family_id = family_assignments[item_id]
            split = split_assignments[item_id]
            generation = item.get("generation_provenance")
            if (
                not valid_generation_provenance(generation)
                or generation.get("training_category") != item.get("category")
                or generation.get("task_sha256") != item.get("task_sha256")
            ):
                raise ValueError(f"invalid task generation provenance for {item_id}")
            source_root = Path(str(item["sandbox_root"]))
            destination_root = output / "environments" / item_id
            try:
                task_lineage = json.loads(
                    (source_root / "task_lineage.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                task_lineage = {}
            if not valid_task_lineage(
                task_lineage,
                Path(str(item["task_path"])),
                source_root / "task.json",
            ):
                raise ValueError(f"task lineage is invalid for {item_id}")
            copied = {}
            for group in ("sandbox_artifacts_sha256", "sandbox_evidence_sha256"):
                values = item.get(group)
                if not isinstance(values, Mapping) or not values:
                    raise ValueError(f"material item lacks {group}")
                for relative, expected in values.items():
                    safe = _safe_relative(str(relative))
                    destination = destination_root / safe
                    if str(safe) in copied and copied[str(safe)] != str(expected):
                        raise ValueError(f"conflicting digest for {safe}")
                    _copy_verified(source_root / safe, destination, str(expected))
                    copied[str(safe)] = str(expected)
            rollout_source = source_root / "live_rollout.json"
            rollout = json.loads(rollout_source.read_text(encoding="utf-8"))
            if digest_json(rollout) != item.get("rollout_sha256"):
                raise ValueError(f"rollout changed for {item_id}")
            if not valid_container_rollout_execution(rollout, source_root):
                raise ValueError(
                    f"rollout did not execute in the validated container for {item_id}"
                )
            rollout_response_provenance = response_provenance(rollout)
            if rollout_response_provenance["verified"] is not True:
                raise ValueError(
                    f"model response provenance is invalid for {item_id}"
                )
            calibration = json.loads(
                (source_root / "agentic_training_value_live.json").read_text(
                    encoding="utf-8"
                )
            )
            governance = json.loads(
                (source_root / "data_governance.json").read_text(
                    encoding="utf-8"
                )
            )
            if not (
                calibration.get("curriculum_training_ready") is True
                and calibration.get("validation_mode") == "live_evaluator"
                and valid_container_rollout_execution(calibration, source_root)
                and valid_provider_binding(rollout, calibration, governance)
            ):
                raise ValueError(
                    "reward calibration/provider authorization is invalid for "
                    f"{item_id}"
                )
            rollout_destination = destination_root / "live_rollout.json"
            rollout_destination.write_text(
                json.dumps(rollout, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
            copied["live_rollout.json"] = file_sha256(rollout_destination)
            records = _transition_records(
                item_id, item, rollout, split=split,
                task_family_id=task_family_id,
                generation_provenance=generation,
            )
            for record in records:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ) + "\n")
            transition_count += len(records)
            episodes = rollout.get("episodes", [])
            episode_count += len(episodes)
            successful_episodes += sum(
                episode.get("agent_success") is True
                for episode in episodes if isinstance(episode, Mapping)
            )
            model_pairs[(
                str(rollout.get("agent_model", "")),
                str(rollout.get("runtime_model", "")),
            )] += 1
            exported_items.append({
                "item_id": item_id,
                "category": item["category"],
                "split": split,
                "task_family_id": task_family_id,
                "generation_provenance": generation,
                "runtime_execution": rollout["runtime_execution"],
                "model_response_provenance": rollout_response_provenance,
                "reward_runtime_execution": calibration["runtime_execution"],
                "score": item["score"],
                "task_sha256": item["task_sha256"],
                "environment_path": f"environments/{item_id}",
                "episode_count": item["episode_count"],
                "transition_count": len(records),
                "files_sha256": copied,
            })

    data_origins = Counter()
    for item in exported_items:
        task_path = output / item["environment_path"] / "task.json"
        task_document = json.loads(task_path.read_text(encoding="utf-8"))
        data_origins[task_document["artifacts"]["data_manifest"]["data_governance"]["origin"]] += 1
    portable_certification = _portable_certification(certification)
    portable_metadata_privacy = audit_portable_metadata(portable_certification)
    if not portable_metadata_privacy["safe"]:
        raise ValueError(
            "portable certification contains non-portable or sensitive metadata: "
            f"{portable_metadata_privacy}"
        )
    (output / EXPERIMENT_CONTRACT_FILE).write_text(
        json.dumps(experiment_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    (output / CERTIFICATION_FILE).write_text(
        json.dumps(portable_certification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    card = _dataset_card(
        certification,
        source_dataset_sha256=source_manifest.get("dataset_sha256"),
        items=exported_items,
        transition_count=transition_count,
        model_pairs=model_pairs,
        successful_episodes=successful_episodes,
        episode_count=episode_count,
        data_origins=data_origins,
    )
    (output / DATASET_CARD_FILE).write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    (output / CONSUMER_CONTRACT_FILE).write_text(
        json.dumps(consumer_contract(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )

    files = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != BUNDLE_MANIFEST
    }
    manifest = {
        "version": BUNDLE_VERSION,
        "kind": "portable_agentic_rl_training_materials",
        "source_dataset_sha256": source_manifest.get("dataset_sha256"),
        "experiment_config_sha256": experiment_config_sha256,
        "experiment_contract_file": EXPERIMENT_CONTRACT_FILE,
        "experiment_contract_sha256": experiment_contract_sha256,
        "certification_policy_sha256": certification_policy_sha256,
        "execution_provenance_sha256": digest_json(
            source_manifest.get("execution_provenance")
        ),
        "items": exported_items,
        "item_count": len(exported_items),
        "transition_count": transition_count,
        "episode_count": episode_count,
        "successful_episodes": successful_episodes,
        "certification_file": CERTIFICATION_FILE,
        "dataset_card_file": DATASET_CARD_FILE,
        "consumer_contract_file": CONSUMER_CONTRACT_FILE,
        "attestation": attestation,
        "transition_visibility": {
            "version": "1.0",
            "policy_projection": "env_factory.evidence.trajectory_schema.policy_transition",
            "trainer_only_evidence": "environments/*/live_rollout.json",
        },
        "trajectory_purpose": {
            "role": "certification_evidence",
            "direct_training_status": "not_certified",
            "environment_status": "certified_for_fresh_rollout_collection",
        },
        "files_sha256": files,
    }
    manifest["bundle_sha256"] = digest_json(manifest)
    (output / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    if signing_private_key is not None and trusted_public_key is not None:
        sign_file(
            output / BUNDLE_MANIFEST,
            output / BUNDLE_SIGNATURE_FILE,
            signing_private_key,
        )
    return verify_bundle(output, trusted_public_key=trusted_public_key)


def verify_bundle(
    root: Path, *, trusted_public_key: Path | None = None
) -> dict[str, Any]:
    failures = []
    try:
        manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "failed_gates": ["bundle_manifest"], "message": str(exc)}
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    attestation = manifest.get("attestation")
    expected_attestation_key = (
        attestation.get("key_identity_sha256")
        if isinstance(attestation, Mapping)
        and attestation.get("status") == "signed"
        else None
    )
    if manifest.get("bundle_sha256") != digest_json(unsigned):
        failures.append("bundle_digest")
    if (
        manifest.get("version") not in {
            "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0",
            BUNDLE_VERSION,
        }
        or manifest.get("kind") != "portable_agentic_rl_training_materials"
        or not isinstance(manifest.get("source_dataset_sha256"), str)
        or len(manifest["source_dataset_sha256"]) != 64
        or (
            supports_bundle_feature(
                str(manifest.get("version")), "certification_policy_binding"
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(manifest.get("certification_policy_sha256", "")),
            ) is None
        )
        or (
            supports_bundle_feature(
                str(manifest.get("version")), "portable_experiment_contract"
            )
            and (
                manifest.get("experiment_contract_file")
                    != EXPERIMENT_CONTRACT_FILE
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(manifest.get("experiment_contract_sha256", "")),
                ) is None
            )
        )
    ):
        failures.append("bundle_schema")
    production_contract_ready = manifest.get("version") in {
        "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0",
        BUNDLE_VERSION,
    }
    if production_contract_ready:
        try:
            contract_relative = _safe_relative(
                str(manifest.get("consumer_contract_file", ""))
            )
            if contract_relative != Path(CONSUMER_CONTRACT_FILE):
                raise ValueError("unexpected consumer contract path")
            consumer_contract_document = json.loads(
                (root / contract_relative).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, ValueError):
            consumer_contract_document = {}
        if consumer_contract_document != consumer_contract(manifest["version"]):
            failures.append("consumer_contract")
            production_contract_ready = False
    visibility = manifest.get("transition_visibility", {})
    if not (
        isinstance(visibility, Mapping)
        and visibility.get("version") == "1.0"
        and visibility.get("policy_projection")
            == "env_factory.evidence.trajectory_schema.policy_transition"
    ):
        failures.append("transition_visibility")
    expected = manifest.get("files_sha256")
    expected_files = dict(expected) if isinstance(expected, Mapping) else {}
    actual = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {BUNDLE_MANIFEST, BUNDLE_SIGNATURE_FILE}
    }
    if not isinstance(expected, Mapping) or expected_files != actual:
        failures.append("bundle_files")
    portable_experiment_contract: Mapping[str, Any] | dict[str, Any]
    try:
        experiment_contract_relative = _safe_relative(
            str(manifest.get("experiment_contract_file", ""))
        )
        if experiment_contract_relative != Path(EXPERIMENT_CONTRACT_FILE):
            raise ValueError("unexpected experiment contract path")
        portable_experiment_contract = json.loads(
            (root / experiment_contract_relative).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        portable_experiment_contract = {}
    experiment_contract_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "portable_experiment_contract"
        )
        and valid_experiment_contract(
            portable_experiment_contract,
            expected_source_config_sha256=manifest.get(
                "experiment_config_sha256"
            ),
        )
        and digest_json(portable_experiment_contract)
            == manifest.get("experiment_contract_sha256")
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "portable_experiment_contract"
    ) and not experiment_contract_ready:
        failures.append("experiment_contract")
    try:
        certification_relative = _safe_relative(
            str(manifest.get("certification_file", ""))
        )
        if certification_relative != Path(CERTIFICATION_FILE):
            raise ValueError("unexpected certification path")
        portable_certification = json.loads(
            (root / certification_relative).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        portable_certification = {}
        failures.append("portable_certification")
    preflight_report = portable_certification.get("measurements", {}).get(
        "production_preflight"
    ) if isinstance(portable_certification, Mapping) else None
    preflight_checks = {
        check.get("name"): check
        for check in preflight_report.get("checks", [])
        if isinstance(check, Mapping)
    } if isinstance(preflight_report, Mapping) else {}
    preflight_model_evidence = preflight_checks.get(
        "model_configuration", {}
    ).get("evidence", {})
    portable_metadata_privacy = audit_portable_metadata(
        portable_certification
    )
    if not portable_metadata_privacy["safe"]:
        failures.append("portable_metadata_privacy")
    required_limitations = {
        "rl_training_execution",
        "downstream_training_system_compatibility",
        "rl_training_convergence",
        "post_training_policy_improvement",
        "cross_model_generalization",
        "exported_rollouts_as_direct_policy_optimization_targets",
    }
    certification_policy_ready = (
        not supports_bundle_feature(
            str(manifest.get("version")), "certification_policy_binding"
        )
        or (
            valid_certification_policy(portable_certification.get("policy"))
            and digest_json(portable_certification.get("policy"))
                == manifest.get("certification_policy_sha256")
            and portable_certification.get("gates", {}).get(
                "certification_policy"
            ) is True
            and portable_certification.get("measurements", {}).get(
                "certification_policy"
            ) == {
                "version": portable_certification["policy"]["version"],
                "sha256": manifest.get("certification_policy_sha256"),
                "verified": True,
            }
        )
    ) if isinstance(portable_certification, Mapping) else False
    if not certification_policy_ready:
        failures.append("certification_policy")
    if not (
        isinstance(portable_certification, Mapping)
        and portable_certification.get("version") == "1.0"
        and portable_certification.get("certification")
            == "production_prepared_for_agentic_rl"
        and portable_certification.get("scope") == "pre_training_material_readiness"
        and portable_certification.get("certified") is True
        and portable_certification.get("failed_gates") == []
        and portable_certification.get("source_dataset_sha256")
            == manifest.get("source_dataset_sha256")
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "portable_experiment_contract"
            )
            or (
                experiment_contract_ready
                and portable_certification.get("experiment_contract_sha256")
                    == manifest.get("experiment_contract_sha256")
                and portable_certification.get("gates", {}).get(
                    "portable_experiment_contract"
                ) is True
                and portable_certification.get("measurements", {}).get(
                    "experiment_contract"
                ) == {
                    "version": portable_experiment_contract["version"],
                    "sha256": manifest.get("experiment_contract_sha256"),
                    "verified": True,
                }
            )
        )
        and certification_policy_ready
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "experiment_binding"
            )
            or portable_certification.get("experiment_config_sha256")
                == manifest.get("experiment_config_sha256")
        )
        and digest_json(portable_certification.get("execution_provenance"))
            == manifest.get("execution_provenance_sha256")
        and valid_execution_provenance(
            portable_certification.get("execution_provenance")
        )
        and required_limitations
            <= set(portable_certification.get("does_not_certify", []))
        and isinstance(portable_certification.get("gates"), Mapping)
        and portable_certification["gates"]
        and all(value is True for value in portable_certification["gates"].values())
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "reward_calibration"
            )
            or {
                "container_rollout_execution",
                "container_reward_calibration",
            } <= set(portable_certification["gates"])
        )
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "provider_binding"
            )
            or "provider_identity_consistency"
                in portable_certification["gates"]
        )
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "production_preflight"
            )
            or (
                {
                    "production_experiment_profile",
                    "production_preflight",
                    "evaluator_independence",
                } <= set(portable_certification["gates"])
                and portable_certification.get("measurements", {}).get(
                    "production_experiment_profile"
                ) is True
                and valid_production_preflight(
                    portable_certification.get("measurements", {}).get(
                        "production_preflight"
                    ),
                    expected_signing_key_identity=expected_attestation_key,
                    expected_experiment_config_sha256=(
                        manifest.get("experiment_config_sha256")
                        if supports_bundle_feature(
                            str(manifest.get("version")), "experiment_binding"
                        ) else None
                    ),
                    expected_bundle_version=str(manifest.get("version")),
                )
            )
        )
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "model_response_provenance"
            )
            or (
                portable_certification.get("gates", {}).get(
                    "model_response_provenance"
                ) is True
                and isinstance(
                    portable_certification.get("measurements", {}).get(
                        "model_response_provenance"
                    ), Mapping
                )
                and portable_certification["measurements"][
                    "model_response_provenance"
                ].get("all_verified") is True
            )
        )
        and (
            not supports_bundle_feature(
                str(manifest.get("version")), "model_response_authorization"
            )
            or portable_certification.get("gates", {}).get(
                "model_response_authorization"
            ) is True
        )
    ):
        failures.append("portable_certification")
    try:
        card_relative = _safe_relative(str(manifest.get("dataset_card_file", "")))
        if card_relative != Path(DATASET_CARD_FILE):
            raise ValueError("unexpected dataset card path")
        dataset_card = json.loads(
            (root / card_relative).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        dataset_card = {}
        failures.append("dataset_card")
    transition_path = root / TRANSITIONS_FILE
    records = []
    try:
        records = [json.loads(line) for line in transition_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        failures.append("transition_jsonl")
    consumer_record_failures = []
    for index, record in enumerate(records):
        errors = consumer_record_errors(
            record, str(manifest.get("version", ""))
        )
        if errors:
            consumer_record_failures.append({"index": index, "errors": errors})
    consumer_records_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "consumer_record_validation"
        )
        and bool(records)
        and not consumer_record_failures
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "consumer_record_validation"
    ) and not consumer_records_ready:
        failures.append("consumer_records")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        failures.append("bundle_items")
        items = []
    family_inputs = []
    verified_data_origins: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id", ""))
        try:
            environment = _safe_relative(str(item.get("environment_path", "")))
            task_document = json.loads(
                (root / environment / "task.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, ValueError):
            task_document = {}
        family_inputs.append({
            "item_id": item_id,
            "task": task_document.get("task") if isinstance(task_document, Mapping) else None,
        })
        if isinstance(task_document, Mapping):
            origin = task_document.get("artifacts", {}).get("data_manifest", {}).get("data_governance", {}).get("origin")
            if isinstance(origin, str):
                verified_data_origins[origin] += 1
    try:
        recomputed_families = task_family_ids(family_inputs)
    except ValueError:
        recomputed_families = {}
        failures.append("task_family_identity")
    expected_records = []
    seen_item_ids = set()
    verified_episode_count = 0
    verified_successes = 0
    verified_categories: Counter[str] = Counter()
    verified_model_pairs: Counter[tuple[str, str]] = Counter()
    verified_container_rollouts = 0
    verified_model_response_provenance = 0
    verified_agent_response_models: Counter[str] = Counter()
    verified_runtime_response_models: Counter[str] = Counter()
    verified_container_reward_calibrations = 0
    verified_provider_bindings = 0
    verified_data_governance_items = 0
    verified_preflight_provider_bindings = 0
    verified_same_provider_evaluator_items = 0
    verified_task_lineages = 0
    verified_portable_build_contexts = 0
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            failures.append("bundle_items")
            continue
        item_id = item.get("item_id")
        if (
            not isinstance(item_id, str)
            or re.fullmatch(r"[0-9a-f]{16}-[0-9a-f]{16}", item_id) is None
            or item_id in seen_item_ids
        ):
            failures.append("item_identity")
            continue
        seen_item_ids.add(item_id)
        expected_family_id = recomputed_families.get(item_id, "")
        if (
            supports_bundle_feature(str(manifest.get("version")), "families")
            and item.get("task_family_id") != expected_family_id
        ):
            failures.append("task_family_identity")
        generation = item.get("generation_provenance")
        if supports_bundle_feature(
            str(manifest.get("version")), "generation"
        ) and (
            not valid_generation_provenance(generation)
            or generation.get("training_category") != item.get("category")
            or generation.get("task_sha256") != item.get("task_sha256")
        ):
            failures.append("generation_provenance")
        try:
            environment = _safe_relative(str(item.get("environment_path", "")))
        except ValueError:
            failures.append("item_path")
            continue
        if environment != Path("environments") / item_id:
            failures.append("item_path")
        item_files = item.get("files_sha256")
        if not isinstance(item_files, Mapping) or not item_files:
            failures.append("item_files")
            continue
        for relative, digest in item_files.items():
            try:
                safe = _safe_relative(str(relative))
            except ValueError:
                failures.append("item_files")
                continue
            if expected_files.get(str(environment / safe)) != digest:
                failures.append("item_files")
        task_path = root / environment / "task.json"
        if not task_path.is_file() or file_sha256(task_path) != item.get("task_sha256"):
            failures.append("item_task")
        try:
            task_document = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            task_document = {}
        try:
            task_lineage = json.loads(
                (root / environment / "task_lineage.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            task_lineage = {}
        if supports_bundle_feature(
            str(manifest.get("version")), "task_lineage"
        ):
            if valid_task_lineage(task_lineage, task_path, task_path):
                verified_task_lineages += 1
            else:
                failures.append("task_lineage")
        runtime_interface = (
            task_document.get("requirements", {}).get("runtime_interface")
            if isinstance(task_document, Mapping) else None
        )
        if not (
            (root / environment / "Dockerfile").is_file()
            and "Dockerfile" in item_files
            and isinstance(runtime_interface, Mapping)
            and bool(runtime_interface)
        ):
            failures.append("environment_reconstruction_contract")
        if supports_bundle_feature(
            str(manifest.get("version")), "portable_build_context"
        ):
            environment_root = root / environment
            try:
                portable_files = portable_artifact_digests(environment_root)
            except (OSError, ValueError):
                portable_files = {}
            context_valid = (
                not docker_context_errors(environment_root)
                and bool(portable_files)
                and all(
                    item_files.get(relative) == digest
                    for relative, digest in portable_files.items()
                )
                and ".dockerignore" in portable_files
                and "Dockerfile" in portable_files
            )
            if context_valid:
                verified_portable_build_contexts += 1
            else:
                failures.append("portable_build_context")
        rollout_path = root / environment / "live_rollout.json"
        try:
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            if supports_bundle_feature(
                str(manifest.get("version")), "container_rollout"
            ):
                execution_valid = valid_container_rollout_execution(
                    rollout, root / environment,
                    require_build_context=supports_bundle_feature(
                        str(manifest.get("version")), "portable_build_context"
                    ),
                ) and item.get("runtime_execution") == rollout.get(
                    "runtime_execution"
                )
                if execution_valid:
                    verified_container_rollouts += 1
                else:
                    failures.append("container_rollout_execution")
            projected = _transition_records(
                item_id, item, rollout, split=str(item.get("split", "")),
                task_family_id=(
                    expected_family_id
                    if supports_bundle_feature(
                        str(manifest.get("version")), "families"
                    ) else ""
                ),
                generation_provenance=(
                    generation
                    if supports_bundle_feature(
                        str(manifest.get("version")), "generation"
                    )
                    else None
                ),
                bundle_version=str(manifest.get("version")),
            )
            if supports_bundle_feature(
                str(manifest.get("version")), "model_response_provenance"
            ):
                actual_response_provenance = response_provenance(rollout)
                if (
                    actual_response_provenance["verified"] is True
                    and item.get("model_response_provenance")
                        == actual_response_provenance
                ):
                    verified_model_response_provenance += 1
                    verified_agent_response_models.update(
                        actual_response_provenance["agent_response_models"]
                    )
                    verified_runtime_response_models.update(
                        actual_response_provenance["runtime_response_models"]
                    )
                else:
                    failures.append("model_response_provenance")
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            failures.append("transition_schema")
            continue
        if supports_bundle_feature(
            str(manifest.get("version")), "reward_calibration"
        ):
            try:
                calibration = json.loads(
                    (root / environment / "agentic_training_value_live.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                calibration = {}
            calibration_valid = (
                calibration.get("curriculum_training_ready") is True
                and calibration.get("validation_mode") == "live_evaluator"
                and valid_container_rollout_execution(
                    calibration, root / environment,
                    require_build_context=supports_bundle_feature(
                        str(manifest.get("version")), "portable_build_context"
                    ),
                )
                and item.get("reward_runtime_execution")
                == calibration.get("runtime_execution")
            )
            if calibration_valid:
                verified_container_reward_calibrations += 1
            else:
                failures.append("container_reward_calibration")
            if supports_bundle_feature(
                str(manifest.get("version")), "provider_binding"
            ):
                try:
                    governance = json.loads(
                        (root / environment / "data_governance.json").read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, json.JSONDecodeError):
                    governance = {}
                governance_valid = valid_governance_report(
                    governance, root / environment, task_document
                )
                if governance_valid:
                    verified_data_governance_items += 1
                else:
                    failures.append("data_governance")
                governed_binding_valid = valid_provider_binding(
                    rollout, calibration, governance
                )
                if governed_binding_valid:
                    verified_provider_bindings += 1
                else:
                    failures.append("provider_identity_binding")
                if supports_bundle_feature(
                    str(manifest.get("version")), "production_preflight"
                ):
                    providers = (
                        governance.get("providers")
                        if isinstance(governance, Mapping) else None
                    )
                    preflight_binding_valid = (
                        governed_binding_valid
                        and isinstance(generation, Mapping)
                        and isinstance(providers, Mapping)
                        and generation.get("generator_provider")
                            == preflight_model_evidence.get(
                                "generation_provider"
                            )
                        and providers.get("agent")
                            == preflight_model_evidence.get("agent_provider")
                        and providers.get("user_simulator_and_reward")
                            == preflight_model_evidence.get("runtime_provider")
                    )
                    if preflight_binding_valid:
                        verified_preflight_provider_bindings += 1
                    else:
                        failures.append("preflight_provider_binding")
                if (
                    rollout.get("agent_provider_sha256")
                    == rollout.get("runtime_provider_sha256")
                    or providers.get("agent", {}).get("host")
                        == providers.get("user_simulator_and_reward", {}).get(
                            "host"
                        )
                ):
                    verified_same_provider_evaluator_items += 1
        if item.get("episode_count") != len(rollout.get("episodes", [])):
            failures.append("item_episode_count")
        if item.get("transition_count") != len(projected):
            failures.append("item_transition_count")
        expected_records.extend(projected)
        episodes = rollout.get("episodes", [])
        verified_episode_count += len(episodes)
        verified_successes += sum(
            episode.get("agent_success") is True
            for episode in episodes if isinstance(episode, Mapping)
        )
        verified_categories[str(item.get("category"))] += 1
        verified_model_pairs[(
            str(rollout.get("agent_model", "")),
            str(rollout.get("runtime_model", "")),
        )] += 1
    if records != expected_records:
        failures.append("transition_projection")
    item_ids = [item.get("item_id") for item in items if isinstance(item, Mapping)]
    record_identities = [
        (
            record.get("item_id"), record.get("episode_index"),
            record.get("transition", {}).get("step")
            if isinstance(record.get("transition"), Mapping) else None,
        )
        for record in records if isinstance(record, Mapping)
    ]
    if item_ids != sorted(item_ids) or len(record_identities) != len(set(record_identities)):
        failures.append("consumer_record_order_or_identity")
    expected_splits = assign_dataset_splits([
        {
            "item_id": str(item.get("item_id")),
            "category": str(item.get("category")),
            "task_family_id": (
                recomputed_families.get(str(item.get("item_id")), "")
                if supports_bundle_feature(
                    str(manifest.get("version")), "families"
                ) else ""
            ),
        }
        for item in items if isinstance(item, Mapping)
    ])
    split_counts = Counter(
        str(item.get("split")) for item in items if isinstance(item, Mapping)
    )
    category_split_counts = Counter(
        (str(item.get("category")), str(item.get("split")))
        for item in items if isinstance(item, Mapping)
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "splits"
    ) and any(
        item.get("split") != expected_splits.get(str(item.get("item_id")))
        for item in items if isinstance(item, Mapping)
    ):
        failures.append("dataset_split_assignment")
    verified_category_names = set(verified_categories)
    dataset_split_ready = (
        all(split_counts.get(split, 0) > 0 for split in DATASET_SPLITS)
        and all(
            category_split_counts.get((category, split), 0) > 0
            for category in verified_category_names for split in DATASET_SPLITS
        )
    )
    family_categories: dict[str, set[str]] = {}
    family_splits: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        family = recomputed_families.get(str(item.get("item_id")), "")
        family_categories.setdefault(family, set()).add(str(item.get("category")))
        family_splits.setdefault(family, set()).add(str(item.get("split")))
    family_split_ready = (
        supports_bundle_feature(str(manifest.get("version")), "families")
        and bool(family_categories)
        and all(len(values) == 1 for values in family_categories.values())
        and all(len(values) == 1 for values in family_splits.values())
    )
    generation_values = [
        item.get("generation_provenance")
        for item in items if isinstance(item, Mapping)
    ]
    generation_seeds = [
        value.get("sample_seed") for value in generation_values
        if isinstance(value, Mapping)
    ]
    generation_provenance_ready = (
        supports_bundle_feature(str(manifest.get("version")), "generation")
        and len(generation_values) == len(items)
        and all(valid_generation_provenance(value) for value in generation_values)
        and len(generation_seeds) == len(set(generation_seeds))
    )
    container_rollout_ready = (
        supports_bundle_feature(str(manifest.get("version")), "container_rollout")
        and bool(items)
        and verified_container_rollouts == len(items)
    )
    portable_build_context_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "portable_build_context"
        )
        and bool(items)
        and verified_portable_build_contexts == len(items)
    )
    container_reward_calibration_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "reward_calibration"
        )
        and bool(items)
        and verified_container_reward_calibrations == len(items)
    )
    provider_identity_ready = (
        supports_bundle_feature(str(manifest.get("version")), "provider_binding")
        and bool(items)
        and verified_provider_bindings == len(items)
    )
    data_governance_ready = (
        supports_bundle_feature(str(manifest.get("version")), "provider_binding")
        and bool(items)
        and verified_data_governance_items == len(items)
        and portable_certification.get("gates", {}).get("data_governance") is True
        and portable_certification.get("measurements", {}).get(
            "data_governance"
        ) == {
            "reports": len(items),
            "verified": len(items),
            "all_verified": True,
        }
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "provider_binding"
    ) and not data_governance_ready:
        failures.append("data_governance")
    task_lineage_ready = (
        supports_bundle_feature(str(manifest.get("version")), "task_lineage")
        and bool(items)
        and verified_task_lineages == len(items)
    )
    production_preflight_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "production_preflight"
        )
        and isinstance(portable_certification, Mapping)
        and portable_certification.get("gates", {}).get(
            "production_experiment_profile"
        ) is True
        and portable_certification.get("gates", {}).get(
            "production_preflight"
        ) is True
        and portable_certification.get("measurements", {}).get(
            "production_experiment_profile"
        ) is True
        and valid_production_preflight(
            portable_certification.get("measurements", {}).get(
                "production_preflight"
            ),
            expected_signing_key_identity=expected_attestation_key,
            expected_experiment_config_sha256=(
                manifest.get("experiment_config_sha256")
                if supports_bundle_feature(
                    str(manifest.get("version")), "experiment_binding"
                ) else None
            ),
            expected_bundle_version=str(manifest.get("version")),
        )
        and bool(items)
        and verified_preflight_provider_bindings == len(items)
    )
    experiment_config_ready = (
        supports_bundle_feature(str(manifest.get("version")), "experiment_binding")
        and isinstance(manifest.get("experiment_config_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", manifest["experiment_config_sha256"]
        ) is not None
        and isinstance(preflight_report, Mapping)
        and preflight_report.get("experiment_config_sha256")
            == manifest.get("experiment_config_sha256")
    )
    trajectory_purpose_ready = (
        supports_bundle_feature(str(manifest.get("version")), "trajectory_purpose")
        and manifest.get("trajectory_purpose") == {
            "role": "certification_evidence",
            "direct_training_status": "not_certified",
            "environment_status": "certified_for_fresh_rollout_collection",
        }
        and bool(records)
        and all(
            record.get("trajectory_role") == "certification_evidence"
            and record.get("direct_training_status") == "not_certified"
            for record in records
        )
    )
    model_response_provenance_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "model_response_provenance"
        )
        and bool(items)
        and verified_model_response_provenance == len(items)
        and portable_certification.get("measurements", {}).get(
            "model_response_provenance"
        ) == {
            "verified": len(items),
            "expected": len(items),
            "all_verified": True,
            "agent_response_models": dict(
                sorted(verified_agent_response_models.items())
            ),
            "runtime_response_models": dict(
                sorted(verified_runtime_response_models.items())
            ),
        }
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "model_response_provenance"
    ) and not model_response_provenance_ready:
        failures.append("model_response_provenance")
    if supports_bundle_feature(
        str(manifest.get("version")), "trajectory_purpose"
    ) and not trajectory_purpose_ready:
        failures.append("trajectory_purpose")
    evaluator_independence_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "production_preflight"
        )
        and portable_certification.get("gates", {}).get(
            "evaluator_independence"
        ) is True
        and portable_certification.get("measurements", {}).get(
            "evaluator_independence"
        ) == {
            "same_provider_items": verified_same_provider_evaluator_items,
            "total_items": len(items),
            "same_provider_rate": (
                verified_same_provider_evaluator_items / len(items)
                if items else 1.0
            ),
        }
        and verified_same_provider_evaluator_items == 0
    )
    if (
        len(records) != manifest.get("transition_count")
        or len(items) != manifest.get("item_count")
    ):
        failures.append("bundle_counts")
    if manifest.get("item_count", 0) <= 0 or manifest.get("transition_count", 0) <= 0:
        failures.append("empty_bundle")
    card_composition = dataset_card.get("composition", {}) if isinstance(
        dataset_card, Mapping
    ) else {}
    expected_model_pairs = [
        {"agent_model": pair[0], "runtime_model": pair[1], "items": count}
        for pair, count in sorted(verified_model_pairs.items())
    ]
    verified_same_model_items = sum(
        count for (agent, runtime), count in verified_model_pairs.items()
        if agent == runtime
    )
    card_limits = set(dataset_card.get("prohibited_interpretations", [])) if isinstance(
        dataset_card, Mapping
    ) else set()
    expected_split_counts = {
        name: split_counts.get(name, 0) for name in DATASET_SPLITS
    }
    expected_category_splits = {
        category: {
            split: category_split_counts.get((category, split), 0)
            for split in DATASET_SPLITS
        }
        for category in sorted(verified_category_names)
    }
    expected_family_count = len(set(recomputed_families.values()))
    expected_generation_configured_models = Counter(
        str(value.get("generator_provider", {}).get("model"))
        for value in generation_values if isinstance(value, Mapping)
    )
    expected_generation_actual_models = Counter()
    for value in generation_values:
        if not isinstance(value, Mapping):
            continue
        for attempt in value.get("attempts", []):
            expected_generation_actual_models.update(
                attempt.get("llm_trace", {}).get("models", {})
            )
    expected_generation_providers = Counter(
        str(value.get("generator_provider", {}).get("identity_sha256"))
        for value in generation_values if isinstance(value, Mapping)
    )
    expected_generation_attempts = sum(
        len(value.get("attempts", []))
        for value in generation_values if isinstance(value, Mapping)
    )
    expected_generation_responses = sum(
        attempt.get("llm_trace", {}).get("responses", 0)
        for value in generation_values if isinstance(value, Mapping)
        for attempt in value.get("attempts", [])
    )
    expected_agent_response_models: Counter[str] = Counter()
    expected_runtime_response_models: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        provenance = item.get("model_response_provenance", {})
        if isinstance(provenance, Mapping):
            expected_agent_response_models.update(
                provenance.get("agent_response_models", {})
            )
            expected_runtime_response_models.update(
                provenance.get("runtime_response_models", {})
            )
    allowed_response_models = {
        "generation": preflight_model_evidence.get(
            "generation_allowed_response_models"
        ),
        "agent": preflight_model_evidence.get("agent_allowed_response_models"),
        "runtime": preflight_model_evidence.get(
            "runtime_allowed_response_models"
        ),
    }
    actual_response_models = {
        "generation": dict(sorted(expected_generation_actual_models.items())),
        "agent": dict(sorted(expected_agent_response_models.items())),
        "runtime": dict(sorted(expected_runtime_response_models.items())),
    }
    expected_authorization_measurement = {
        "all_authorized": all(
            isinstance(allowed_response_models[role], list)
            and bool(actual_response_models[role])
            and set(actual_response_models[role])
                <= set(allowed_response_models[role])
            for role in ("generation", "agent", "runtime")
        ),
        "allowed": allowed_response_models,
        "actual": actual_response_models,
    }
    model_response_authorization_ready = (
        supports_bundle_feature(
            str(manifest.get("version")), "model_response_authorization"
        )
        and expected_authorization_measurement["all_authorized"] is True
        and portable_certification.get("measurements", {}).get(
            "model_response_authorization"
        ) == expected_authorization_measurement
    )
    if supports_bundle_feature(
        str(manifest.get("version")), "model_response_authorization"
    ) and not model_response_authorization_ready:
        failures.append("model_response_authorization")
    expected_runtime_modes = Counter(
        str(item.get("runtime_execution", {}).get("mode"))
        for item in items if isinstance(item, Mapping)
    )
    expected_container_images = {
        str(item.get("runtime_execution", {}).get("container_image_id"))
        for item in items if isinstance(item, Mapping)
        and item.get("runtime_execution", {}).get("mode") == "docker_http"
    }
    expected_reward_runtime_modes = Counter(
        str(item.get("reward_runtime_execution", {}).get("mode"))
        for item in items if isinstance(item, Mapping)
    )
    expected_reward_container_images = {
        str(item.get("reward_runtime_execution", {}).get("container_image_id"))
        for item in items if isinstance(item, Mapping)
        and item.get("reward_runtime_execution", {}).get("mode") == "docker_http"
    }
    version = manifest.get("version")
    versioned_card_composition = True
    if supports_bundle_feature(str(version), "splits"):
        versioned_card_composition = (
            card_composition.get("splits") == expected_split_counts
            and card_composition.get("category_splits") == expected_category_splits
        )
    if supports_bundle_feature(str(version), "families"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("task_families") == expected_family_count
            and card_composition.get("near_duplicate_items")
                == len(items) - expected_family_count
            and card_composition.get("cross_split_family_overlap") == 0
        )
    if supports_bundle_feature(str(version), "generation"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("task_generation") == {
                "configured_models": dict(
                    sorted(expected_generation_configured_models.items())
                ),
                "actual_response_models": dict(
                    sorted(expected_generation_actual_models.items())
                ),
                "provider_identities": dict(
                    sorted(expected_generation_providers.items())
                ),
                "attempts": expected_generation_attempts,
                "responses": expected_generation_responses,
            }
        )
    if supports_bundle_feature(str(version), "model_response_provenance"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("rollout_response_provenance") == {
                "agent_models": dict(sorted(expected_agent_response_models.items())),
                "runtime_models": dict(
                    sorted(expected_runtime_response_models.items())
                ),
            }
        )
    if supports_bundle_feature(str(version), "model_response_authorization"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("model_response_authorization")
                == expected_authorization_measurement
        )
    if supports_bundle_feature(str(version), "container_rollout"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("runtime_execution") == {
                "modes": dict(sorted(expected_runtime_modes.items())),
                "validated_container_items": expected_runtime_modes.get(
                    "docker_http", 0
                ),
                "unique_container_images": len(expected_container_images),
            }
        )
    if supports_bundle_feature(str(version), "reward_calibration"):
        versioned_card_composition = versioned_card_composition and (
            card_composition.get("reward_calibration_execution") == {
                "modes": dict(sorted(expected_reward_runtime_modes.items())),
                "validated_container_items": expected_reward_runtime_modes.get(
                    "docker_http", 0
                ),
                "unique_container_images": len(
                    expected_reward_container_images
                ),
            }
        )
    if not (
        isinstance(dataset_card, Mapping)
        and dataset_card.get("version") == (
            "2.3" if manifest.get("version") == BUNDLE_VERSION
            else "2.2" if manifest.get("version") == "16.0"
            else "2.1" if manifest.get("version") == "15.0"
            else "2.0" if manifest.get("version") == "14.0"
            else "1.9" if manifest.get("version") == "13.0"
            else "1.8" if manifest.get("version") == "12.0"
            else "1.7" if manifest.get("version") == "11.0"
            else "1.6" if manifest.get("version") == "10.0"
            else "1.5" if manifest.get("version") == "9.0"
            else "1.4" if manifest.get("version") == "8.0"
            else "1.3" if manifest.get("version") == "7.0"
            else "1.2" if manifest.get("version") == "6.0"
            else "1.1" if manifest.get("version") in {"4.0", "5.0"}
            else "1.0"
        )
        and dataset_card.get("kind") == "agentic_rl_pretraining_material_dataset_card"
        and dataset_card.get("source_dataset_sha256")
            == manifest.get("source_dataset_sha256")
        and dataset_card.get("build_environment")
            == portable_certification.get("execution_provenance")
        and digest_json(dataset_card.get("build_environment"))
            == manifest.get("execution_provenance_sha256")
        and dataset_card.get("certification", {}).get("certified") is True
        and (
            not supports_bundle_feature(
                str(version), "portable_experiment_contract"
            )
            or dataset_card.get("certification", {}).get(
                "experiment_contract_sha256"
            ) == manifest.get("experiment_contract_sha256")
        )
        and (
            not supports_bundle_feature(
                str(version), "certification_policy_binding"
            )
            or dataset_card.get("certification", {}).get("policy_sha256")
                == manifest.get("certification_policy_sha256")
        )
        and dataset_card.get("distribution_status")
            == "internal_only_until_legal_and_security_review"
        and dataset_card.get("license_status") == "not_asserted_by_envfactory"
        and (
            manifest.get("version") not in {
                "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0",
                BUNDLE_VERSION,
            }
            or dataset_card.get("consumer_contract") == CONSUMER_CONTRACT_FILE
        )
        and card_composition.get("items") == len(items)
        and card_composition.get("transitions") == len(records)
        and card_composition.get("episodes") == verified_episode_count
        and card_composition.get("successful_episodes") == verified_successes
        and card_composition.get("failed_episodes")
            == verified_episode_count - verified_successes
        and card_composition.get("categories")
            == dict(sorted(verified_categories.items()))
        and card_composition.get("model_pairs") == expected_model_pairs
        and card_composition.get("same_model_pair_items")
            == verified_same_model_items
        and versioned_card_composition
        and required_limitations <= card_limits
        and "offline_rl_algorithm_compatibility" in card_limits
        and dataset_card.get("data_boundary", {}).get("policy_transitions")
            == TRANSITIONS_FILE
        and dataset_card.get("data_boundary", {}).get("origin")
            == (next(iter(verified_data_origins)) if len(verified_data_origins) == 1 else "mixed_verified_origins")
        and (
            dataset_card.get("data_boundary", {}).get("origins")
                == dict(sorted(verified_data_origins.items()))
            or (version != BUNDLE_VERSION
                and dataset_card.get("data_boundary", {}).get("origins") is None
                and set(verified_data_origins) == {"model_generated_synthetic"})
        )
        and dataset_card.get("data_boundary", {}).get("contains_real_user_data")
            == (False if set(verified_data_origins) == {"model_generated_synthetic"}
                else "undetermined")
        and (
            not supports_bundle_feature(str(version), "trajectory_purpose")
            or (
                dataset_card.get("data_boundary", {}).get(
                    "exported_trajectory_role"
                ) == "certification_evidence"
                and dataset_card.get("data_boundary", {}).get(
                    "direct_training_status"
                ) == "not_certified"
            )
        )
        and "do_not_feed_trainer_only_evidence_to_the_policy"
            in dataset_card.get("consumer_requirements", [])
    ):
        failures.append("dataset_card")
    if (
        manifest.get("episode_count") != verified_episode_count
        or manifest.get("successful_episodes") != verified_successes
    ):
        failures.append("bundle_episode_counts")
    trusted_attestation = (
        manifest.get("version") in {
            "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0",
            BUNDLE_VERSION,
        }
        and trusted_public_key is not None
        and verify_file(
            root / BUNDLE_MANIFEST,
            root / BUNDLE_SIGNATURE_FILE,
            trusted_public_key,
            attestation,
        )
    )
    if trusted_public_key is not None and not trusted_attestation:
        failures.append("trusted_attestation")
    return {
        "verified": not failures,
        "bundle_version": manifest.get("version"),
        "bundle_sha256": manifest.get("bundle_sha256"),
        "source_dataset_sha256": manifest.get("source_dataset_sha256"),
        "items": manifest.get("item_count", 0),
        "transitions": len(records),
        "production_contract_ready": production_contract_ready,
        "trusted_attestation": trusted_attestation,
        "dataset_split_ready": dataset_split_ready,
        "task_family_split_ready": family_split_ready,
        "generation_provenance_ready": generation_provenance_ready,
        "container_rollout_ready": container_rollout_ready,
        "portable_build_context_ready": portable_build_context_ready,
        "certification_policy_ready": certification_policy_ready,
        "container_reward_calibration_ready": container_reward_calibration_ready,
        "provider_identity_ready": provider_identity_ready,
        "data_governance_ready": data_governance_ready,
        "task_lineage_ready": task_lineage_ready,
        "production_preflight_ready": production_preflight_ready,
        "experiment_config_ready": experiment_config_ready,
        "experiment_contract_ready": experiment_contract_ready,
        "trajectory_purpose_ready": trajectory_purpose_ready,
        "model_response_provenance_ready": model_response_provenance_ready,
        "model_response_authorization_ready": model_response_authorization_ready,
        "consumer_records_ready": consumer_records_ready,
        "consumer_record_failures": consumer_record_failures,
        "preflight_provider_bindings": verified_preflight_provider_bindings,
        "metadata_privacy_ready": portable_metadata_privacy["safe"],
        "evaluator_independence_ready": evaluator_independence_ready,
        "attestation_key_identity_sha256": (
            attestation.get("key_identity_sha256")
            if isinstance(attestation, Mapping) else None
        ),
        "failed_gates": sorted(set(failures)),
    }


def export_bundle(
    certification: Mapping[str, Any], output: Path, project: Path,
    *, signing_private_key: Path | None = None, trusted_public_key: Path | None = None,
) -> dict[str, Any]:
    """Stage the complete bundle and publish it with one directory rename."""
    if (signing_private_key is None) != (trusted_public_key is None):
        raise ValueError("signing private key and trusted public key are both required")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"bundle output is not empty: {output}")
        output.rmdir()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        report = _export_bundle_uncommitted(
            certification, staged, project,
            signing_private_key=signing_private_key,
            trusted_public_key=trusted_public_key,
        )
        if report.get("verified") is not True:
            raise ValueError(f"staged bundle verification failed: {report['failed_gates']}")
        staged.replace(output)
    return verify_bundle(output, trusted_public_key=trusted_public_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--signing-private-key", type=Path)
    parser.add_argument("--trusted-public-key", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        report = verify_bundle(
            args.output.resolve(), trusted_public_key=args.trusted_public_key
        )
    else:
        certification = json.loads(args.certification.read_text(encoding="utf-8"))
        report = export_bundle(
            certification, args.output.resolve(), args.project.resolve(),
            signing_private_key=args.signing_private_key,
            trusted_public_key=args.trusted_public_key,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
