#!/usr/bin/env python3
"""Verify that certified Agentic-RL material artifacts are still immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from env_factory.evidence.material_artifacts import (
    MATERIAL_MANIFEST_VERSION,
    SUPPORTED_MATERIAL_MANIFEST_VERSIONS,
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digests,
)
from env_factory.evidence.execution_provenance import valid_execution_provenance
from env_factory.evidence.generation_provenance import valid_generation_provenance
from env_factory.evidence.certification_policy import valid_certification_policy
from env_factory.evidence.experiment_contract import valid_experiment_contract


V5_ITEM_FIELDS = {
    "task_path",
    "task_sha256",
    "sandbox_root",
    "sandbox_evidence_fingerprint",
    "sandbox_artifacts_sha256",
    "sandbox_evidence_sha256",
    "category",
    "score",
    "rollout_sha256",
    "episode_count",
    "successful_episodes",
    "generation_provenance",
}
TRAINING_CATEGORIES = {
    "direct_response", "simple_agentic", "multi_step_agentic",
}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _qualified_item_errors(
    item: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[str]:
    errors = []
    score = item.get("score")
    episode_count = item.get("episode_count")
    successful_episodes = item.get("successful_episodes")
    generation = item.get("generation_provenance")
    score_threshold = (
        policy.get("score_threshold") if isinstance(policy, Mapping) else None
    )
    if set(item) != V5_ITEM_FIELDS:
        errors.append("fields")
    for field in ("task_sha256", "sandbox_evidence_fingerprint", "rollout_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(item.get(field, ""))) is None:
            errors.append(field)
    if item.get("category") not in TRAINING_CATEGORIES:
        errors.append("category")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not math.isfinite(float(score))
        or not float(score_threshold) <= float(score) <= 10.0
    ):
        errors.append("score")
    if (
        not _integer(episode_count)
        or episode_count < 0
        or not _integer(successful_episodes)
        or not 0 <= successful_episodes <= episode_count
    ):
        errors.append("episode_counts")
    if (
        not valid_generation_provenance(generation)
        or generation.get("task_sha256") != item.get("task_sha256")
        or generation.get("training_category") != item.get("category")
    ):
        errors.append("generation_provenance")
    return sorted(set(errors))


def verify_artifact_digests(root: Path, expected: Mapping[str, Any]) -> list[str]:
    failures = []
    normalized = {}
    for relative, wanted in expected.items():
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            relative_path is None or relative_path.is_absolute()
            or ".." in relative_path.parts or not isinstance(wanted, str)
        ):
            failures.append(str(relative))
            continue
        normalized[relative] = wanted
    try:
        actual = portable_artifact_digests(root)
    except OSError as exc:
        return [f"artifact inventory unavailable: {exc}"]
    failures.extend(sorted(set(normalized) ^ set(actual)))
    failures.extend(
        relative for relative in sorted(set(normalized) & set(actual))
        if normalized[relative] != actual[relative]
    )
    return sorted(set(failures))


def verify(
    manifest: Mapping[str, Any], project: Path,
    *, fingerprint: Callable[[Path, Path], str] | None = None,
) -> dict[str, Any]:
    failures = []
    version = manifest.get("version")
    if (
        version not in SUPPORTED_MATERIAL_MANIFEST_VERSIONS
        or manifest.get("kind") != "agentic_rl_pretraining_materials"
    ):
        failures.append({
            "gate": "manifest_schema",
            "message": "unsupported material manifest version or kind",
        })
    base = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest.get("dataset_sha256") != digest_json(base):
        failures.append({"gate": "dataset_digest", "message": "manifest digest changed"})
    if version in {"3.0", "4.0", "5.0", MATERIAL_MANIFEST_VERSION} and not valid_execution_provenance(
        manifest.get("execution_provenance")
    ):
        failures.append({
            "gate": "execution_provenance",
            "message": "v3+ manifest needs a valid execution environment snapshot",
        })
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        failures.append({"gate": "manifest_schema", "message": "items must be non-empty"})
        items = []
    if version in {"4.0", "5.0", MATERIAL_MANIFEST_VERSION} and any(
        not isinstance(item, Mapping)
        or not valid_generation_provenance(item.get("generation_provenance"))
        or item["generation_provenance"].get("task_sha256")
            != item.get("task_sha256")
        for item in items
    ):
        failures.append({
            "gate": "generation_provenance",
            "message": "v4+ manifest items need valid generation provenance",
        })
    policy = manifest.get("certification_policy")
    policy_digest = manifest.get("certification_policy_sha256")
    if version in {"5.0", MATERIAL_MANIFEST_VERSION} and not (
        valid_certification_policy(policy)
        and policy_digest == digest_json(policy)
    ):
        failures.append({
            "gate": "certification_policy",
            "message": "v5 manifest needs a canonical certification policy binding",
        })
    experiment_contract = manifest.get("experiment_contract")
    experiment_contract_digest = manifest.get("experiment_contract_sha256")
    if version == MATERIAL_MANIFEST_VERSION and not (
        valid_experiment_contract(
            experiment_contract,
            expected_source_config_sha256=manifest.get(
                "experiment_config_sha256"
            ),
        )
        and experiment_contract_digest == digest_json(experiment_contract)
    ):
        failures.append({
            "gate": "experiment_contract",
            "message": "v6 manifest needs a portable frozen experiment contract",
        })
    seen = set()
    seen_roots = set()
    seen_tasks = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            failures.append({"gate": "manifest_schema", "item": index})
            continue
        if version in {"5.0", MATERIAL_MANIFEST_VERSION}:
            item_errors = _qualified_item_errors(item, policy)
            if item_errors:
                failures.append({
                    "gate": "manifest_item_schema",
                    "item": index,
                    "message": f"invalid qualified item fields: {item_errors}",
                })
        task = Path(str(item.get("task_path", "")))
        root = Path(str(item.get("sandbox_root", "")))
        resolved_root = str(root.resolve())
        if resolved_root in seen_roots:
            failures.append({"gate": "duplicate_sandbox_root", "item": index})
        seen_roots.add(resolved_root)
        try:
            task_digest = hashlib.sha256(task.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append({"gate": "task_artifact", "item": index, "message": str(exc)})
            continue
        if task_digest != item.get("task_sha256"):
            failures.append({"gate": "task_digest", "item": index})
        if item.get("task_sha256") in seen_tasks:
            failures.append({"gate": "duplicate_task_identity", "item": index})
        seen_tasks.add(item.get("task_sha256"))
        expected_fingerprint = item.get("sandbox_evidence_fingerprint")
        artifact_hashes = item.get("sandbox_artifacts_sha256")
        if version in {"2.0", "3.0", "4.0", "5.0", MATERIAL_MANIFEST_VERSION} and not (
            isinstance(artifact_hashes, Mapping) and artifact_hashes
        ):
            failures.append({"gate": "manifest_schema", "item": index,
                             "message": "v2/v3 item needs sandbox_artifacts_sha256"})
            continue
        if isinstance(artifact_hashes, Mapping) and artifact_hashes:
            changed = verify_artifact_digests(root, artifact_hashes)
            if changed:
                failures.append({
                    "gate": "sandbox_digest", "item": index,
                    "message": f"changed or missing artifacts: {changed}",
                })
        else:
            # Backward compatibility for v1 manifests. V2/v3 deliberately avoid
            # recomputing a fingerprint that also depends on the current
            # evaluator source tree.
            try:
                if fingerprint is None:
                    from env_factory.sandbox_scoring import evidence_fingerprint
                    fingerprint = evidence_fingerprint
                actual_fingerprint = fingerprint(root, project)
            except Exception as exc:  # verifier must report, not abort the batch
                failures.append({
                    "gate": "sandbox_artifact", "item": index,
                    "message": f"{type(exc).__name__}: {exc}",
                })
                continue
            if actual_fingerprint != expected_fingerprint:
                failures.append({"gate": "sandbox_digest", "item": index})
        evidence_hashes = item.get("sandbox_evidence_sha256")
        if version in {"2.0", "3.0", "4.0", "5.0", MATERIAL_MANIFEST_VERSION} and not (
            isinstance(evidence_hashes, Mapping) and evidence_hashes
        ):
            failures.append({"gate": "manifest_schema", "item": index,
                             "message": "v2/v3 item needs sandbox_evidence_sha256"})
        elif isinstance(evidence_hashes, Mapping):
            actual_evidence = evidence_artifact_digests(root)
            if dict(evidence_hashes) != actual_evidence:
                failures.append({"gate": "sandbox_evidence_digest", "item": index})
        if expected_fingerprint in seen:
            failures.append({"gate": "duplicate_sandbox_identity", "item": index})
        seen.add(expected_fingerprint)
        try:
            rollout = json.loads((root / "live_rollout.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"gate": "rollout_artifact", "item": index, "message": str(exc)})
            continue
        if digest_json(rollout) != item.get("rollout_sha256"):
            failures.append({"gate": "rollout_digest", "item": index})
        episodes = rollout.get("episodes", []) if isinstance(rollout, Mapping) else []
        if len(episodes) != item.get("episode_count"):
            failures.append({"gate": "episode_count", "item": index})
        successful_episodes = sum(
            episode.get("agent_success") is True
            for episode in episodes if isinstance(episode, Mapping)
        )
        if (
            version in {"5.0", MATERIAL_MANIFEST_VERSION}
            and successful_episodes != item.get("successful_episodes")
        ):
            failures.append({"gate": "successful_episode_count", "item": index})
    return {
        "verified": not failures,
        "items": len(items),
        "failed_gates": sorted({item["gate"] for item in failures}),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify(value, args.project.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
