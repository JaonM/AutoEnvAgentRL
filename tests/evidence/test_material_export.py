import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from collections import Counter

from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE,
    MATERIAL_MANIFEST_VERSION,
    docker_build_context_digest,
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digests,
)
from env_factory.evidence.execution_provenance import collect_execution_provenance
from env_factory.evidence.material_attestation import public_key_identity
from env_factory.evidence.certification_policy import canonical_certification_policy
from env_factory.evidence.experiment_contract import build_experiment_contract
from env_factory.evidence.data_governance import (
    FORBIDDEN_OUTBOUND,
    OUTBOUND_SURFACES,
    SYNTHETIC_ORIGIN,
)


ROOT = Path(__file__).resolve().parents[2]


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_training_materials", ROOT / "scripts/export_training_materials.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


class MaterialExportTest(unittest.TestCase):
    @staticmethod
    def remove_current_trajectory_purpose(bundle: Path, manifest: dict) -> None:
        manifest.pop("trajectory_purpose", None)
        transition_path = bundle / "transitions.jsonl"
        records = [
            json.loads(line) for line in transition_path.read_text().splitlines()
        ]
        for record in records:
            record.pop("trajectory_role", None)
            record.pop("direct_training_status", None)
        transition_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )
        manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
            transition_path
        )

    def generation_provenance(self):
        return {
            "version": "1.0",
            "task_id": "task-1",
            "batch_index": 1,
            "run_seed": 100,
            "sample_seed": 101,
            "task_sha256": "PLACEHOLDER",
            "training_category": "simple_agentic",
            "resolved_task_intent": "tool_execution",
            "hops": 3,
            "generator_provider": {
                "host": "generator.example",
                "model": "generator-model",
                "identity_sha256": "c" * 64,
            },
            "generation_settings": {
                "route_attempt_limit": 3,
                "timeout_seconds": 60.0,
                "network_retries": 2,
            },
            "successful_attempt": 1,
            "attempts": [{
                "attempt": 1,
                "seed": 101,
                "status": "completed",
                "llm_trace": {
                    "version": "1.0",
                    "responses": 2,
                    "models": {"generator-model": 2},
                    "finish_reasons": {"stop": 2},
                    "usage": {"total_tokens": 42},
                    "response_id_sha256": ["d" * 64, "e" * 64],
                },
            }],
        }

    def signing_keys(self, root: Path) -> tuple[Path, Path]:
        private = root / "private.pem"
        public = root / "public.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
            check=True, capture_output=True,
        )
        return private, public

    def source(self, root: Path):
        sandbox = root / "sandbox"
        sandbox.mkdir()
        task = sandbox / "task.json"
        task.write_text(json.dumps({
            "task": "use the tool",
            "requirements": {"runtime_interface": {
                "protocol": "http", "base_path": "/v1", "version": "1.0",
            }},
            "artifacts": {"data_manifest": {
                "data_governance": SYNTHETIC_ORIGIN,
            }},
        }))
        task_document = json.loads(task.read_text())
        task_digest = hashlib.sha256(task.read_bytes()).hexdigest()
        (sandbox / "task_lineage.json").write_text(json.dumps({
            "version": "1.0",
            "source_is_task_list": False,
            "source_task_content_sha256": digest_json(task_document),
            "runtime_task_content_sha256": digest_json(task_document),
            "source_file_sha256": task_digest,
            "runtime_task_sha256": task_digest,
            "identity_preserved": True,
            "legacy_manifest_relocations": [],
        }))
        (sandbox / "app.py").write_text("# portable runtime\n")
        (sandbox / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
        nested = sandbox / "business" / "rules"
        nested.mkdir(parents=True)
        (nested / "engine.py").write_text("VALUE = 1\n")
        (nested / "config.json").write_text('{"enabled":true}\n')
        (sandbox / "Dockerfile").write_text(
            "FROM python@sha256:" + "a" * 64 + "\nUSER sandbox\n"
        )
        (sandbox / "docker_image_metadata.json").write_text(json.dumps({
            "version": "5.0",
            "image_id": "sha256:" + "d" * 64,
            "runtime_user": "sandbox",
            "build_context_sha256": docker_build_context_digest(sandbox),
            "smoke_test": {
                "passed": True,
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
                "service_health": True,
                "runtime_tmpfs": {
                    "path": "/app/.runtime",
                    "uid": 10001,
                    "gid": 10001,
                    "mode": "0700",
                },
            },
        }))
        (sandbox / "acceptance_result.json").write_text(
            '{"business_acceptance":"passed"}'
        )
        user_result = {
            "user_query": "accepted", "should_end": True,
            "termination_reason": "completed", "match_status": "matched",
            "outcome_category": "user_acceptance", "reason_code": "accepted",
            "fsm_script_id": "dialogue", "fsm_transition_id": "accept",
            "fsm_state_before": "start", "fsm_state_after": "done",
            "fsm_transition_applied": True, "fsm_recovery_count": 0,
            "attachments": [],
        }
        transition = {
            "step": 0,
            "agent_input": [{"role": "user", "content": "use the tool"}],
            "assistant_output": '{"kind":"respond","content":"done"}',
            "observation": {},
            "action": {"kind": "respond", "content": "done"},
            "result": {"status": 200, "user_query": user_result["user_query"]},
            "next_observation": {},
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "trainer_metadata": {
                "user_simulator": user_result,
            },
        }
        rollout = {
            "schema_version": "2.0",
            "material_visibility_version": "1.0",
            "agent_model": "policy",
            "runtime_model": "simulator",
            "agent_provider_sha256": "a" * 64,
            "runtime_provider_sha256": "b" * 64,
            "runtime_execution": {
                "version": "1.0",
                "mode": "docker_http",
                "container_image_id": "sha256:" + "d" * 64,
                "transport": "loopback_http",
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
            },
            "episodes": [{
                "schema_version": "2.0", "seed": 1, "agent_success": True,
                "termination": "completed", "transitions": [transition],
                "initial_reward": 0.0, "final_reward": 1.0,
                "trajectory": [{
                    "method": "POST", "path": "/v1/reset", "status": 200,
                    "body": {"episode_id": "live-1", "seed": 1}, "result": {},
                }, {
                    "method": "GET", "path": "/v1/state", "status": 200,
                    "result": {"business_state": {}},
                }, {
                    "method": "GET", "path": "/v1/reward", "status": 200,
                    "result": {"reward": 0.0},
                }, {
                    "method": "GET", "path": "/v1/tools", "status": 200,
                    "result": {"tools": []},
                }, {
                    "method": "GET", "path": "/v1/observation", "status": 200,
                    "result": {},
                }, {
                    "method": "POST", "path": "/v1/agent_response",
                    "body": {"content": "done"}, "status": 200, "result": {},
                }, {
                    "method": "POST", "path": "/v1/user_simulator",
                    "status": 200, "result": user_result,
                }, {
                    "method": "GET", "path": "/v1/observation", "status": 200,
                    "result": {},
                }, *[{
                    "method": "GET", "path": "/v1/reward", "status": 200,
                    "result": {"reward": 1.0},
                } for _ in range(3)], {
                    "method": "GET", "path": "/v1/replay", "status": 200,
                    "result": {"events": [{
                        "event": "runtime_llm_call",
                        "payload": {"summary": {
                            "version": "1.0", "responses": 1,
                            "mock_responses": 0, "models": {"simulator": 1},
                            "usage": {"total_tokens": 5},
                            "response_id_sha256": ["8" * 64],
                        }},
                    }]},
                }, {
                    "method": "GET", "path": "/v1/state", "status": 200,
                    "result": {"business_state": {}},
                }],
                "replay": {"events": [{
                    "event": "runtime_llm_call",
                    "payload": {"summary": {
                        "version": "1.0", "responses": 1,
                        "mock_responses": 0,
                        "models": {"simulator": 1},
                        "usage": {"total_tokens": 5},
                        "response_id_sha256": ["8" * 64],
                    }},
                }]},
                "initial_state": {}, "final_state": {},
                "usage": [{
                    "response_model": "policy",
                    "response_id_sha256": "9" * 64,
                    "finish_reason": "stop",
                    "token_usage": {"total_tokens": 10},
                }], "issues": [],
            }],
        }
        (sandbox / "live_rollout.json").write_text(json.dumps(rollout))
        (sandbox / "agentic_training_value_live.json").write_text(json.dumps({
            "curriculum_training_ready": True,
            "validation_mode": "live_evaluator",
            "evaluator_provider": {
                "host": "runtime.example", "model": "simulator",
                "identity_sha256": "b" * 64,
            },
            "runtime_execution": rollout["runtime_execution"],
            "evidence": {"counterfactuals": {
                "goal_success": {"reward": 1.0},
                "goal_failure": {"reward": 0.0},
            }},
        }))
        (sandbox / "data_governance.json").write_text(json.dumps({
            "version": "1.0",
            "eligible_for_external_model_processing": True,
            "data_origin": SYNTHETIC_ORIGIN,
            "providers": {
                "agent": {
                    "host": "agent.example", "model": "policy",
                    "identity_sha256": "a" * 64,
                },
                "user_simulator_and_reward": {
                    "host": "runtime.example", "model": "simulator",
                    "identity_sha256": "b" * 64,
                },
            },
            "outbound_surfaces": OUTBOUND_SURFACES,
            "forbidden_outbound": FORBIDDEN_OUTBOUND,
            "credential_findings": [],
            "pii_findings": [],
        }))
        item = {
            "task_path": str(task),
            "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
            "sandbox_root": str(sandbox),
            "sandbox_evidence_fingerprint": "f" * 64,
            "sandbox_artifacts_sha256": portable_artifact_digests(sandbox),
            "sandbox_evidence_sha256": evidence_artifact_digests(sandbox),
            "category": "simple_agentic",
            "score": 9.0,
            "rollout_sha256": digest_json(rollout),
            "episode_count": 1,
            "successful_episodes": 1,
            "generation_provenance": self.generation_provenance(),
        }
        item["generation_provenance"]["task_sha256"] = item["task_sha256"]
        policy = canonical_certification_policy()
        experiment_config = {
            "certification_profile": "production",
            "threshold": 8.0,
            "validation": "live",
            "sandbox_runtime": "docker",
            "holdout_count": 300,
            "holdout_batches": 3,
            "holdout_rollout_episodes": 10,
            "source_digest": "1" * 64,
            "input_digests": [],
            "bundle_attestation_key_identity_sha256": "e" * 64,
            "generation_provider": {
                "host": "generator.example", "model": "generator-model",
                "identity_sha256": "c" * 64,
            },
            "rollout_provider": {
                "host": "agent.example", "model": "policy",
                "identity_sha256": "a" * 64,
            },
            "runtime_provider": {
                "host": "runtime.example", "model": "simulator",
                "identity_sha256": "b" * 64,
            },
            "generation_model": "generator-model",
            "rollout_model": "policy",
            "runtime_model": "simulator",
            "generation_allowed_response_models": ["generator-model"],
            "rollout_allowed_response_models": ["policy"],
            "runtime_allowed_response_models": ["simulator"],
        }
        experiment_config_sha256 = digest_json(experiment_config)
        experiment_contract = build_experiment_contract(experiment_config)
        manifest = {
            "version": MATERIAL_MANIFEST_VERSION,
            "kind": "agentic_rl_pretraining_materials",
            "evaluator_source_digest": "source",
            "execution_provenance": collect_execution_provenance(ROOT),
            "experiment_config_sha256": experiment_config_sha256,
            "experiment_contract": experiment_contract,
            "experiment_contract_sha256": digest_json(experiment_contract),
            "certification_policy": policy,
            "certification_policy_sha256": digest_json(policy),
            "items": [item],
        }
        manifest["dataset_sha256"] = digest_json(manifest)
        certification = {
            "certification": "production_prepared_for_agentic_rl",
            "scope": "pre_training_material_readiness",
            "certified": True,
            "does_not_certify": [
                "rl_training_execution",
                "downstream_training_system_compatibility",
                "rl_training_convergence",
                "post_training_policy_improvement",
                "cross_model_generalization",
                "exported_rollouts_as_direct_policy_optimization_targets",
            ],
            "policy": policy,
            "measurements": {
                "certification_policy": {
                    "version": policy["version"],
                    "sha256": digest_json(policy),
                    "verified": True,
                },
                "experiment_contract": {
                    "version": experiment_contract["version"],
                    "sha256": digest_json(experiment_contract),
                    "verified": True,
                },
                "training_ready": 1,
                "data_governance": {
                    "reports": 1,
                    "verified": 1,
                    "all_verified": True,
                },
                "production_experiment_profile": True,
                "evaluator_independence": {
                    "same_provider_items": 0,
                    "total_items": 1,
                    "same_provider_rate": 0.0,
                },
                "model_response_provenance": {
                    "verified": 1,
                    "expected": 1,
                    "all_verified": True,
                    "agent_response_models": {"policy": 1},
                    "runtime_response_models": {"simulator": 1},
                },
                "model_response_authorization": {
                    "all_authorized": True,
                    "allowed": {
                        "generation": ["generator-model"],
                        "agent": ["policy"],
                        "runtime": ["simulator"],
                    },
                    "actual": {
                        "generation": {"generator-model": 2},
                        "agent": {"policy": 1},
                        "runtime": {"simulator": 1},
                    },
                },
                "production_preflight": {
                    "version": "1.3",
                    "experiment_config_sha256": experiment_config_sha256,
                    "scope": "production_pre_training_material_experiment",
                    "network_probe_performed": False,
                    "ready": True,
                    "failed_checks": [],
                    "checks": [
                        {
                            "name": name,
                            "passed": True,
                            "evidence": (
                                {
                                    "generation_provider": {
                                        "host": "generator.example",
                                        "model": "generator-model",
                                        "identity_sha256": "c" * 64,
                                    },
                                    "agent_provider": {
                                        "host": "agent.example",
                                        "model": "policy",
                                        "identity_sha256": "a" * 64,
                                    },
                                    "runtime_provider": {
                                        "host": "runtime.example",
                                        "model": "simulator",
                                        "identity_sha256": "b" * 64,
                                    },
                                    "generation_allowed_response_models": [
                                        "generator-model"
                                    ],
                                    "agent_allowed_response_models": ["policy"],
                                    "runtime_allowed_response_models": ["simulator"],
                                    "response_allowlists_valid": True,
                                }
                                if name == "model_configuration" else {
                                    "key_identity_sha256": "e" * 64,
                                    "bundle_version": exporter.BUNDLE_VERSION,
                                }
                                if name == "bundle_signing_identity" else {}
                                if name != "evaluator_role_separation" else {
                                    "agent_and_evaluator_distinct": True,
                                    "provider_hosts_distinct": True,
                                    "models_distinct": True,
                                }
                            ),
                        }
                        for name in sorted(exporter.REQUIRED_CHECKS)
                    ],
                },
            },
            "gates": {
                "fixture": True,
                "container_rollout_execution": True,
                "container_reward_calibration": True,
                "provider_identity_consistency": True,
                "data_governance": True,
                "production_experiment_profile": True,
                "production_preflight": True,
                "evaluator_independence": True,
                "model_response_provenance": True,
                "model_response_authorization": True,
                "certification_policy": True,
                "portable_experiment_contract": True,
            },
            "failed_gates": [],
            "material_verification": {"verified": True},
            "materials_manifest": manifest,
        }
        return certification

    def test_export_is_portable_and_detects_post_export_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            report = exporter.export_bundle(certification, bundle, ROOT)
            self.assertTrue(report["verified"], report)
            self.assertEqual(report["items"], 1)
            self.assertEqual(report["transitions"], 1)
            self.assertTrue(report["production_contract_ready"])
            self.assertTrue(report["consumer_records_ready"])
            self.assertTrue(report["container_rollout_ready"])
            self.assertTrue(report["portable_build_context_ready"])
            record = json.loads((bundle / "transitions.jsonl").read_text())
            self.assertEqual(record["transition"]["reward"], 1.0)
            self.assertEqual(record["episode_final_reward"], 1.0)
            self.assertEqual(record["agent_usage"]["response_model"], "policy")
            self.assertRegex(record["agent_usage"]["response_id_sha256"], r"^[0-9a-f]{64}$")
            manifest = json.loads((bundle / "bundle_manifest.json").read_text())
            self.assertEqual(
                record["task_family_id"], manifest["items"][0]["task_family_id"]
            )
            self.assertNotIn("trainer_metadata", record["transition"])
            card = json.loads((bundle / "dataset_card.json").read_text())
            self.assertEqual(card["composition"]["items"], 1)

            self.assertEqual(card["composition"]["transitions"], 1)
            self.assertEqual(
                card["composition"]["task_generation"]["configured_models"],
                {"generator-model": 1},
            )
            self.assertEqual(
                card["composition"]["task_generation"]["actual_response_models"],
                {"generator-model": 2},
            )
            self.assertEqual(
                card["composition"]["rollout_response_provenance"],
                {
                    "agent_models": {"policy": 1},
                    "runtime_models": {"simulator": 1},
                },
            )
            self.assertEqual(
                card["composition"]["runtime_execution"],
                {
                    "modes": {"docker_http": 1},
                    "validated_container_items": 1,
                    "unique_container_images": 1,
                },
            )
            self.assertEqual(
                card["composition"]["reward_calibration_execution"],
                {
                    "modes": {"docker_http": 1},
                    "validated_container_items": 1,
                    "unique_container_images": 1,
                },
            )
            self.assertEqual(
                card["distribution_status"],
                "internal_only_until_legal_and_security_review",
            )
            portable = json.loads((bundle / "certification.json").read_text())
            self.assertNotIn("materials_manifest", portable)
            self.assertTrue(portable["certified"])
            self.assertEqual(
                card["build_environment"], portable["execution_provenance"]
            )
            self.assertEqual(
                json.loads((bundle / "bundle_manifest.json").read_text())[
                    "execution_provenance_sha256"
                ],
                digest_json(card["build_environment"]),
            )
            contract = json.loads((bundle / "consumer_contract.json").read_text())
            self.assertEqual(contract["bundle_version"], exporter.BUNDLE_VERSION)
            experiment = json.loads(
                (bundle / "experiment_contract.json").read_text()
            )
            self.assertEqual(
                experiment,
                certification["materials_manifest"]["experiment_contract"],
            )
            self.assertEqual(
                contract["records"]["policy_transition_fields"],
                exporter.consumer_contract()["records"]["policy_transition_fields"],
            )
            copied_app = next((bundle / "environments").glob("*/app.py"))
            copied_environment = copied_app.parent
            self.assertTrue((copied_environment / "business/rules/engine.py").is_file())
            self.assertTrue((copied_environment / "business/rules/config.json").is_file())
            copied_app.write_text("# tampered\n")
            changed = exporter.verify_bundle(bundle)
            self.assertFalse(changed["verified"])
            self.assertIn("bundle_files", changed["failed_gates"])

    def test_export_rejects_rehashed_source_episode_success_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            manifest = certification["materials_manifest"]
            manifest["items"][0]["successful_episodes"] = 0
            manifest["dataset_sha256"] = digest_json({
                key: value for key, value in manifest.items()
                if key != "dataset_sha256"
            })
            with self.assertRaisesRegex(
                ValueError, "source material verification failed"
            ):
                exporter.export_bundle(certification, root / "bundle", ROOT)

    def test_transition_termination_must_match_user_simulator_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            certification = self.source(Path(directory))
            item = certification["materials_manifest"]["items"][0]
            rollout = json.loads(
                (Path(item["sandbox_root"]) / "live_rollout.json").read_text()
            )
            rollout["episodes"][0]["transitions"][0]["terminated"] = False
            with self.assertRaisesRegex(ValueError, "user_simulator_terminal"):
                exporter._transition_records("fixture", item, rollout)

    def test_export_redacts_machine_local_measurement_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            certification["measurements"]["diagnostic"] = {
                "task_path": "/Users/person/private/task.json",
                "message": "/private/tmp/build-attempt",
            }
            bundle = root / "bundle"
            report = exporter.export_bundle(certification, bundle, ROOT)
            self.assertTrue(report["metadata_privacy_ready"])
            portable = json.loads((bundle / "certification.json").read_text())
            diagnostic = portable["measurements"]["diagnostic"]
            self.assertNotIn("task_path", diagnostic)
            self.assertEqual(
                diagnostic["message"], "<redacted-local-path>"
            )

    def test_export_rejects_credential_in_portable_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            certification["measurements"]["diagnostic"] = (
                "sk-abcdefghijklmnopqrstuvwxyz123456"
            )
            with self.assertRaisesRegex(
                ValueError, "non-portable or sensitive metadata"
            ):
                exporter.export_bundle(
                    certification, root / "bundle", ROOT
                )

    def test_v17_verifier_rejects_rehashed_missing_preflight_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            certification_path = bundle / "certification.json"
            portable = json.loads(certification_path.read_text())
            portable["gates"].pop("production_preflight")
            portable["measurements"].pop("production_preflight")
            certification_path.write_text(json.dumps(portable))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["certification.json"] = (
                exporter.file_sha256(certification_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["production_preflight_ready"])
            self.assertIn("portable_certification", report["failed_gates"])

    def test_v17_verifier_binds_preflight_to_environment_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            certification_path = bundle / "certification.json"
            portable = json.loads(certification_path.read_text())
            checks = portable["measurements"]["production_preflight"]["checks"]
            model_evidence = next(
                check["evidence"] for check in checks
                if check["name"] == "model_configuration"
            )
            model_evidence["agent_provider"] = {
                "host": "other.example",
                "model": "other-policy",
                "identity_sha256": "d" * 64,
            }
            certification_path.write_text(json.dumps(portable))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["certification.json"] = (
                exporter.file_sha256(certification_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["production_preflight_ready"])
            self.assertIn("preflight_provider_binding", report["failed_gates"])

    def test_v17_verifier_rejects_rehashed_experiment_binding_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["experiment_config_sha256"] = "0" * 64
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["experiment_config_ready"])
            self.assertIn("portable_certification", report["failed_gates"])

    def test_v17_verifier_rejects_rehashed_relaxed_certification_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)

            certification_path = bundle / "certification.json"
            certification = json.loads(certification_path.read_text())
            certification["policy"]["min_tasks"] = 1
            policy_digest = digest_json(certification["policy"])
            certification["measurements"]["certification_policy"][
                "sha256"
            ] = policy_digest
            certification_path.write_text(json.dumps(certification))

            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["certification"]["policy_sha256"] = policy_digest
            card_path.write_text(json.dumps(card))

            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["certification_policy_sha256"] = policy_digest
            manifest["files_sha256"]["certification.json"] = (
                exporter.file_sha256(certification_path)
            )
            manifest["files_sha256"]["dataset_card.json"] = (
                exporter.file_sha256(card_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))

            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["certification_policy_ready"])
            self.assertIn("certification_policy", report["failed_gates"])

    def test_v17_verifier_rejects_rehashed_relaxed_experiment_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)

            contract_path = bundle / "experiment_contract.json"
            contract = json.loads(contract_path.read_text())
            contract["configuration"]["holdout_count"] = 1
            contract["configuration_sha256"] = digest_json(
                contract["configuration"]
            )
            contract_digest = digest_json(contract)
            contract_path.write_text(json.dumps(contract))

            certification_path = bundle / "certification.json"
            certification = json.loads(certification_path.read_text())
            certification["experiment_contract_sha256"] = contract_digest
            certification["measurements"]["experiment_contract"][
                "sha256"
            ] = contract_digest
            certification_path.write_text(json.dumps(certification))

            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["certification"][
                "experiment_contract_sha256"
            ] = contract_digest
            card_path.write_text(json.dumps(card))

            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["experiment_contract_sha256"] = contract_digest
            for relative in (
                "experiment_contract.json", "certification.json",
                "dataset_card.json",
            ):
                manifest["files_sha256"][relative] = exporter.file_sha256(
                    bundle / relative
                )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))

            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["experiment_contract_ready"])
            self.assertIn("experiment_contract", report["failed_gates"])

    def test_v17_verifier_rejects_direct_training_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record["direct_training_status"] = "certified"
            transition_path.write_text(json.dumps(record) + "\n")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["transitions.jsonl"] = (
                exporter.file_sha256(transition_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["consumer_records_ready"])
            self.assertIn(
                "record:direct_training_status",
                report["consumer_record_failures"][0]["errors"],
            )
            self.assertFalse(report["trajectory_purpose_ready"])
            self.assertIn("transition_projection", report["failed_gates"])

    def test_v17_verifier_rejects_rehashed_response_model_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["items"][0]["model_response_provenance"][
                "agent_response_models"
            ] = {"forged-model": 1}
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["model_response_provenance_ready"])
            self.assertIn("model_response_provenance", report["failed_gates"])

    def test_v17_verifier_rejects_rehashed_response_authorization_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            exporter.export_bundle(self.source(root), bundle, ROOT)
            certification_path = bundle / "certification.json"
            portable = json.loads(certification_path.read_text())
            portable["measurements"]["model_response_authorization"][
                "allowed"
            ]["agent"] = ["forged-policy"]
            certification_path.write_text(json.dumps(portable))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["certification.json"] = (
                exporter.file_sha256(certification_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["model_response_authorization_ready"])
            self.assertIn("model_response_authorization", report["failed_gates"])


    def test_v17_verifier_rejects_rehashed_local_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            certification_path = bundle / "certification.json"
            portable = json.loads(certification_path.read_text())
            portable["measurements"]["diagnostic"] = {
                "sandbox_root": "/Users/person/private/sandbox"
            }
            certification_path.write_text(json.dumps(portable))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["certification.json"] = (
                exporter.file_sha256(certification_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["metadata_privacy_ready"])
            self.assertIn("portable_metadata_privacy", report["failed_gates"])

    def test_bundle_verifier_rejects_semantically_rewritten_transition_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            record = json.loads((bundle / "transitions.jsonl").read_text())
            record["transition"]["terminated"] = False
            (bundle / "transitions.jsonl").write_text(json.dumps(record) + "\n")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                bundle / "transitions.jsonl"
            )
            unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("transition_projection", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_misleading_dataset_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["composition"]["transitions"] = 999
            card["license_status"] = "public_domain"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(card_path)
            unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("dataset_card", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_generation_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["items"][0]["generation_provenance"]["sample_seed"] = 999
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("generation_provenance", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_container_rollout_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            rollout_path = bundle / item["environment_path"] / "live_rollout.json"
            rollout = json.loads(rollout_path.read_text())
            rollout["runtime_execution"]["container_image_id"] = (
                "sha256:" + "e" * 64
            )
            rollout_path.write_text(json.dumps(rollout))
            relative = str(rollout_path.relative_to(bundle))
            digest = exporter.file_sha256(rollout_path)
            item["files_sha256"]["live_rollout.json"] = digest
            manifest["files_sha256"][relative] = digest
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["container_rollout_ready"])
            self.assertIn("container_rollout_execution", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_reward_container_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            calibration_path = (
                bundle / item["environment_path"]
                / "agentic_training_value_live.json"
            )
            calibration = json.loads(calibration_path.read_text())
            changed_image = "sha256:" + "e" * 64
            calibration["runtime_execution"]["container_image_id"] = changed_image
            calibration_path.write_text(json.dumps(calibration))
            item["reward_runtime_execution"]["container_image_id"] = changed_image
            relative = str(calibration_path.relative_to(bundle))
            digest = exporter.file_sha256(calibration_path)
            item["files_sha256"]["agentic_training_value_live.json"] = digest
            manifest["files_sha256"][relative] = digest
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["container_reward_calibration_ready"])
            self.assertIn(
                "container_reward_calibration", report["failed_gates"]
            )

    def test_bundle_verifier_rejects_rehashed_reward_provider_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            calibration_path = (
                bundle / item["environment_path"]
                / "agentic_training_value_live.json"
            )
            calibration = json.loads(calibration_path.read_text())
            calibration["evaluator_provider"]["identity_sha256"] = "f" * 64
            calibration_path.write_text(json.dumps(calibration))
            relative = str(calibration_path.relative_to(bundle))
            digest = exporter.file_sha256(calibration_path)
            item["files_sha256"]["agentic_training_value_live.json"] = digest
            manifest["files_sha256"][relative] = digest
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["provider_identity_ready"])
            self.assertIn("provider_identity_binding", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_task_lineage_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            lineage_path = (
                bundle / item["environment_path"] / "task_lineage.json"
            )
            lineage = json.loads(lineage_path.read_text())
            lineage["source_file_sha256"] = "f" * 64
            lineage_path.write_text(json.dumps(lineage))
            relative = str(lineage_path.relative_to(bundle))
            digest = exporter.file_sha256(lineage_path)
            item["files_sha256"]["task_lineage.json"] = digest
            manifest["files_sha256"][relative] = digest
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["task_lineage_ready"])
            self.assertIn("task_lineage", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_environment_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["build_environment"]["python"]["version"] = "0.0"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("dataset_card", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_consumer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract = json.loads(contract_path.read_text())
            contract["visibility"]["trainer_only"] = "transitions.jsonl"
            contract_path.write_text(json.dumps(contract))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["consumer_contract.json"] = exporter.file_sha256(
                contract_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["production_contract_ready"])
            self.assertIn("consumer_contract", report["failed_gates"])

    def test_legacy_v3_bundle_keeps_integrity_status_but_not_production_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.0"
            card.pop("consumer_contract")
            card_path.write_text(json.dumps(card))
            contract_path = bundle / "consumer_contract.json"
            contract_path.unlink()
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "3.0"
            manifest.pop("consumer_contract_file")
            manifest["items"][0].pop("split")
            manifest["items"][0].pop("task_family_id")
            manifest["items"][0].pop("generation_provenance")
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record.pop("split")
            record.pop("task_family_id")
            record.pop("generation_model")
            record.pop("generation_provider_identity_sha256")
            record.pop("generation_sample_seed")
            transition_path.write_text(json.dumps(record) + "\n")
            self.remove_current_trajectory_purpose(bundle, manifest)
            manifest["files_sha256"].pop("consumer_contract.json")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertFalse(report["production_contract_ready"])

    def test_legacy_v4_bundle_keeps_contract_but_has_no_trusted_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract = exporter.consumer_contract("4.0")
            contract_path.write_text(json.dumps(contract))
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.1"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "4.0"
            manifest.pop("attestation")
            manifest["items"][0].pop("split")
            manifest["items"][0].pop("task_family_id")
            manifest["items"][0].pop("generation_provenance")
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record.pop("split")
            record.pop("task_family_id")
            record.pop("generation_model")
            record.pop("generation_provider_identity_sha256")
            record.pop("generation_sample_seed")
            transition_path.write_text(json.dumps(record) + "\n")
            self.remove_current_trajectory_purpose(bundle, manifest)
            manifest["files_sha256"]["consumer_contract.json"] = exporter.file_sha256(
                contract_path
            )
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["production_contract_ready"])
            self.assertFalse(report["trusted_attestation"])

    def test_legacy_v8_bundle_keeps_generation_contract_without_container_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract_path.write_text(json.dumps(exporter.consumer_contract("8.0")))
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.4"
            card["composition"].pop("runtime_execution")
            card["composition"].pop("reward_calibration_execution")
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "8.0"
            self.remove_current_trajectory_purpose(bundle, manifest)
            manifest["items"][0].pop("runtime_execution")
            manifest["items"][0].pop("reward_runtime_execution")
            manifest["files_sha256"]["consumer_contract.json"] = (
                exporter.file_sha256(contract_path)
            )
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["generation_provenance_ready"])
            self.assertFalse(report["container_rollout_ready"])

    def test_legacy_v9_bundle_keeps_rollout_claim_without_reward_container_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract_path.write_text(json.dumps(exporter.consumer_contract("9.0")))
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.5"
            card["composition"].pop("reward_calibration_execution")
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "9.0"
            self.remove_current_trajectory_purpose(bundle, manifest)
            manifest["items"][0].pop("reward_runtime_execution")
            manifest["files_sha256"]["consumer_contract.json"] = (
                exporter.file_sha256(contract_path)
            )
            manifest["files_sha256"]["dataset_card.json"] = (
                exporter.file_sha256(card_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["container_rollout_ready"])
            self.assertFalse(report["container_reward_calibration_ready"])

    def test_legacy_v10_bundle_keeps_reward_claim_without_v11_identity_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract_path.write_text(json.dumps(exporter.consumer_contract("10.0")))
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.6"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "10.0"
            self.remove_current_trajectory_purpose(bundle, manifest)
            item = manifest["items"][0]
            lineage_path = bundle / item["environment_path"] / "task_lineage.json"
            lineage_relative = str(lineage_path.relative_to(bundle))
            lineage_path.unlink()
            item["files_sha256"].pop("task_lineage.json")
            manifest["files_sha256"].pop(lineage_relative)
            manifest["files_sha256"]["consumer_contract.json"] = (
                exporter.file_sha256(contract_path)
            )
            manifest["files_sha256"]["dataset_card.json"] = (
                exporter.file_sha256(card_path)
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["container_reward_calibration_ready"])
            self.assertFalse(report["provider_identity_ready"])
            self.assertFalse(report["task_lineage_ready"])

    def test_rehashed_bundle_cannot_remove_environment_rebuild_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            relative = f"{item['environment_path']}/Dockerfile"
            (bundle / relative).unlink()
            item["files_sha256"].pop("Dockerfile")
            manifest["files_sha256"].pop(relative)
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["portable_build_context_ready"])
            self.assertIn("portable_build_context", report["failed_gates"])
            self.assertIn(
                "environment_reconstruction_contract", report["failed_gates"]
            )

    def test_signed_bundle_requires_the_trusted_key_and_detects_reserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.signing_keys(root)
            certification = self.source(root)
            signing_check = next(
                check
                for check in certification["measurements"]
                    ["production_preflight"]["checks"]
                if check["name"] == "bundle_signing_identity"
            )
            signing_check["evidence"]["key_identity_sha256"] = (
                public_key_identity(public)
            )
            bundle = root / "bundle"
            report = exporter.export_bundle(
                certification, bundle, ROOT,
                signing_private_key=private, trusted_public_key=public,
            )
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["trusted_attestation"])
            without_trust = exporter.verify_bundle(bundle)
            self.assertTrue(without_trust["verified"])
            self.assertFalse(without_trust["trusted_attestation"])

            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
            changed = exporter.verify_bundle(bundle, trusted_public_key=public)
            self.assertFalse(changed["verified"])
            self.assertFalse(changed["trusted_attestation"])
            self.assertIn("trusted_attestation", changed["failed_gates"])

    def test_signed_bundle_rejects_preflight_from_another_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.signing_keys(root)
            certification = self.source(root)
            with self.assertRaisesRegex(
                ValueError, "does not match bundle signing identity"
            ):
                exporter.export_bundle(
                    certification,
                    root / "bundle",
                    ROOT,
                    signing_private_key=private,
                    trusted_public_key=public,
                )

    def test_rehashed_item_and_records_cannot_change_content_based_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            self.assertEqual(item["split"], "train")
            item["split"] = "test"

            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record["split"] = "test"
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )

            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["composition"]["splits"] = {
                "train": 0, "validation": 0, "test": 1,
            }
            card["composition"]["category_splits"]["simple_agentic"] = {
                "train": 0, "validation": 0, "test": 1,
            }
            card_path.write_text(json.dumps(card))
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("dataset_split_assignment", report["failed_gates"])

    def test_rehashed_bundle_cannot_forge_task_family_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["items"][0]["task_family_id"] = "0" * 64
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record["task_family_id"] = "0" * 64
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("task_family_identity", report["failed_gates"])

    def test_uncertified_report_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            certification["certified"] = False
            with self.assertRaisesRegex(ValueError, "certified"):
                exporter.export_bundle(certification, root / "bundle", ROOT)

    def test_export_rejects_action_that_does_not_match_raw_model_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            item = certification["materials_manifest"]["items"][0]
            rollout = json.loads(
                (Path(item["sandbox_root"]) / "live_rollout.json").read_text()
            )
            rollout["episodes"][0]["transitions"][0]["action"] = {
                "kind": "respond", "content": "fabricated",
            }
            with self.assertRaisesRegex(ValueError, "parsed_action_mismatch"):
                exporter._transition_records("a" * 16 + "-" + "b" * 16, item, rollout)


if __name__ == "__main__":
    unittest.main()
