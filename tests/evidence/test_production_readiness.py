import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from env_factory.contracts.reward_contract import REFERENCE_FACTUALITY_CRITERION
from env_factory.sandbox_scoring import (
    SCORE_RUBRIC,
    evidence_fingerprint,
    sandbox_quality_factors,
    valid_score_report,
)
from env_factory.tasks.task_quality import score_file
from env_factory.task_pipeline import TaskGenerationPipeline
from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE,
    MATERIAL_MANIFEST_VERSION,
    docker_build_context_digest,
)
from env_factory.evidence.trajectory_schema import episode_errors

ROOT = Path(__file__).resolve().parents[2]


def load_script(name="certify_training_materials"):
    path = ROOT / f"scripts/{name}.py"
    if not path.is_file():
        path = ROOT / f"scripts/rollout/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


certifier = load_script()
verifier = load_script("verify_training_materials")
rollout_runner = load_script("run_live_rollout")


class ProductionReadinessTest(unittest.TestCase):
    @staticmethod
    def production_preflight(config):
        agent_provider = {
            "host": "agent.example", "model": "policy-model",
            "identity_sha256": "a" * 64,
        }
        generation_provider = {
            "host": "generator.example", "model": "generator-model",
            "identity_sha256": "c" * 64,
        }
        runtime_provider = {
            "host": "runtime.example", "model": "simulator-model",
            "identity_sha256": "b" * 64,
        }
        return {
            "version": "1.3",
            "experiment_config_sha256": certifier.digest_json(config),
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
                            "generation_provider": generation_provider,
                            "agent_provider": agent_provider,
                            "runtime_provider": runtime_provider,
                            "generation_allowed_response_models": config[
                                "generation_allowed_response_models"
                            ],
                            "agent_allowed_response_models": config[
                                "rollout_allowed_response_models"
                            ],
                            "runtime_allowed_response_models": config[
                                "runtime_allowed_response_models"
                            ],
                            "response_allowlists_valid": True,
                        }
                        if name == "model_configuration" else {
                            "key_identity_sha256": "e" * 64,
                            "bundle_version": certifier.BUNDLE_VERSION,
                        }
                        if name == "bundle_signing_identity" else {}
                        if name != "evaluator_role_separation" else {
                            "agent_and_evaluator_distinct": True,
                            "provider_hosts_distinct": True,
                            "models_distinct": True,
                        }
                    ),
                }
                for name in sorted(certifier.REQUIRED_CHECKS)
            ],
        }

    @staticmethod
    def sandbox_revalidation(history):
        revalidation = {}
        for holdout in history.get("holdouts", []):
            for job in holdout.get("jobs", []):
                result = job.get("result", {})
                report = result.get("sandbox_score")
                if not isinstance(report, dict):
                    continue
                root = Path(str(result.get("output", "")))
                task_path = Path(str(result.get("task_path", "")))
                revalidation[certifier.sandbox_revalidation_key(
                    root, task_path
                )] = {
                    "verified": True,
                    "exit_code": 0,
                    "report": report,
                    "error": None,
                }
        return revalidation

    def certify(self, history, *, image_inspector=None):
        if image_inspector is None:
            image_inspector = lambda tag: {
                "image_id": "sha256:" + "d" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
                "runtime_user": "sandbox",
            } if tag == "fixture" else None
        return certifier.certify(
            history,
            certifier.default_policy(),
            sandbox_revalidation=self.sandbox_revalidation(history),
            production_preflight=self.production_preflight(history["config"]),
            image_inspector=image_inspector,
        )

    @staticmethod
    def fixture_user_turn(episode: int) -> dict:
        if episode < 8:
            return {
                "user_query": "accepted", "should_end": True,
                "termination_reason": "completed", "match_status": "matched",
                "outcome_category": "user_acceptance",
                "reason_code": "fixture_acceptance",
                "fsm_script_id": "fixture-dialogue",
                "fsm_transition_id": "accept", "fsm_state_before": "start",
                "fsm_state_after": "done", "fsm_transition_applied": True,
                "fsm_recovery_count": 0,
            }
        if episode == 8:
            return {
                "user_query": "need details", "should_end": False,
                "match_status": "matched",
                "outcome_category": "information_required",
                "reason_code": "fixture_information",
                "fsm_script_id": "fixture-dialogue",
                "fsm_transition_id": "clarify", "fsm_state_before": "start",
                "fsm_state_after": "clarification",
                "fsm_transition_applied": True, "fsm_recovery_count": 0,
            }
        return {
            "user_query": "try again", "should_end": False,
            "match_status": "unmatched", "outcome_category": "agent_off_topic",
            "reason_code": "fixture_recovery",
            "fsm_script_id": "fixture-dialogue", "fsm_transition_id": None,
            "fsm_state_before": "start", "fsm_state_after": "start",
            "fsm_transition_applied": False, "fsm_recovery_count": 1,
        }

    @staticmethod
    def fixture_task(category: str, description: str) -> dict:
        task = {
            "training_category": "simple_agentic",
            "task": description,
            "task_intent": "recommend",
            "complexity": "standard",
            "requirements": {
                "output_format": "markdown",
                "runtime_interface": TaskGenerationPipeline._build_runtime_interface([], {}),
            },
            "environment_plan": {"mode": "reference_data"},
            "actions": [
                {"name": "读取库存"}, {"name": "筛选候选"},
                {"name": "比较候选"}, {"name": "形成建议"},
            ],
            "tools": [
                {"function": {"name": "read_inventory"}},
                {"function": {"name": "get_weather"}},
            ],
            "noise_tools": [{"name": "get_weather", "category": "unrelated"}],
            "metrics": [
                {"id": "process_read", "category": "process", "type": "hybrid", "evaluator": {}},
                {"id": "outcome", "category": "outcome", "type": "model-based",
                 "evaluator": {"kind": "external_llm_judge"},
                 "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
                 "criteria": [REFERENCE_FACTUALITY_CRITERION]},
                {"id": "noise", "category": "penalty", "type": "rule-based", "evaluator": {}},
            ],
            "metric_implementations": [
                {"metric_id": "process_read"}, {"metric_id": "noise"},
            ],
            "reward_formula": {"score_range": [-1, 1]},
            "acceptance_contract": {
                "executable_scenarios": [
                    {"kind": "goal_success", "steps": [{
                        "operation": "tool_call", "tool_name": "read_inventory",
                        "arguments": {"category": "办公设备"},
                    }]},
                    {"kind": "goal_failure"}, {"kind": "noise_selection"},
                ],
                "mutation_tests": [{"id": "constant_reward"}],
            },
            "task_readiness": {"ready": True, "warnings": []},
            "user_scripts": [{
                "script_id": "fixture-dialogue",
                "initial_state": "start",
                "states": [
                    {"state_id": "start", "terminal": False},
                    {"state_id": "clarification", "terminal": False},
                    {"state_id": "done", "terminal": True},
                ],
                "transitions": [
                    {
                        "transition_id": "accept",
                        "outcome_category": "user_acceptance",
                        "from_state": "start", "to_state": "done",
                        "should_end": True,
                    },
                    {
                        "transition_id": "clarify",
                        "outcome_category": "information_required",
                        "from_state": "start", "to_state": "clarification",
                        "should_end": False,
                    },
                ],
            }],
            "task_spec": {
                "version": "1.0", "task_contract": {},
                "training_contract": {
                    "category": "simple_agentic", "environment_archetype": "single_read",
                },
                "environment_contract": {"mode": "reference_data", "archetype": "single_read"},
                "tool_contracts": [{"name": "read_inventory", "role": "business"}],
                "capability_dag": {"nodes": ["read_inventory"], "edges": []},
                "goal_contract": {"expected_delta": []},
            },
            "artifacts": {"data_manifest": {
                "version": "1.0", "environment_mode": "reference_data",
                "root": "data/business_data",
                "tables": [{
                    "table_name": "inventory",
                    "schema_file": "schemas/inventory.json",
                    "rows_file": "rows/inventory.jsonl",
                }],
                "data_governance": {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            }}},
        }
        if category == "direct_response":
            task["training_category"] = category
            task["environment_plan"] = {"mode": "stateless"}
            task["artifacts"]["data_manifest"]["environment_mode"] = "stateless"
            task["artifacts"]["data_manifest"]["tables"] = []
            task["tools"] = [task["tools"][1]]
            task["acceptance_contract"]["executable_scenarios"] = [
                {"kind": "goal_success", "steps": [{"operation": "respond"}]},
                {"kind": "goal_failure"}, {"kind": "noise_selection"},
            ]
            task["task_spec"] = {
                "version": "1.0", "task_contract": {},
                "training_contract": {"category": category, "environment_archetype": "text_only"},
                "environment_contract": {"mode": "stateless", "archetype": "text_only"},
                "tool_contracts": [{"name": "get_weather", "role": "noise"}],
                "capability_dag": {"nodes": [], "edges": []},
                "goal_contract": {"expected_delta": []},
            }
        elif category == "multi_step_agentic":
            task["training_category"] = category
            task["tools"].insert(1, {"function": {"name": "rank_inventory"}})
            task["acceptance_contract"]["executable_scenarios"][0]["steps"] = [
                {
                    "operation": "tool_call", "tool_name": "read_inventory",
                    "arguments": {"category": "办公设备"},
                    "capture": {"inventory": "$.items"},
                },
                {
                    "operation": "tool_call", "tool_name": "rank_inventory",
                    "arguments": {"items": {"$ref": "inventory"}},
                },
            ]
            task["task_spec"] = {
                "version": "1.0", "task_contract": {},
                "training_contract": {"category": category, "environment_archetype": "multi_read_join"},
                "environment_contract": {"mode": "reference_data", "archetype": "multi_read_join"},
                "tool_contracts": [
                    {"name": "read_inventory", "role": "business"},
                    {"name": "rank_inventory", "role": "business"},
                    {"name": "get_weather", "role": "noise"},
                ],
                "capability_dag": {
                    "nodes": ["read_inventory", "rank_inventory"],
                    "edges": [{
                        "from_tool": "read_inventory", "to_tool": "rank_inventory",
                        "via": "inventory",
                    }],
                },
                "goal_contract": {"expected_delta": []},
            }
        return task

    def make_history(self, root: Path, *, count: int = 10, batches: int = 3):
        # Gate-specific failures use smaller evidence; the positive test uses
        # 3 x 300 to meet the reward-error confidence bound.
        evidence = root / "evidence"
        evidence.mkdir()
        business = root / "data/business_data"
        (business / "schemas").mkdir(parents=True)
        (business / "rows").mkdir()
        (business / "schemas/inventory.json").write_text(json.dumps({
            "table_name": "inventory",
            "columns": [
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "category", "type": "text", "nullable": False},
            ],
            "primary_key": ["id"], "foreign_keys": [], "indexes": [], "constraints": [],
        }))
        (business / "rows/inventory.jsonl").write_text(
            json.dumps({"id": 1, "category": "办公设备"}) + "\n"
        )
        (evidence / "app.py").write_text("# immutable sandbox\n")
        (evidence / ".dockerignore").write_text(DOCKERIGNORE_SOURCE)
        pinned_image = "registry.example/python@sha256:" + "c" * 64
        (evidence / "Dockerfile").write_text(
            f"FROM {pinned_image}\nUSER sandbox\n"
        )
        (evidence / "requirements-dev.txt").write_text("pytest==9.1.1\n")
        (evidence / "python_packages.json").write_text(json.dumps({
            "version": "1.0",
            "packages": [{"name": "pytest", "version": "9.1.1"}],
        }))
        (evidence / "docker_image_metadata.json").write_text(json.dumps({
            "version": "5.0",
            "tag": "fixture",
            "base_image": pinned_image,
            "image_id": "sha256:" + "d" * 64,
            "platform": {"os": "linux", "architecture": "arm64"},
            "runtime_user": "sandbox",
            "dockerfile_sha256": __import__("hashlib").sha256(
                (evidence / "Dockerfile").read_bytes()
            ).hexdigest(),
            "requirements_sha256": __import__("hashlib").sha256(
                (evidence / "requirements-dev.txt").read_bytes()
            ).hexdigest(),
            "python_packages_sha256": __import__("hashlib").sha256(
                (evidence / "python_packages.json").read_bytes()
            ).hexdigest(),
            "build_context_sha256": docker_build_context_digest(evidence),
            "smoke_test": {
                "passed": True,
                "network": "none",
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
        (evidence / "training_readiness.json").write_text(json.dumps({
            "training_ready": True,
            "validation_mode": "offline_mock",
            "live_rollout_verified": False,
            "hard_gates_passed": True,
            "failed_gates": [],
            "failures": [],
            "evidence": {
                "determinism": True,
                "state_causality": {
                    "archetype": "fixture", "expected_delta": [], "invalid": [],
                },
                "rewards": {"success": 1.0, "failure": 0.0},
                "observation_scan": {"status": 200, "forbidden_paths": []},
                "runtime_state": {
                    "reset_reproducible": True,
                    "episode_isolation": True,
                    "replay_consistent": True,
                    "episode_a_event_count": 1,
                    "episode_b_event_count": 0,
                },
            },
        }))
        (evidence / "agentic_training_value.json").write_text(json.dumps({
            "curriculum_training_ready": True,
            "evidence": {"business_data_access": {"tables_read": ["inventory"], "tool_reads": [
                {"tool_name": "read_inventory", "tables_read": ["inventory"]},
                {"tool_name": "rank_inventory", "tables_read": ["inventory"]},
            ]},
                         "business_data_causality": {"proved": True, "table": "inventory", "tool": "lookup",
                                                     "field": "quantity", "answer_value_mentioned": False},
                         "reference_data_immutability": {"unchanged": True},
                         "counterfactuals": {
                "goal_success": {"reward": 1.0},
                "goal_failure": {"reward": 0.0},
                "no_tools": {"reward": 0.0},
            }},
        }))
        (evidence / "agentic_training_value_live.json").write_text(json.dumps({
            "curriculum_training_ready": True,
            "agentic_training_ready": True,
            "hard_gates_passed": True,
            "validation_mode": "live_evaluator",
            "evaluator_provider": {
                "host": "runtime.example", "model": "simulator-model",
                "identity_sha256": "b" * 64,
            },
            "failed_gates": [],
            "failures": [],
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
            "evidence": {"business_data_access": {"tables_read": ["inventory"], "tool_reads": [
                {"tool_name": "read_inventory", "tables_read": ["inventory"]},
                {"tool_name": "rank_inventory", "tables_read": ["inventory"]},
            ]},
                         "business_data_causality": {"proved": True, "table": "inventory", "tool": "lookup",
                                                     "field": "quantity", "answer_value_mentioned": False},
                         "reference_data_immutability": {"unchanged": True},
                         "counterfactuals": {
                "goal_success": {"reward": 1.0, "status": "completed"},
                "goal_failure": {"reward": 0.0, "status": "completed"},
                "no_tools": {"reward": 0.0, "status": "completed"},
                "corrupted_arguments_1": {"reward": 0.0, "status": "completed"},
                "corrupted_arguments_2": {"reward": 0.0, "status": "completed"},
                "skipped_tool_1": {"reward": 0.0, "status": "completed"},
                "skipped_tool_2": {"reward": 0.0, "status": "completed"},
                "reordered_tools": {"reward": 0.0, "status": "completed"},
                "noise_selection": {"reward": -1.0, "status": "completed"},
            }},
        }))
        (evidence / "data_governance.json").write_text(json.dumps({
            "version": "1.0",
            "eligible_for_external_model_processing": True,
            "data_origin": {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            },
            "providers": {
                "agent": {
                    "host": "agent.example",
                    "model": "policy-model",
                    "identity_sha256": "a" * 64,
                },
                "user_simulator_and_reward": {
                    "host": "runtime.example",
                    "model": "simulator-model",
                    "identity_sha256": "b" * 64,
                },
            },
            "outbound_surfaces": certifier.OUTBOUND_SURFACES,
            "forbidden_outbound": certifier.FORBIDDEN_OUTBOUND,
            "credential_findings": [],
            "pii_findings": [],
        }))
        sandbox_artifacts_digest = certifier.portable_artifact_digest(evidence)
        holdouts = []
        sandbox_reports = []
        for batch in range(batches):
            jobs = []
            for index in range(count):
                global_index = batch * count + index
                task = root / f"task-{global_index}.json"
                category = (
                    "direct_response" if index % 10 < 2
                    else "simple_agentic" if index % 10 < 5
                    else "multi_step_agentic"
                )
                # A digest-like description avoids intentionally triggering the
                # semantic near-duplicate detector in this passing fixture.
                description = __import__("hashlib").sha256(
                    f"task-{global_index}".encode()
                ).hexdigest().translate(str.maketrans("0123456789", "ghijklmnop"))
                task.write_text(json.dumps(self.fixture_task(category, description)))
                task_score = score_file(task, min_score=8.0).to_dict()
                sample_seed = global_index + 1000
                sample_manifest = root / f"sample-{global_index}.json"
                sample_manifest.write_text(json.dumps({
                    "version": "2.0",
                    "status": "completed",
                    "task_id": f"task-{global_index}",
                    "batch_index": index + 1,
                    "run_seed": batch + 10,
                    "sample_seed": sample_seed,
                    "task_sha256": __import__("hashlib").sha256(
                        task.read_bytes()
                    ).hexdigest(),
                    "training_category": category,
                    "resolved_task_intent": "fixture_intent",
                    "hops": 3,
                    "successful_attempt": 1,
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
                    "attempts": [{
                        "attempt": 1,
                        "seed": sample_seed,
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
                }))
                episodes = [
                {
                    "schema_version": "2.0", "seed": episode, "agent_success": episode < 8,
                    "termination": "completed" if episode < 8 else "step_budget",
                    "initial_reward": 0.0, "final_reward": 1.0 if episode < 8 else 0.0,
                    "issues": [], "usage": [{
                        "response_model": "policy-model",
                        "response_id_sha256": "9" * 64,
                        "finish_reason": "stop",
                        "token_usage": {"total_tokens": 10},
                    }], "initial_state": {}, "final_state": {},
                    "replay": {"events": [{
                        "event": "runtime_llm_call",
                        "payload": {"summary": {
                            "version": "1.0", "responses": 1,
                            "mock_responses": 0,
                            "models": {"simulator-model": 1},
                            "usage": {"total_tokens": 5},
                            "response_id_sha256": ["8" * 64],
                        }},
                    }]},
                    "transitions": [{
                        "step": 0,
                        "agent_input": [{"role": "user", "content": "do it"}],
                        "assistant_output": '{"kind":"respond","content":"done"}',
                        "observation": {},
                        "action": {"kind": "respond", "content": "done"},
                        "result": {
                            "status": 200,
                            "user_query": self.fixture_user_turn(episode)["user_query"],
                        },
                        "trainer_metadata": {
                            "user_simulator": self.fixture_user_turn(episode),
                        },
                        "next_observation": {},
                        "reward": 1.0 if episode < 8 else 0.0,
                        "terminated": episode < 8,
                        "truncated": episode >= 8,
                    }],
                    "trajectory": [
                        {
                            "method": "POST", "path": "/v1/reset", "status": 200,
                            "body": {"episode_id": f"live-{episode}", "seed": episode},
                            "result": {},
                        },
                        {
                            "method": "GET", "path": "/v1/state", "status": 200,
                            "body": None, "result": {"business_state": {}},
                        },
                        {
                            "method": "GET", "path": "/v1/reward", "status": 200,
                            "body": None, "result": {"reward": 0.0},
                        },
                        {
                            "method": "GET", "path": "/v1/tools", "status": 200,
                            "body": None, "result": {"tools": []},
                        },
                        {
                            "method": "GET", "path": "/v1/observation", "status": 200,
                            "body": None, "result": {},
                        },
                        {
                            "method": "POST", "path": "/v1/agent_response", "status": 200,
                            "body": {"content": "done"}, "result": {},
                        },
                        {
                            "method": "POST", "path": "/v1/user_simulator", "status": 200,
                            "body": {"messages": []},
                            "result": self.fixture_user_turn(episode),
                        },
                        {
                            "method": "GET", "path": "/v1/observation", "status": 200,
                            "body": None, "result": {},
                        },
                        *[{
                            "method": "GET", "path": "/v1/reward", "status": 200,
                            "body": None,
                            "result": {"reward": 1.0 if episode < 8 else 0.0},
                        } for _ in range(3)],
                        {
                            "method": "GET", "path": "/v1/replay", "status": 200,
                            "body": None, "result": {"events": [{
                                "event": "runtime_llm_call",
                                "payload": {"summary": {
                                    "version": "1.0", "responses": 1,
                                    "mock_responses": 0,
                                    "models": {"simulator-model": 1},
                                    "usage": {"total_tokens": 5},
                                    "response_id_sha256": ["8" * 64],
                                }},
                            }]},
                        },
                        {
                            "method": "GET", "path": "/v1/state", "status": 200,
                            "body": None, "result": {"business_state": {}},
                        },
                    ],
                }
                for episode in range(10)
                ]
                check_weights = {
                    "delivery_integrity": 1.0,
                    "contract_and_tool_identity": 1.0,
                    "semantic_business_fidelity": 0.5,
                    "business_acceptance": 1.0,
                    "sandbox_pytest": 1.0,
                    "runtime_genericity": 0.5,
                    "outer_conformance": 1.0,
                    "mutation_resistance": 1.0,
                    "training_readiness": 1.0,
                    "declared_training_policy": 2.0,
                }
                sandbox_report = {
                    "score": 10.0,
                    "eligible": True,
                    "passed": True,
                    "threshold": 8.0,
                    "failed_critical_gates": [],
                    "checks": [
                        {
                            "name": name,
                            "weight": weight,
                            "passed": True,
                            "evidence": "fixture evidence",
                            "critical": True,
                        }
                        for name, weight in check_weights.items()
                    ],
                    "root": str(evidence),
                    "mode": "offline_executable",
                    "live_rollout_verified": False,
                    "evidence_fingerprint": evidence_fingerprint(
                        evidence, ROOT, task_path=task
                    ),
                    "network_used": False,
                    "model_used": False,
                    "model": "gpt-5.6-luna",
                    "review_model": "gpt-5.6-luna",
                }
                sandbox_reports.append(sandbox_report)
                result = {
                    "task_path": str(task), "output": str(evidence),
                    "container_image_tag": "fixture",
                    "sample_seed": sample_seed,
                    "sample_manifest": str(sample_manifest),
                    "category": category,
                    "task_score": task_score,
                    "sandbox_score": sandbox_report,
                    "score": 10.0, "passed": True,
                    "live_rollout": {
                        "schema_version": "2.0",
                        "material_visibility_version": "1.0",
                        "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                        "sandbox_artifacts_digest": sandbox_artifacts_digest,
                        "agent_model": "policy-model",
                        "runtime_model": "simulator-model",
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
                        "live_rollout_verified": True,
                        "agent_success_rate": 0.8,
                        "environment_checks_passed": True,
                        "all_episodes_fallback_free": True,
                        "all_episodes_environment_clean": True,
                        "minimum_success_rate": 2 / 3,
                        "success_rate_gate_passed": True,
                        "quality_score": 1.0,
                        "passed": True,
                        "failure_owner": None,
                        "conclusion": "live_success_witness",
                        "episodes": episodes,
                    },
                }
                jobs.append({"id": index + 1, "state": "complete", "result": result})
            holdouts.append({
                "holdout_batch": batch + 1,
                "jobs": jobs,
                "summary": {"fresh_tasks_verified": True},
            })
        (evidence / "score_summary.json").write_text(json.dumps({
            "mode": "offline_executable",
            "threshold": 8.0,
            "total": len(sandbox_reports),
            "passed": len(sandbox_reports),
            "failed": 0,
            "all_passed": True,
            "sandboxes": sandbox_reports,
        }))
        representative = holdouts[0]["jobs"][0]["result"]["live_rollout"]
        (evidence / "trajectory_privacy.json").write_text(json.dumps(
            certifier.audit_rollout_privacy(representative)
        ))
        return {
            "config": {
                "certification_profile": "production",
                "threshold": 8.0,
                "validation": "live",
                "sandbox_runtime": "docker",
                "holdout_count": 300,
                "holdout_batches": 3,
                "holdout_rollout_episodes": 10,
                "source_digest": "d" * 64,
                "input_digests": [],
                "execution_provenance": certifier.verify_execution_provenance(
                    ROOT, {}
                )["current"],
                "generation_provider": {
                    "host": "generator.example",
                    "model": "generator-model",
                    "identity_sha256": "c" * 64,
                },
                "runtime_provider": {
                    "host": "runtime.example",
                    "model": "simulator-model",
                    "identity_sha256": "b" * 64,
                },
                "rollout_provider": {
                    "host": "agent.example",
                    "model": "policy-model",
                    "identity_sha256": "a" * 64,
                },
                "bundle_attestation_key_identity_sha256": "e" * 64,
                "generation_model": "generator-model",
                "rollout_model": "policy-model",
                "runtime_model": "simulator-model",
                "generation_allowed_response_models": ["generator-model"],
                "rollout_allowed_response_models": ["policy-model"],
                "runtime_allowed_response_models": ["simulator-model"],
            },
            "holdout": holdouts[0], "holdouts": holdouts,
        }

    def test_wilson_bound_accounts_for_sample_size(self):
        self.assertLess(certifier.wilson_lower(9, 10), .9)
        self.assertGreater(certifier.wilson_lower(900, 1000), .87)
        self.assertGreater(certifier.wilson_upper(0, 100), .005)
        self.assertLess(certifier.wilson_upper(0, 1000), .005)
        self.assertEqual(certifier.wilson_upper(0, 0), 1.0)

    def test_certification_json_publication_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            value = {"certified": False, "原因": "证据不足"}
            certifier.write_json_atomic(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tmp")), []
            )

    def test_production_profile_can_pass_complete_pretraining_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the authoritative on-disk representation: tuples in a
            # freshly computed score report become lists after JSON storage.
            history = json.loads(json.dumps(self.make_history(Path(directory), count=300)))
            report = self.certify(history)
            self.assertTrue(report["certified"], report["failed_gates"])
            self.assertEqual(report["scope"], "pre_training_material_readiness")
            self.assertIn("rl_training_execution", report["does_not_certify"])
            self.assertIn(
                "downstream_training_system_compatibility",
                report["does_not_certify"],
            )
            self.assertIn("rl_training_convergence", report["does_not_certify"])
            self.assertEqual(report["measurements"]["episodes"], 9000)
            self.assertEqual(report["measurements"]["post_score_survival_rate"], 1.0)
            self.assertTrue(report["gates"]["rollout_provenance"])
            self.assertTrue(report["gates"]["container_rollout_execution"])
            self.assertTrue(report["gates"]["container_reward_calibration"])
            self.assertTrue(report["gates"]["category_mix"])
            self.assertTrue(report["gates"]["user_simulator_outcome_coverage"])
            self.assertTrue(report["gates"]["data_governance"])
            self.assertTrue(report["gates"]["container_reproducibility"])
            self.assertLessEqual(
                report["measurements"]["reward_false_positive_ci95_upper"],
                certifier.canonical_certification_policy()["max_reward_false_positive_rate"],
            )
            self.assertLessEqual(
                report["measurements"]["reward_false_negative_ci95_upper"],
                certifier.canonical_certification_policy()["max_reward_false_negative_rate"],
            )
            self.assertRegex(
                report["measurements"]["container_reproducibility"]
                    ["observed_images_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertTrue(report["gates"]["trajectory_privacy"])
            self.assertTrue(report["gates"]["execution_environment"])
            self.assertTrue(report["gates"]["production_experiment_profile"])
            self.assertTrue(report["gates"]["production_preflight"])
            self.assertEqual(len(report["materials_manifest"]["items"]), 900)
            self.assertEqual(
                report["materials_manifest"]["version"],
                MATERIAL_MANIFEST_VERSION,
            )
            self.assertRegex(
                report["measurements"]["sandbox_executable_revalidation"]
                    ["evidence_set_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                report["materials_manifest"]["evaluator_source_digest"],
                "d" * 64,
            )
            self.assertEqual(len(report["materials_manifest"]["dataset_sha256"]), 64)

    def test_docker_tag_drift_cannot_pass_container_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            report = self.certify(history, image_inspector=lambda tag: {
                "image_id": "sha256:" + "e" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
                "runtime_user": "sandbox",
            })
            self.assertFalse(report["gates"]["container_reproducibility"])

    def test_small_fixture_preserves_non_cohort_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.certify(self.make_history(Path(directory)))
            self.assertEqual(set(report["failed_gates"]), {
                "materialized_sample_size",
                "independent_holdout_batches",
                "fresh_holdout",
                "rollout_coverage",
                "reward_false_positive_rate",
                "reward_false_negative_rate",
            })
            self.assertEqual(report["measurements"]["reward_false_positive_rate"], 0.0)
            self.assertGreater(
                report["measurements"]["reward_false_positive_ci95_upper"],
                certifier.canonical_certification_policy()["max_reward_false_positive_rate"],
            )

    def test_relaxed_policy_cannot_certify_training_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            policy = certifier.default_policy()
            policy["min_tasks"] = 1
            report = certifier.certify(
                history,
                policy,
                sandbox_revalidation=self.sandbox_revalidation(history),
                production_preflight=self.production_preflight(history["config"]),
            )
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["certification_policy"])
            self.assertIn("certification_policy", report["failed_gates"])
            self.assertEqual(report["policy"]["min_tasks"], 30)
            self.assertFalse(
                report["measurements"]["certification_policy"]["verified"]
            )
            forged_manifest = json.loads(json.dumps(report["materials_manifest"]))
            forged_manifest["certification_policy"]["min_tasks"] = 1
            forged_manifest["certification_policy_sha256"] = (
                certifier.digest_json(forged_manifest["certification_policy"])
            )
            forged_manifest["dataset_sha256"] = certifier.digest_json({
                key: value for key, value in forged_manifest.items()
                if key != "dataset_sha256"
            })
            verification = verifier.verify(forged_manifest, ROOT)
            self.assertFalse(verification["verified"])
            self.assertIn(
                "certification_policy", verification["failed_gates"]
            )

    def test_material_manifest_rejects_unknown_schema_version(self):
        manifest = {
            "version": "99.0",
            "kind": "agentic_rl_pretraining_materials",
            "items": [],
        }
        manifest["dataset_sha256"] = certifier.digest_json(manifest)
        report = verifier.verify(manifest, ROOT)
        self.assertFalse(report["verified"])
        self.assertIn("manifest_schema", report["failed_gates"])

    def test_pilot_profile_cannot_claim_production_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"]["certification_profile"] = "pilot"
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertIn("production_experiment_profile", report["failed_gates"])

    def test_missing_production_preflight_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            report = certifier.certify(
                history,
                certifier.default_policy(),
                sandbox_revalidation=self.sandbox_revalidation(history),
            )
            self.assertFalse(report["certified"])
            self.assertIn("production_preflight", report["failed_gates"])

    def test_preflight_provider_drift_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"]["runtime_provider"] = {
                "host": "other.example", "model": "simulator-model",
                "identity_sha256": "d" * 64,
            }
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertIn("production_preflight", report["failed_gates"])
            self.assertFalse(report["gates"]["provider_identity_consistency"])

    def test_rollout_agent_provider_drift_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"]["rollout_provider"] = {
                "host": "other.example", "model": "policy-model",
                "identity_sha256": "d" * 64,
            }
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertIn("production_preflight", report["failed_gates"])
            self.assertFalse(report["gates"]["provider_identity_consistency"])

    def test_missing_actual_model_response_provenance_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            rollout = history["holdouts"][0]["jobs"][0]["result"][
                "live_rollout"
            ]
            rollout["episodes"][0]["usage"][0].pop("response_model")
            report = self.certify(history)
            self.assertFalse(report["gates"]["model_response_provenance"])
            self.assertIn("model_response_provenance", report["failed_gates"])

    def test_unapproved_actual_response_model_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            rollout = history["holdouts"][0]["jobs"][0]["result"][
                "live_rollout"
            ]
            rollout["episodes"][0]["usage"][0][
                "response_model"
            ] = "unapproved-policy-model"
            report = self.certify(history)
            self.assertTrue(report["gates"]["model_response_provenance"])
            self.assertFalse(report["gates"]["model_response_authorization"])
            self.assertIn("model_response_authorization", report["failed_gates"])


    def test_preflight_signing_key_drift_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"][
                "bundle_attestation_key_identity_sha256"
            ] = "f" * 64
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertIn("production_preflight", report["failed_gates"])

    def test_preflight_cannot_be_reused_after_frozen_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            preflight = self.production_preflight(history["config"])
            history["config"]["score_threshold"] = 9.0
            report = certifier.certify(
                history,
                certifier.default_policy(),
                sandbox_revalidation=self.sandbox_revalidation(history),
                production_preflight=preflight,
            )
            self.assertFalse(report["gates"]["production_preflight"])
            self.assertIn("production_preflight", report["failed_gates"])

    def test_execution_environment_drift_breaks_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"]["execution_provenance"]["python"]["version"] = "0.0"
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["execution_environment"])
            self.assertIn("execution_environment", report["failed_gates"])

    def test_task_generation_provider_drift_breaks_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            sample_manifest = Path(result["sample_manifest"])
            value = json.loads(sample_manifest.read_text())
            value["generator_provider"]["identity_sha256"] = "f" * 64
            sample_manifest.write_text(json.dumps(value))
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["generation_provenance"])
            self.assertIn("generation_provenance", report["failed_gates"])

    def test_thirty_task_holdout_lacks_reward_confidence_for_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory), count=30)
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertTrue(report["gates"]["materialized_sample_size"])
            self.assertTrue(report["gates"]["rollout_coverage"])
            self.assertIn("reward_false_positive_rate", report["failed_gates"])
            self.assertIn("reward_false_negative_rate", report["failed_gates"])
            self.assertEqual(report["measurements"]["post_score_survival_rate"], 1.0)

    def test_near_duplicate_task_family_cannot_cross_holdout_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            first = history["holdouts"][0]["jobs"][0]["result"]
            second = history["holdouts"][1]["jobs"][0]["result"]
            first_task = json.loads(Path(first["task_path"]).read_text())
            second_path = Path(second["task_path"])
            second_task = json.loads(second_path.read_text())
            second_task["task"] = first_task["task"] + " 2026!"
            second_path.write_text(json.dumps(second_task))
            changed_digest = __import__("hashlib").sha256(
                second_path.read_bytes()
            ).hexdigest()
            second["live_rollout"]["task_sha256"] = changed_digest
            sample_path = Path(second["sample_manifest"])
            sample = json.loads(sample_path.read_text())
            sample["task_sha256"] = changed_digest
            sample_path.write_text(json.dumps(sample))
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["holdout_partition_isolation"])
            self.assertEqual(
                report["measurements"]["partition_isolation"]
                ["cross_partition_family_count"],
                1,
            )

    def test_any_fallback_breaks_environment_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["holdout"]["jobs"][0]["result"]["live_rollout"]["episodes"][0][
                "issues"
            ] = ["runtime_llm_fallback"]
            report = self.certify(history)
            self.assertFalse(report["gates"]["environment_integrity"])
            self.assertEqual(report["measurements"]["llm_fallbacks"], 1)

    def test_same_provider_agent_and_evaluator_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            rollout = history["holdouts"][0]["jobs"][0]["result"][
                "live_rollout"
            ]
            rollout["runtime_provider_sha256"] = rollout[
                "agent_provider_sha256"
            ]
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["evaluator_independence"])
            self.assertEqual(
                report["measurements"]["evaluator_independence"]
                    ["same_provider_items"],
                1,
            )

    def test_same_endpoint_different_models_cannot_certify(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["config"]["runtime_provider"]["host"] = (
                history["config"]["rollout_provider"]["host"]
            )
            report = self.certify(history)
            self.assertFalse(report["gates"]["evaluator_independence"])
            self.assertEqual(
                report["measurements"]["evaluator_independence"],
                {
                    "same_provider_items": 30,
                    "total_items": 30,
                    "same_provider_rate": 1.0,
                },
            )

    def test_invalid_user_simulator_protocol_breaks_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            jobs = [job for batch in history["holdouts"] for job in batch["jobs"]]
            for index, job in enumerate(jobs[:46]):
                turns = job["result"]["live_rollout"]["episodes"][0]["trajectory"]
                user_turn = next(step for step in turns if step["path"] == "/v1/user_simulator")
                if index % 2:
                    user_turn["result"].pop("outcome_category")
                else:
                    user_turn["result"]["fsm_state_after"] = "undeclared-state"
            report = self.certify(history)
            self.assertFalse(report["gates"]["user_simulator_protocol"])
            self.assertFalse(report["gates"]["trajectory_schema"])

    def test_rollout_must_be_bound_to_the_exact_task_and_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["task_sha256"] = "0" * 64
            report = self.certify(history)
            self.assertFalse(report["gates"]["rollout_provenance"])

    def test_in_process_rollout_cannot_certify_container_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            rollout = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]
            rollout["runtime_execution"] = {
                "version": "1.0", "mode": "in_process", "container_image_id": None,
            }
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["rollout_provenance"])
            self.assertFalse(report["gates"]["container_rollout_execution"])

    def test_mock_reward_counterfactuals_cannot_certify_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            live_report = root / "evidence/agentic_training_value_live.json"
            value = json.loads(live_report.read_text())
            value["validation_mode"] = "offline_mock"
            live_report.write_text(json.dumps(value))
            report = self.certify(history)
            self.assertFalse(report["gates"]["tool_and_reward_integrity"])

    def test_live_reward_report_cannot_hide_answer_bearing_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            live_report = root / "evidence/agentic_training_value_live.json"
            value = json.loads(live_report.read_text())
            value["evidence"]["business_data_causality"]["answer_value_mentioned"] = True
            live_report.write_text(json.dumps(value))
            report = self.certify(history)
            self.assertFalse(report["gates"]["tool_and_reward_integrity"])

    def test_in_process_reward_calibration_cannot_certify_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            live_report = root / "evidence/agentic_training_value_live.json"
            value = json.loads(live_report.read_text())
            value["runtime_execution"] = {
                "version": "1.0",
                "mode": "in_process",
                "container_image_id": None,
            }
            live_report.write_text(json.dumps(value))
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["container_reward_calibration"])

    def test_final_pass_cannot_bypass_task_and_sandbox_qualification_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["task_score"]["eligible"] = False
            # A stale or tampered outer result still claims final success.
            self.assertTrue(result["passed"])
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["qualification_lineage"])
            self.assertEqual(
                report["measurements"]["qualification_lineage"],
                {"verified": False, "violations": 1},
            )
            self.assertEqual(report["measurements"]["training_ready"], 29)
            self.assertEqual(report["measurements"]["episodes"], 290)

    def test_rollout_summary_is_recomputed_from_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            rollout = history["holdouts"][0]["jobs"][0]["result"][
                "live_rollout"
            ]
            rollout["passed"] = False
            rollout["agent_success_rate"] = 1.0
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["rollout_outcome_integrity"])
            self.assertEqual(
                report["measurements"]["rollout_outcome_integrity"],
                {"verified": 29, "expected": 30, "all_verified": False},
            )

    def test_rollout_success_flag_must_match_terminal_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            rollout = result["live_rollout"]
            task = json.loads(Path(result["task_path"]).read_text())
            policy = certifier.canonical_certification_policy()
            self.assertTrue(certifier.valid_rollout_outcome(
                rollout, policy, task=task,
            ))
            rollout["episodes"][0]["agent_success"] = False
            rollout["agent_success_rate"] = 0.7
            self.assertFalse(certifier.valid_rollout_outcome(
                rollout, policy, task=task,
            ))

    def test_rollout_state_goal_is_recomputed_from_business_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            rollout = result["live_rollout"]
            task = json.loads(Path(result["task_path"]).read_text())
            task["task_spec"]["goal_contract"] = {
                "row_predicates": [{
                    "table": "inventory", "where": {"id": 1},
                    "values": {"status": "done"}, "count": 1,
                }],
                "requires_state_change": True,
            }
            for episode in rollout["episodes"]:
                episode["initial_state"] = {"inventory": [{"id": 1, "status": "pending"}]}
                succeeded = episode["agent_success"]
                episode["final_state"] = {"inventory": [{
                    "id": 1, "status": "done" if succeeded else "pending",
                }]}
                episode["state_goal_satisfied"] = succeeded
                next(step for step in episode["trajectory"] if step["path"] == "/v1/state")["result"]["business_state"] = episode["initial_state"]
                episode["trajectory"][-1]["result"]["business_state"] = episode["final_state"]
            policy = certifier.canonical_certification_policy()
            self.assertTrue(certifier.valid_rollout_outcome(
                rollout, policy, task=task,
            ))
            rollout["episodes"][0]["final_state"]["inventory"][0]["status"] = "pending"
            self.assertFalse(certifier.valid_rollout_outcome(
                rollout, policy, task=task,
            ))

    def test_rollout_provider_identity_must_match_data_governance(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["agent_provider_sha256"] = "f" * 64
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["provider_identity_consistency"])
            self.assertEqual(
                report["measurements"]["provider_identity_consistency"],
                {"verified": 29, "expected": 30, "all_verified": False},
            )

    def test_reward_evaluator_identity_must_match_data_governance(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory), count=1, batches=1)
            result = history["holdouts"][0]["jobs"][0]["result"]
            self.assertTrue(certifier.valid_provider_binding(result))
            calibration_path = (
                Path(result["output"]) / "agentic_training_value_live.json"
            )
            calibration = json.loads(calibration_path.read_text())
            calibration["evaluator_provider"]["identity_sha256"] = "f" * 64
            calibration_path.write_text(json.dumps(calibration))
            self.assertFalse(certifier.valid_provider_binding(result))

    def test_rollout_uses_governance_provider_identity_algorithm(self):
        class Provider:
            def __init__(self, base_url, model):
                self.base_url = base_url
                self.model = model

        agent = Provider("https://agent.example/v1/", "policy-model")
        runtime = Provider("https://runtime.example/v1", "simulator-model")
        attestation = rollout_runner.rollout_provider_attestation(agent, runtime)
        from env_factory.evidence.data_governance import provider_identity

        self.assertEqual(
            attestation["agent_provider_sha256"],
            provider_identity(agent.base_url, agent.model)["identity_sha256"],
        )
        self.assertEqual(
            attestation["runtime_provider_sha256"],
            provider_identity(runtime.base_url, runtime.model)["identity_sha256"],
        )

    def test_rollout_provider_digest_must_be_canonical_hex(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["agent_provider_sha256"] = "z" * 64
            report = self.certify(history)
            self.assertFalse(report["gates"]["rollout_provenance"])

    def test_evidence_fingerprint_changes_after_same_directory_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.json").write_text("{}")
            (root / "app.py").write_text("VALUE = 1\n")
            before = evidence_fingerprint(root, ROOT)
            (root / "app.py").write_text("VALUE = 2\n")
            self.assertNotEqual(before, evidence_fingerprint(root, ROOT))

    def test_evidence_fingerprint_covers_nested_business_data_and_scorer_code(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            project = workspace / "project"
            scorer = project / "scripts/sandbox/score_sandbox.py"
            scorer.parent.mkdir(parents=True)
            scorer.write_text("SCORE = 1\n")
            sandbox = workspace / "sandbox"
            business = sandbox / "data/business_data/records.json"
            business.parent.mkdir(parents=True)
            business.write_text('{"amount": 1}')
            (sandbox / "task.json").write_text("{}")
            before = evidence_fingerprint(sandbox, project)
            business.write_text('{"amount": 2}')
            after_business = evidence_fingerprint(sandbox, project)
            self.assertNotEqual(before, after_business)
            scorer.write_text("SCORE = 2\n")
            self.assertNotEqual(after_business, evidence_fingerprint(sandbox, project))

    def test_score_report_rejects_symlinked_build_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sandbox"
            root.mkdir()
            target = Path(directory) / "outside.json"
            target.write_text("{}")
            (root / "linked.json").symlink_to(target)
            report = {
                "checks": [{"name": name, "weight": weight, "passed": True,
                            "evidence": "ok", "critical": critical}
                           for name, weight, critical in SCORE_RUBRIC],
                "mode": "offline_executable", "network_used": False, "model_used": False,
                "live_rollout_verified": False, "threshold": 8, "score": 10.0,
                "eligible": True, "passed": True, "failed_critical_gates": [],
                "evidence_fingerprint": "claimed", "model": "builder", "review_model": "reviewer",
            }
            self.assertFalse(valid_score_report(report, root=root, project=ROOT, threshold=8))

    def test_sandbox_score_must_match_frozen_structural_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["sandbox_score"] = dict(result["sandbox_score"])
            result["sandbox_score"]["score"] = 9.5
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["sandbox_score_provenance"])
            self.assertEqual(
                report["measurements"]["sandbox_score_provenance"],
                {"verified": 29, "failures": 1},
            )

    def test_production_certification_requires_executable_sandbox_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["sandbox_executable_revalidation"])
            self.assertEqual(
                report["measurements"]["sandbox_executable_revalidation"],
                {
                    "verified": 0, "expected": 30, "failures": 30,
                    "evidence_set_sha256": None,
                },
            )

    def test_revalidated_score_must_match_the_frozen_score_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            revalidation = self.sandbox_revalidation(history)
            result = history["holdouts"][0]["jobs"][0]["result"]
            key = certifier.sandbox_revalidation_key(
                Path(result["output"]), Path(result["task_path"])
            )
            rerun = dict(revalidation[key]["report"])
            rerun["score"] = 9.5
            revalidation[key] = {**revalidation[key], "report": rerun}
            report = certifier.certify(
                history,
                certifier.default_policy(),
                sandbox_revalidation=revalidation,
            )
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["sandbox_executable_revalidation"])
            self.assertEqual(
                {
                    key: value for key, value in
                    report["measurements"]["sandbox_executable_revalidation"].items()
                    if key != "evidence_set_sha256"
                },
                {"verified": 29, "expected": 30, "failures": 1},
            )

    def test_sandbox_revalidation_invokes_offline_executable_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sandbox"
            root.mkdir()
            task = root / "task.json"
            task.write_text("{}")
            claimed = {"passed": True, "score": 8.0}
            history = {"holdouts": [{"jobs": [{"result": {
                "output": str(root),
                "task_path": str(task),
                "sandbox_score": claimed,
            }}]}]}

            def fake_run(command, **kwargs):
                self.assertIn("score_sandbox_offline.py", command[1])
                self.assertIn("--no-individual", command)
                self.assertEqual(kwargs["env"]["SANDBOX_EVALUATOR_MOCK"], "1")
                self.assertNotIn("SANDBOX_LLM_API_KEY", kwargs["env"])
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"sandboxes": [claimed]}))
                return certifier.subprocess.CompletedProcess(command, 0)

            with patch.object(certifier.subprocess, "run", side_effect=fake_run):
                result = certifier.run_sandbox_revalidation(
                    history,
                    project=ROOT,
                    policy=certifier.default_policy(),
                    max_workers=1,
                    timeout=10,
                )
            key = certifier.sandbox_revalidation_key(root, task)
            self.assertEqual(result[key]["report"], claimed)
            self.assertTrue(result[key]["verified"])

    def test_sandbox_score_rubric_weights_and_critical_flags_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "score": 10.0, "eligible": True, "passed": True,
                "threshold": 8.0, "failed_critical_gates": [],
                "checks": [
                    {
                        "name": name, "weight": weight, "passed": True,
                        "evidence": "verified", "critical": critical,
                    }
                    for name, weight, critical in SCORE_RUBRIC
                ],
                "mode": "offline_executable", "network_used": False,
                "model_used": False, "live_rollout_verified": False,
                "model": "gpt-5.6-luna", "review_model": "gpt-5.6-luna",
                "evidence_fingerprint": evidence_fingerprint(root, ROOT),
            }
            self.assertTrue(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))
            report["verification_status"] = "live_verified"
            self.assertFalse(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))
            report.pop("verification_status")
            report["checks"][0]["weight"] = 2.0
            report["score"] = 11.0
            self.assertFalse(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))
            report["checks"][0]["weight"] = 1.0
            report["score"] = 10.0
            report["checks"][0]["critical"] = False
            self.assertFalse(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))

    def test_sandbox_continuous_score_is_recomputed_from_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review_report.json").write_text(json.dumps({"score": 0.9}))
            (root / "agentic_training_value.json").write_text(json.dumps({
                "evidence": {"counterfactuals": {
                    "goal_success": {"reward": 1.0, "status": "completed"},
                    "goal_failure": {"reward": 0.25, "status": "completed"},
                }},
            }))
            factors = sandbox_quality_factors(root)
            self.assertEqual(factors, {
                "semantic_business_fidelity": 0.9,
                "declared_training_policy": 0.75,
            })
            report = {
                "score": 9.45, "raw_score": 10.0,
                "score_kind": "gated_weighted_10_point", "quality_factors": factors,
                "eligible": True, "passed": True, "threshold": 8.0,
                "failed_critical_gates": [],
                "checks": [{
                    "name": name, "weight": weight, "passed": True,
                    "evidence": "verified", "critical": critical,
                } for name, weight, critical in SCORE_RUBRIC],
                "mode": "offline_executable", "verification_status": "requires_live",
                "network_used": False, "model_used": False,
                "live_rollout_verified": False,
                "model": "builder", "review_model": "reviewer",
                "evidence_fingerprint": evidence_fingerprint(root, ROOT),
            }
            self.assertTrue(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))
            report["score"] = 10.0
            self.assertFalse(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))

    def test_sandbox_score_cannot_hide_failed_critical_gate_behind_high_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = [
                {"name": name, "weight": weight, "passed": name != "semantic_business_fidelity",
                 "evidence": "checked", "critical": critical}
                for name, weight, critical in SCORE_RUBRIC
            ]
            report = {
                "score": 0.0, "eligible": False, "passed": False,
                "threshold": 8.0,
                "failed_critical_gates": ["semantic_business_fidelity"],
                "checks": checks,
                "mode": "offline_executable", "network_used": False,
                "model_used": False, "live_rollout_verified": False,
                "model": "builder", "review_model": "reviewer",
                "evidence_fingerprint": evidence_fingerprint(root, ROOT),
            }
            self.assertTrue(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))
            report["score"] = 9.5
            self.assertFalse(valid_score_report(
                report, root=root, project=ROOT, threshold=8.0,
            ))

    def test_task_score_and_category_are_recomputed_from_frozen_task(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["task_score"] = dict(result["task_score"])
            result["task_score"]["score"] = 9.5
            result["category"] = "multi_step_agentic"
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["task_score_provenance"])
            self.assertEqual(
                report["measurements"]["task_score_provenance"],
                {"verified": 29, "failures": 1},
            )

    def test_final_score_is_recomputed_from_sandbox_and_rollout_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["score"] = 9.5
            report = self.certify(history)
            self.assertFalse(report["certified"])
            self.assertFalse(report["gates"]["final_result_provenance"])
            self.assertEqual(
                report["measurements"]["final_result_provenance"],
                {"verified": 29, "failures": 1},
            )

    def test_reward_calibration_requires_task_derived_counterfactual_coverage(self):
        task = self.fixture_task("multi_step_agentic", "验证多步工具奖励覆盖矩阵。")
        task["acceptance_contract"]["executable_scenarios"][0]["steps"].append({
            "operation": "agent_response", "content": "done",
        })
        report = {
            "curriculum_training_ready": True,
            "agentic_training_ready": True,
            "hard_gates_passed": True,
            "validation_mode": "live_evaluator",
            "failed_gates": [],
            "failures": [],
            "evidence": {"business_data_access": {"tables_read": ["inventory"], "tool_reads": [
                {"tool_name": "read_inventory", "tables_read": ["inventory"]},
                {"tool_name": "rank_inventory", "tables_read": ["inventory"]},
            ]},
                         "business_data_causality": {"proved": True, "table": "inventory", "tool": "lookup"},
                         "reference_data_immutability": {"unchanged": True},
                         "counterfactuals": {
                name: {
                    "reward": 1.0 if name == "goal_success" else (
                        -1.0 if name == "noise_selection" else 0.0
                    ),
                    "status": "completed",
                }
                for name in (
                    "goal_success", "goal_failure", "no_tools",
                    "corrupted_arguments_1", "corrupted_arguments_2",
                    "skipped_tool_1", "skipped_tool_2", "reordered_tools",
                    "noise_selection",
                )
            }},
        }
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        cases = report["evidence"]["counterfactuals"]
        for name in ("goal_failure", "no_tools", "noise_selection"):
            original_case = dict(cases[name])
            cases[name] = {
                "reward": None, "status": "rejected", "http_status": 422,
                "error_type": "ScenarioRejected",
            }
            self.assertFalse(certifier.valid_reward_calibration(report, task), name)
            cases[name] = original_case
        for invalid in (float("nan"), float("inf"), -float("inf"), 1.1, -1.1, 10**1000, True):
            original = cases["goal_success"]["reward"]
            cases["goal_success"]["reward"] = invalid
            self.assertFalse(certifier.valid_reward_calibration(report, task))
            cases["goal_success"]["reward"] = original
            original = cases["goal_failure"]["reward"]
            cases["goal_failure"]["reward"] = invalid
            self.assertFalse(certifier.valid_reward_calibration(report, task))
            cases["goal_failure"]["reward"] = original

        outcome = next(item for item in task["metrics"] if item["category"] == "outcome")
        outcome.update({"weight": 0.8, "evaluation_inputs": ["final_agent_response"]})
        report["evidence"]["counterfactuals"]["wrong_final_answer"] = {
            "reward": 0.0, "status": "completed",
        }
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        causality = report["evidence"]["business_data_causality"]
        causality["answer_value_mentioned"] = True
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_reward_sensitivity"] = {
            "proved": True, "tool_results_changed": True,
            "baseline_reward": 1.0, "baseline_rewards": [1.0, 1.0],
            "baseline_tool_results_stable": True, "changed_reward": 0.0,
            "changed_rewards": [0.0, 0.0],
        }
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        sensitivity = report["evidence"]["business_data_reward_sensitivity"]
        sensitivity["baseline_rewards"] = [1.0, 0.0]
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        sensitivity["baseline_rewards"] = [1.0, 1.0]
        sensitivity["changed_rewards"] = [0.0, 1.0]
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        sensitivity["changed_rewards"] = [0.0]
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        sensitivity["changed_rewards"] = [0.0, 0.0]
        causality.pop("answer_value_mentioned")
        report["evidence"].pop("business_data_reward_sensitivity")
        outcome.pop("weight")
        outcome.pop("evaluation_inputs")
        report["evidence"]["counterfactuals"].pop("wrong_final_answer")
        first_tool = task["tools"][0]["function"]
        first_tool["parameters"] = {"required": ["category", "date"]}
        first_call = task["acceptance_contract"]["executable_scenarios"][0]["steps"][0]
        first_call["arguments"]["date"] = "2026-01-01"
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["counterfactuals"]["corrupted_arguments_1__2"] = {
            "reward": 0.0, "status": "completed",
        }
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        first_call["arguments"].pop("date")
        first_tool.pop("parameters")
        tool_reads = report["evidence"]["business_data_access"]["tool_reads"]
        report["evidence"]["business_data_access"]["tool_reads"] = tool_reads[:1]
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_access"]["tool_reads"] = tool_reads
        access = report["evidence"].pop("business_data_access")
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_access"] = access
        causality = report["evidence"].pop("business_data_causality")
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_causality"] = causality
        immutability = report["evidence"].pop("reference_data_immutability")
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["reference_data_immutability"] = immutability
        task["environment_plan"] = {"mode": "reference_data"}
        task["metrics"] = [{
            "id": "answer", "category": "outcome", "type": "model-based",
            "weight": 0.7, "evaluation_inputs": ["final_agent_response"],
        }]
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        counterfactuals = report["evidence"]["counterfactuals"]
        counterfactuals["wrong_final_answer"] = {"reward": 0.7, "status": "completed"}
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        counterfactuals["wrong_final_answer"]["reward"] = 0.3
        report["evidence"]["business_data_reward_sensitivity"] = {
            "proved": True, "baseline_reward": 1.0,
            "baseline_rewards": [1.0, 1.0], "baseline_tool_results_stable": True,
            "changed_reward": 0.2, "changed_rewards": [0.2, 0.2],
            "tool_results_changed": True,
        }
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        task["task_intent"] = "calculate"
        task["acceptance_contract"]["executable_scenarios"][0]["steps"][-1]["content"] = "合计 42 元"
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        counterfactuals["wrong_numeric_answer"] = {"reward": 0.7, "status": "completed"}
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        counterfactuals["wrong_numeric_answer"]["reward"] = 0.3
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_reward_sensitivity"] = {
            "proved": True, "baseline_reward": 1.0,
            "baseline_rewards": [1.0, 1.0], "baseline_tool_results_stable": True,
            "changed_reward": 0.2, "changed_rewards": [0.2, 0.2],
            "tool_results_changed": True,
        }
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_reward_sensitivity"]["changed_reward"] = 1.0
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        report["evidence"]["business_data_reward_sensitivity"]["changed_reward"] = 0.2
        failure = report["evidence"]["counterfactuals"]["goal_failure"]
        failure.update(reward=None, status="rejected")
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        failure.update(http_status=422, error_type="ScenarioRejected")
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        failure.update(reward=0.0, status="completed")
        failure.pop("http_status")
        failure.pop("error_type")
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        rejected_argument = report["evidence"]["counterfactuals"]["corrupted_arguments_1"]
        rejected_argument.update(reward=None, status="rejected", http_status=422,
                                 error_type="ScenarioRejected")
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        rejected_argument["message"] = "untrusted provider response"
        self.assertFalse(certifier.valid_reward_calibration(report, task))
        rejected_argument.clear()
        rejected_argument.update(reward=0.0, status="completed")
        self.assertTrue(certifier.valid_reward_calibration(report, task))
        del report["evidence"]["counterfactuals"]["skipped_tool_2"]
        self.assertFalse(certifier.valid_reward_calibration(report, task))

    def test_live_causality_must_match_offline_mutation_witness(self):
        task = {"environment_plan": {"mode": "reference_data"}}
        witness = {"proved": True, "table": "inventory", "tool": "lookup",
                   "field": "quantity", "answer_value_mentioned": True}
        offline = {"evidence": {"business_data_causality": dict(witness)}}
        live = {"evidence": {"business_data_causality": dict(witness)}}
        self.assertTrue(certifier.consistent_causality_witness(offline, live, task))
        live["evidence"]["business_data_causality"]["answer_value_mentioned"] = False
        self.assertFalse(certifier.consistent_causality_witness(offline, live, task))
        offline["evidence"]["business_data_causality"].pop("answer_value_mentioned")
        self.assertFalse(certifier.consistent_causality_witness(offline, live, task))

    def test_wrong_answers_count_toward_reward_false_positive_rate(self):
        counts = certifier._counterfactual_counts([{"evidence": {"counterfactuals": {
            "goal_success": {"reward": 1.0, "status": "completed"},
            "wrong_final_answer": {"reward": 0.7, "status": "completed"},
            "wrong_numeric_answer": {"reward": 0.3, "status": "completed"},
        }}}])
        self.assertEqual(counts["negative_total"], 2)
        self.assertEqual(counts["negative_errors"], 1)
        invalid = certifier._counterfactual_counts([{"evidence": {"counterfactuals": {
            "goal_success": {"reward": float("nan")},
            "wrong_final_answer": {"reward": float("nan"), "status": "completed"},
        }}}])
        self.assertEqual(invalid["positive_errors"], 1)
        self.assertEqual(invalid["negative_errors"], 1)
        rejected = certifier._counterfactual_counts([{"evidence": {"counterfactuals": {
            "goal_failure": {"reward": None, "status": "rejected", "http_status": 422},
            "corrupted_arguments_1": {"reward": None, "status": "rejected", "http_status": 422},
        }}}])
        self.assertEqual(rejected["negative_errors"], 1)

    def test_reward_confidence_uses_sandbox_not_correlated_case_count(self):
        cases = {"goal_success": {"reward": 1.0, "status": "completed"}}
        cases.update({
            f"corrupted_arguments_{index}": {"reward": 0.0, "status": "completed"}
            for index in range(100)
        })
        reports = [{"evidence": {"counterfactuals": cases}} for _ in range(100)]
        per_case = certifier._counterfactual_counts(reports)
        per_sandbox = certifier._counterfactual_sandbox_counts(reports)
        self.assertEqual(per_case["negative_total"], 10_000)
        self.assertEqual(per_sandbox, {
            "total": 100, "positive_errors": 0, "negative_errors": 0,
        })
        self.assertLess(certifier.wilson_upper(0, per_case["negative_total"]), .005)
        self.assertGreater(certifier.wilson_upper(0, per_sandbox["total"]), .005)
        reports[0]["evidence"]["counterfactuals"] = {
            "goal_success": {"reward": 1.0, "status": "completed"},
        }
        self.assertEqual(
            certifier._counterfactual_sandbox_counts(reports)["negative_errors"], 1,
        )

    def test_runtime_readiness_requires_complete_executable_evidence(self):
        report = {
            "training_ready": True,
            "validation_mode": "offline_mock",
            "live_rollout_verified": False,
            "hard_gates_passed": True,
            "failed_gates": [],
            "failures": [],
            "evidence": {
                "determinism": True,
                "state_causality": {
                    "archetype": "single_read", "expected_delta": [], "invalid": [],
                },
                "rewards": {"success": 1.0, "failure": 0.0},
                "observation_scan": {"status": 200, "forbidden_paths": []},
                "runtime_state": {
                    "reset_reproducible": True,
                    "episode_isolation": True,
                    "replay_consistent": True,
                    "episode_a_event_count": 2,
                    "episode_b_event_count": 0,
                },
            },
        }
        self.assertTrue(certifier.valid_training_readiness(report))
        del report["evidence"]["runtime_state"]["episode_b_event_count"]
        self.assertFalse(certifier.valid_training_readiness(report))

    def test_credential_finding_breaks_data_governance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            path = root / "evidence/data_governance.json"
            value = json.loads(path.read_text())
            value["credential_findings"] = [{
                "kind": "assigned_secret", "path": "$.data/config.txt",
            }]
            path.write_text(json.dumps(value))
            report = self.certify(history)
            self.assertFalse(report["gates"]["data_governance"])
            self.assertIn("data_governance", report["failed_gates"])

    def test_synthetic_pii_breaks_data_governance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            task_path = Path(
                history["holdouts"][0]["jobs"][0]["result"]["task_path"]
            )
            task = json.loads(task_path.read_text())
            task["public_input"] = {
                "initial_user_message": "请联系 fixture@example.test",
                "materials": [],
            }
            task_path.write_text(json.dumps(task))
            report = self.certify(history)
            self.assertFalse(report["gates"]["data_governance"])
            self.assertIn("data_governance", report["failed_gates"])

    def test_governance_report_cannot_hide_a_credential_in_the_frozen_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            result = history["holdouts"][0]["jobs"][0]["result"]
            task_path = Path(result["task_path"])
            task = json.loads(task_path.read_text())
            task["public_input"] = {
                "initial_user_message": "use api_key=abcdefghijklmnopqrstuvwx",
                "materials": [],
            }
            task_path.write_text(json.dumps(task))
            result["live_rollout"]["task_sha256"] = __import__("hashlib").sha256(
                task_path.read_bytes()
            ).hexdigest()
            report = self.certify(history)
            self.assertFalse(report["gates"]["data_governance"])
            self.assertFalse(report["measurements"]["data_governance"]["all_verified"])

    def test_unpinned_container_dependency_breaks_production_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            (root / "evidence/requirements-dev.txt").write_text("pytest>=8,<10\n")
            report = self.certify(history)
            self.assertFalse(report["gates"]["container_reproducibility"])
            self.assertIn("container_reproducibility", report["failed_gates"])

    def test_privacy_report_cannot_hide_policy_visible_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["episodes"][0]["transitions"][0]["result"] = {
                "status": 200,
                "token": "sk-abcdefghijklmnopqrstuvwxyz123456",
            }
            report = self.certify(history)
            self.assertFalse(report["gates"]["trajectory_privacy"])
            self.assertIn("trajectory_privacy", report["failed_gates"])

    def test_policy_visible_pii_breaks_trajectory_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["episodes"][0]["transitions"][0][
                "result"
            ] = {
                "status": 200,
                "user_query": "email fixture@example.test",
            }
            report = self.certify(history)
            self.assertFalse(report["gates"]["trajectory_privacy"])
            self.assertIn("trajectory_privacy", report["failed_gates"])

    def test_trainer_only_pii_breaks_trajectory_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["episodes"][0]["transitions"][0][
                "trainer_metadata"
            ] = {"operator_note": "contact fixture@example.test"}
            report = self.certify(history)
            self.assertFalse(report["gates"]["trajectory_privacy"])
            self.assertIn("trajectory_privacy", report["failed_gates"])

    def test_category_mix_prevents_single_route_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            for batch in history["holdouts"]:
                for job in batch["jobs"]:
                    job["result"]["category"] = "multi_step_agentic"
            report = self.certify(history)
            self.assertFalse(report["gates"]["category_mix"])
            self.assertFalse(report["gates"]["independent_holdout_batches"])

    def test_user_simulator_needs_multiple_real_outcome_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            for batch in history["holdouts"]:
                for job in batch["jobs"]:
                    for episode in job["result"]["live_rollout"]["episodes"]:
                        result = next(
                            step["result"] for step in episode["trajectory"]
                            if step["path"] == "/v1/user_simulator"
                        )
                        result.update(
                            match_status="matched",
                            outcome_category="user_acceptance",
                            reason_code="accepted",
                        )
            report = self.certify(history)
            self.assertFalse(report["gates"]["user_simulator_outcome_coverage"])

    def test_holdout_summary_cannot_hide_reused_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            first = history["holdouts"][0]["jobs"][0]["result"]
            second = history["holdouts"][1]["jobs"][0]["result"]
            second["sample_seed"] = first["sample_seed"]
            report = self.certify(history)
            self.assertFalse(report["gates"]["fresh_holdout"])
            self.assertFalse(report["measurements"]["fresh_holdout_evidence"]["verified"])

    def test_holdout_checks_all_generation_retry_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batches = []
            for index, (base, attempts) in enumerate(((100, [100, 101]), (101, [101]))):
                task = root / f"task-{index}.json"
                task.write_text(json.dumps({"task": str(index)}))
                manifest = root / f"sample-{index}.json"
                manifest.write_text(json.dumps({
                    "attempts": [{"seed": seed} for seed in attempts],
                }))
                batches.append({
                    "holdout_batch": index + 1,
                    "summary": {"fresh_tasks_verified": True},
                    "jobs": [{"result": {
                        "task_path": str(task), "sample_manifest": str(manifest),
                        "sample_seed": base,
                    }}],
                })
            policy = {"min_holdout_batches": 2, "min_tasks": 1}
            self.assertFalse(certifier.fresh_holdout_evidence(batches, policy)["verified"])
            second = Path(batches[1]["jobs"][0]["result"]["sample_manifest"])
            second.write_text(json.dumps({"attempts": [{"seed": 200}]}))
            batches[1]["jobs"][0]["result"]["sample_seed"] = 200
            self.assertTrue(certifier.fresh_holdout_evidence(batches, policy)["verified"])
            development_manifest = root / "development.json"
            development_manifest.write_text(json.dumps({
                "attempts": [{"seed": 199}, {"seed": 200}],
            }))
            development = [{"jobs": [{"result": {
                "sample_seed": 199,
                "sample_manifest": str(development_manifest),
            }}]}]
            self.assertFalse(certifier.fresh_holdout_evidence(
                batches, policy, development_reports=development,
            )["verified"])

    def test_transition_terminal_flags_must_match_episode_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            episode = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]["episodes"][0]
            episode["transitions"][-1]["terminated"] = False
            report = self.certify(history)
            self.assertFalse(report["gates"]["trajectory_schema"])

    def test_episode_reward_and_state_match_http_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            episode = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]["episodes"][0]
            self.assertTrue(certifier.complete_episode(episode))
            episode["trajectory"][-3]["result"]["reward"] = 0.0
            self.assertFalse(certifier.complete_episode(episode))
            episode["trajectory"][-3]["result"]["reward"] = 1.0
            episode["trajectory"][-1]["result"]["business_state"] = {"tampered": []}
            self.assertFalse(certifier.complete_episode(episode))

    def test_episode_action_and_observation_match_http_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            episode = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]["episodes"][0]
            self.assertTrue(certifier.complete_episode(episode))
            response = next(step for step in episode["trajectory"] if step["path"] == "/v1/agent_response")
            response["body"]["content"] = "different answer"
            self.assertFalse(certifier.complete_episode(episode))
            response["body"]["content"] = "done"
            observation = next(step for step in episode["trajectory"] if step["path"] == "/v1/observation")
            observation["result"] = {"tampered": True}
            self.assertFalse(certifier.complete_episode(episode))
            observation["result"] = {}
            episode["trajectory"].insert(-1, {
                "method": "POST", "path": "/v1/agent_response",
                "body": {"content": "extra"}, "status": 200, "result": {},
            })
            self.assertFalse(certifier.complete_episode(episode))

    def test_episode_requires_reset_tools_and_replay_http_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            episode = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]["episodes"][0]
            self.assertTrue(certifier.complete_episode(episode))
            episode["trajectory"][0]["body"]["seed"] = -1
            self.assertFalse(certifier.complete_episode(episode))
            episode["trajectory"][0]["body"]["seed"] = episode["seed"]
            episode["trajectory"][3]["result"] = {"tools": "forged"}
            self.assertFalse(certifier.complete_episode(episode))
            episode["trajectory"][3]["result"] = {"tools": []}
            episode["trajectory"][-2]["result"] = {"events": []}
            self.assertFalse(certifier.complete_episode(episode))

    def test_real_rollout_emitter_matches_episode_http_schema(self):
        class App:
            reward = 0.0

            def __init__(self, with_tool=False):
                self.with_tool = with_tool

            def handle(self, method, path, body, headers):
                if path == "/v1/reset":
                    self.reward = 0.0
                    result = {}
                elif path == "/v1/state":
                    result = {"business_state": {}}
                elif path == "/v1/reward":
                    result = {"reward": self.reward}
                elif path == "/v1/tools":
                    result = {"tools": ([{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {"type": "object"}},
                    }] if self.with_tool else [])}
                elif path == "/v1/tools/lookup":
                    result = {"value": 1}
                elif path == "/v1/observation":
                    result = {}
                elif path == "/v1/agent_response":
                    self.reward = 1.0
                    result = {}
                elif path == "/v1/user_simulator":
                    result = {
                        "user_query": "accepted", "should_end": True,
                        "termination_reason": "completed",
                    }
                elif path == "/v1/replay":
                    result = {"events": []}
                else:
                    raise AssertionError(path)
                return 200, result, {}

        class Client:
            def __init__(self, actions):
                self.actions = iter(actions)

            def chat(self, messages, **kwargs):
                return SimpleNamespace(
                    content=json.dumps(next(self.actions)),
                    model="policy", id="response-1", finish_reason="stop", usage={},
                )

        with patch.dict("os.environ", {"SANDBOX_TRAINER_API_KEY": "test-key"}):
            for with_tool in (False, True):
                actions = ([{"kind": "tool", "name": "lookup", "arguments": {"key": 1}}]
                           if with_tool else [])
                actions.append({"kind": "respond", "content": "done"})
                episode = rollout_runner.episode(
                    App(with_tool), {"task": "hello"}, Client(actions), 7, 3,
                )
                self.assertTrue(certifier.complete_episode(episode), episode_errors(episode))

    def test_material_manifest_detects_post_certification_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            task = sandbox / "task.json"
            task.write_text('{"task":"immutable"}')
            rollout = {"episodes": [{"agent_success": True}]}
            (sandbox / "live_rollout.json").write_text(json.dumps(rollout))
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(sandbox),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 1,
            }
            manifest = {"version": "1.0", "kind": "agentic_rl_pretraining_materials", "items": [item]}
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            fingerprint = lambda sandbox_root, project: "sandbox-proof"
            self.assertTrue(verifier.verify(manifest, ROOT, fingerprint=fingerprint)["verified"])
            task.write_text('{"task":"tampered"}')
            report = verifier.verify(manifest, ROOT, fingerprint=fingerprint)
            self.assertFalse(report["verified"])
            self.assertIn("task_digest", report["failed_gates"])

    def test_failed_artifact_verification_revokes_certification(self):
        report = {"certified": True, "gates": {}, "failed_gates": []}
        certifier.attach_artifact_verification(report, {
            "verified": False, "failed_gates": ["task_digest"],
        })
        self.assertFalse(report["certified"])
        self.assertIn("material_artifacts_immutable", report["failed_gates"])

    def test_portable_bundle_is_a_final_certification_gate(self):
        report = {
            "certified": True,
            "gates": {"base": True},
            "failed_gates": [],
            "materials_manifest": {"dataset_sha256": "dataset"},
        }
        certifier.attach_bundle_verification(report, {
            "verified": True, "bundle_version": certifier.BUNDLE_VERSION,
            "production_contract_ready": True,
            "trusted_attestation": True,
            "dataset_split_ready": True,
            "task_family_split_ready": True,
            "generation_provenance_ready": True,
            "container_rollout_ready": True,
            "portable_build_context_ready": True,
            "certification_policy_ready": True,
            "container_reward_calibration_ready": True,
            "provider_identity_ready": True,
            "data_governance_ready": True,
            "task_lineage_ready": True,
            "production_preflight_ready": True,
            "experiment_config_ready": True,
            "experiment_contract_ready": True,
            "trajectory_purpose_ready": True,
            "model_response_provenance_ready": True,
            "model_response_authorization_ready": True,
            "consumer_records_ready": True,
            "metadata_privacy_ready": True,
            "evaluator_independence_ready": True,
            "source_dataset_sha256": "different",
        })
        self.assertFalse(report["certified"])
        self.assertIn("portable_materials_bundle", report["failed_gates"])
        certifier.attach_bundle_verification(report, {
            "verified": True, "bundle_version": certifier.BUNDLE_VERSION,
            "production_contract_ready": True,
            "trusted_attestation": True,
            "dataset_split_ready": True,
            "task_family_split_ready": True,
            "generation_provenance_ready": True,
            "container_rollout_ready": True,
            "portable_build_context_ready": True,
            "certification_policy_ready": True,
            "container_reward_calibration_ready": True,
            "provider_identity_ready": True,
            "data_governance_ready": True,
            "task_lineage_ready": True,
            "production_preflight_ready": True,
            "experiment_config_ready": True,
            "experiment_contract_ready": True,
            "trajectory_purpose_ready": True,
            "model_response_provenance_ready": True,
            "model_response_authorization_ready": True,
            "consumer_records_ready": True,
            "metadata_privacy_ready": True,
            "evaluator_independence_ready": True,
            "source_dataset_sha256": "dataset",
        })
        self.assertTrue(report["certified"])

    def test_verified_v11_bundle_cannot_satisfy_current_release_gate(self):
        report = {
            "certified": True,
            "gates": {"base": True},
            "failed_gates": [],
            "materials_manifest": {"dataset_sha256": "dataset"},
        }
        certifier.attach_bundle_verification(report, {
            "verified": True,
            "bundle_version": "11.0",
            "production_contract_ready": True,
            "trusted_attestation": True,
            "dataset_split_ready": True,
            "task_family_split_ready": True,
            "generation_provenance_ready": True,
            "container_rollout_ready": True,
            "container_reward_calibration_ready": True,
            "provider_identity_ready": True,
            "task_lineage_ready": True,
            "production_preflight_ready": False,
            "source_dataset_sha256": "dataset",
        })
        self.assertFalse(report["certified"])
        self.assertIn("portable_materials_bundle", report["failed_gates"])

    def test_legacy_bundle_cannot_satisfy_production_contract_gate(self):
        report = {
            "certified": True,
            "gates": {"base": True},
            "failed_gates": [],
            "materials_manifest": {"dataset_sha256": "dataset"},
        }
        certifier.attach_bundle_verification(report, {
            "verified": True,
            "production_contract_ready": False,
            "trusted_attestation": True,
            "dataset_split_ready": True,
            "task_family_split_ready": True,
            "generation_provenance_ready": True,
            "container_rollout_ready": True,
            "source_dataset_sha256": "dataset",
        })
        self.assertFalse(report["certified"])
        self.assertIn("portable_materials_bundle", report["failed_gates"])

    def test_v2_manifest_tracks_sandbox_files_independently_of_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            app = root / "app.py"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            app.write_text("# v1\n")
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "historical-evaluator-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "old-evaluator",
                "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            self.assertTrue(verifier.verify(manifest, ROOT)["verified"])
            app.write_text("# changed\n")
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_digest", report["failed_gates"])

    def test_v2_manifest_detects_changed_certification_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            (root / "app.py").write_text("# app\n")
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "evaluator", "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            (root / "acceptance_result.json").write_text('{"business_acceptance":"failed"}')
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_evidence_digest", report["failed_gates"])

    def test_v2_manifest_detects_added_executable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            (root / "app.py").write_text("# app\n")
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "evaluator", "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            (root / "helper.py").write_text("# injected\n")
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_digest", report["failed_gates"])


if __name__ == "__main__":
    unittest.main()
