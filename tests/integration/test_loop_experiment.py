import importlib.util
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from env_factory.tasks.task_portability import prepare_sandbox_task
from env_factory.sandbox_scoring import SCORE_RUBRIC, evidence_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        path = ROOT / "scripts" / "rollout" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loop = load("loop_experiment")
rollout = load("run_live_rollout")


class ExperimentTest(unittest.TestCase):
    def test_direct_replay_gates_failed_delivery_score(self):
        with patch.object(loop, "_build_one_unfinalized", return_value={
            "passed": False, "score": 9.44,
            "offline_score": 9.38, "failure_class": "live_reward_calibration",
        }):
            result = loop._build_one(ROOT, Path("task.json"), Path("sandbox"), {})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["diagnostic_score"], 9.44)
        self.assertEqual(result["delivery_score_kind"], "gated_final_10_point")

    def test_failed_delivery_has_zero_effective_score(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            loop, "_build_one", return_value={
                "passed": False, "score": 9.44,
                "offline_score": 9.38, "failure_class": "live_reward_calibration",
            },
        ):
            result = loop.build_one(
                ROOT, Path(directory) / "task.json", Path(directory) / "sandbox",
                {"sandbox_runtime": "none"},
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["diagnostic_score"], 9.44)
        self.assertEqual(result["offline_score"], 9.38)
        self.assertEqual(result["delivery_score_kind"], "gated_final_10_point")

    def test_training_ready_is_published_after_final_cleanup(self):
        for cleanup_code in (0, 1):
            with self.subTest(cleanup_code=cleanup_code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "status.json").write_text(json.dumps({
                    "status": "success", "success": True, "exit_code": 0}))
                verified = {"passed": True, "score": 10, "build": {"exit_code": 0},
                    "live_rollout_verified": True, "live_reward_calibration_verified": True,
                    "data_governance_verified": True, "trajectory_privacy_verified": True,
                    "sandbox_score": {"passed": True}}
                with patch.object(loop, "_build_one", return_value=verified), patch.object(
                    loop, "run_process", return_value={"exit_code": cleanup_code}):
                    result = loop.build_one(ROOT, output / "task.json", output,
                                            {"sandbox_runtime": "docker"})
                self.assertEqual(result["training_ready"], cleanup_code == 0)
                self.assertEqual(json.loads((output / "status.json").read_text())["training_ready"], cleanup_code == 0)

    def test_builder_owner_codes_survive_failure_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for phase, code, target in (
                ("task_contract_rejected", "TASK_CONTRACT", "task_pipeline"),
                ("platform_contract_rejected", "PLATFORM_RUNTIME", "platform_runtime_boundary"),
                ("runtime_timeout_rejected", "BUILD_RUNTIME_TIMEOUT", "platform_or_task_contract"),
                ("repair_no_progress", "BUILD_REPAIR_NO_PROGRESS", "sandbox_builder"),
                ("repair_budget_exhausted", "BUILD_REPAIR_BUDGET", "sandbox_builder"),
            ):
                (output / "status.json").write_text(json.dumps({
                    "status": "failed", "failed_phase": phase,
                    "failure_code": code,
                }), encoding="utf-8")
                result = loop.classify_build_failure(output)
                self.assertEqual(result["failure_code"], code)
                self.assertEqual(result["repair_target"], target)

    @staticmethod
    def write_task_lineage(output: Path, task_path: Path) -> None:
        prepare_sandbox_task(task_path, output)

    @staticmethod
    def mock_sandbox_score(output: Path, threshold: float = 8) -> dict:
        return {
            "checks": [{"name": name, "weight": weight, "passed": True,
                        "evidence": "verified", "critical": critical}
                       for name, weight, critical in SCORE_RUBRIC],
            "mode": "offline_executable", "network_used": False, "model_used": False,
            "live_rollout_verified": False, "threshold": threshold,
            "score": 10.0, "eligible": True, "passed": True,
            "failed_critical_gates": [],
            "model": loop.MODEL, "review_model": loop.MODEL,
            "evidence_fingerprint": evidence_fingerprint(output, ROOT),
        }

    def test_concurrent_attempts_receive_distinct_stable_container_tags(self):
        first = loop.sandbox_image_tag(Path("/tmp/experiment/sample-1/attempt-1"))
        second = loop.sandbox_image_tag(Path("/tmp/experiment/sample-2/attempt-1"))
        self.assertEqual(first, loop.sandbox_image_tag(
            Path("/tmp/experiment/sample-1/attempt-1")
        ))
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^envfactory-sandbox-[0-9a-f]{20}$")

    def test_zero_round_limit_is_unbounded(self):
        self.assertEqual(list(itertools.islice(loop.round_numbers(0), 25))[-1], 25)
        self.assertEqual(list(loop.round_numbers(3)), [1, 2, 3])

    def test_denominator_includes_generation_failure_and_threshold_is_inclusive(self):
        report = loop.summarize([
            {"task_path": "a", "score": 9, "passed": True, "category": "direct_response"},
            {"task_path": "b", "score": 8, "passed": True, "category": "simple_agentic"},
            loop.failure("generation", "missing"),
        ], 8)
        self.assertEqual(report["requested"], 3)
        self.assertEqual(report["generated"], 2)
        self.assertEqual(report["qualified"], 2)
        self.assertEqual(report["end_to_end_rate"], 2 / 3)
        self.assertIsNone(report["post_score_survival_rate"])
        self.assertFalse(report["all_passed"])
        self.assertFalse(report["live_rollout_verified"])

    def test_timeout_and_nonzero_exit_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = loop.run_process([sys.executable, "-c", "import time; time.sleep(10)"], root, root / "timeout.log", .05)
            self.assertEqual(result["exit_code"], 124)
            self.assertTrue(result["timed_out"])
            result = loop.run_process([sys.executable, "-c", "raise SystemExit(7)"], root, root / "exit.log", 5)
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(loop.ACTIVE_PROCESSES)

    def test_pause_persists_only_active_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "runtime_state.json"
            previous_path = loop.BUDGET_STATE_PATH
            previous_started = loop.ACTIVE_RUN_STARTED
            try:
                loop.BUDGET_STATE_PATH = state_path
                loop.ACTIVE_RUN_STARTED = 100.0
                with patch.object(loop.time, "time", return_value=112.5):
                    loop.finish_active_budget("paused")
                state = json.loads(state_path.read_text())
                self.assertEqual(state["active_seconds"], 12.5)
                self.assertEqual(state["status"], "paused")
                self.assertNotIn("run_started_at", state)
            finally:
                loop.BUDGET_STATE_PATH = previous_path
                loop.ACTIVE_RUN_STARTED = previous_started

    def test_promotion_uses_yield_targets_instead_of_all_samples(self):
        passing = {
            "task_path": "task", "score": 9, "passed": True,
            "category": "simple_agentic",
            "task_score": {"eligible": True, "score": 9},
            "sandbox_score": {"passed": True, "score": 9},
            "live_rollout_verified": True,
        }
        results = [dict(passing) for _ in range(8)] + [
            {"task_path": "bad", "score": 0, "passed": False,
            "category": "simple_agentic", "failure_class": "build",
            "failure_code": "BUILD_BUSINESS",
             "task_score": {"eligible": True, "score": 9},
             "sandbox_score": {"passed": False, "score": 0}},
            loop.failure("generation", "missing", category="simple_agentic"),
        ]
        summary = loop.summarize(results, 8, targets={
            "task_yield": .9, "build_yield": .8, "end_to_end_rate": .7,
            "qualified_mean": 8.5, "category_rate": .6,
        })
        self.assertTrue(summary["target_met"])
        self.assertFalse(summary["all_passed"])
        self.assertEqual(summary["task_good_yield"], .9)
        self.assertAlmostEqual(summary["sandbox_build_yield"], 8 / 9)

    def test_rollout_failure_does_not_reduce_build_yield(self):
        result = {
            "task_path": "task",
            "score": 7.2,
            "passed": False,
            "failure_class": "live_rollout",
            "task_score": {"eligible": True, "score": 9},
            "sandbox_score": {"passed": True, "score": 8.5},
        }
        summary = loop.summarize([result], 8)
        self.assertEqual(summary["sandbox_build_yield"], 1)
        self.assertEqual(summary["end_to_end_rate"], 0)
        self.assertEqual(summary["final_after_offline"], 0)
        self.assertEqual(summary["post_score_survival_rate"], 0)

    def test_offline_pass_is_not_counted_as_live_survival(self):
        result = {
            "task_path": "task", "score": 10.0, "passed": True,
            "task_score": {"eligible": True, "score": 10.0},
            "sandbox_score": {"passed": True, "score": 10.0},
            "live_rollout_verified": False,
        }
        summary = loop.summarize([result], 8, validation="offline")
        self.assertEqual(summary["offline_qualified"], 1)
        self.assertEqual(summary["final_after_offline"], 0)
        self.assertIsNone(summary["post_score_survival_rate"])

    def test_all_infrastructure_failures_do_not_form_a_quality_round(self):
        results = [loop.failure("infrastructure", "network unavailable") for _ in range(5)]
        summary = loop.summarize(results, 8, targets={
            "task_yield": .85, "build_yield": .8, "end_to_end_rate": .7,
            "qualified_mean": 8.5, "category_rate": .6,
        })
        self.assertFalse(summary["valid_quality_round"])
        self.assertFalse(summary["target_met"])

    def test_resume_keeps_completed_jobs_and_restarts_incomplete_in_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = {"passed": True, "score": 9, "task_path": "one"}
            report = {"round": 1, "state": "running", "jobs": [
                {"id": 1, "state": "complete", "result": completed},
                {"id": 2, "state": "running", "attempt": 1, "task_path": "two"}]}
            config = {"threshold": 8, "max_concurrency": 1, "build_mode": "clean", "generate_count": 0}
            with patch.object(loop, "build_one", return_value={"passed": True, "score": 9, "task_path": "two"}) as build:
                result = loop.run_round(ROOT, root, config, report)
            self.assertEqual(build.call_count, 1)
            self.assertTrue(str(build.call_args.args[2]).endswith("sample-002/attempt-2"))
            self.assertTrue(result["summary"]["all_passed"])
            disk = json.loads((root / "round-01/round_report.json").read_text())
            self.assertEqual(disk["jobs"][0]["result"], completed)

    def test_interrupted_generation_is_not_silently_resampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"generate_count": 5, "threshold": 8, "max_concurrency": 1}
            with patch.object(loop, "run_process") as process:
                report = loop.run_round(ROOT, Path(tmp), config, {"round": 1, "state": "running", "generation_started": True})
            process.assert_not_called()
            self.assertEqual(report["summary"]["requested"], 5)
            self.assertEqual(report["summary"]["failures"], {"generation": 5})
            self.assertEqual(report["summary"]["failure_codes"], {"INFRA": 5})

    def test_generation_receives_reproducible_sampling_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "generate_count": 1,
                "threshold": 8,
                "max_concurrency": 1,
                "generation_timeout": 10,
                "experiment_seed": 700,
                "generation_hops": 4,
                "route_attempts": 2,
                "training_mix": "direct_response=0.2,simple_agentic=0.3,multi_step_agentic=0.5",
            }
            with patch.object(
                loop,
                "run_process",
                return_value={"exit_code": 1, "timed_out": False, "seconds": 0.1},
            ) as process:
                loop.run_round(ROOT, Path(tmp), config, {"round": 3, "state": "running"})
            command = process.call_args.args[0]
            self.assertEqual(command[command.index("--seed") + 1], "702")
            self.assertEqual(command[command.index("--hops") + 1], "4")
            self.assertEqual(command[command.index("--route-attempts") + 1], "2")
            self.assertEqual(
                command[command.index("--training-mix") + 1],
                config["training_mix"],
            )

    def test_completed_sample_builds_while_generation_is_still_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_started = threading.Event()
            overlap = []

            def generate(*_args):
                task_dir = root / "round-01/generation/task/task-1"
                task_dir.mkdir(parents=True)
                task_path = task_dir / "task.json"
                task_path.write_text("{}", encoding="utf-8")
                (task_dir / "sample_manifest.json").write_text(json.dumps({
                    "batch_index": 1, "status": "completed",
                    "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                }), encoding="utf-8")
                overlap.append(build_started.wait(3))
                return {"exit_code": 0, "timed_out": False, "seconds": .1}

            def build(*_args):
                build_started.set()
                return {"passed": True, "score": 9}

            config = {
                "generate_count": 1, "threshold": 8, "max_concurrency": 2,
                "generation_timeout": 10, "build_mode": "clean",
            }
            with patch.object(loop, "run_process", side_effect=generate), \
                    patch.object(loop, "build_one", side_effect=build) as builder:
                report = loop.run_round(ROOT, root, config, {"round": 1, "state": "running"})
            self.assertEqual(overlap, [True])
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(report["jobs"][0]["state"], "complete")

    def test_generation_manifests_preserve_failed_route_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "round-01/generation/task/task-9"
            task_dir.mkdir(parents=True)
            manifest = {
                "batch_index": 1, "training_category": "multi_step_agentic",
                "sample_seed": 99, "status": "failed",
            }
            (task_dir / "sample_manifest.json").write_text(json.dumps(manifest))
            (task_dir / "failure.json").write_text(json.dumps({
                "failure_class": "GEN_SCHEMA", "message": "bad schema",
            }))
            config = {"generate_count": 1, "threshold": 8, "max_concurrency": 1, "build_mode": "clean"}
            report = loop.run_round(
                ROOT, root, config,
                {"round": 1, "state": "running", "generation_started": True},
            )
            result = report["jobs"][0]["result"]
            self.assertEqual(result["failure_code"], "GEN_SCHEMA")
            self.assertEqual(result["category"], "multi_step_agentic")
            self.assertEqual(result["sample_seed"], 99)

    def test_failed_manifest_waits_for_failure_report_before_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task-1"
            task_dir.mkdir()
            manifest_path = task_dir / "sample_manifest.json"
            sample = {"batch_index": 1, "status": "failed", "training_category": "simple_agentic"}
            manifest_path.write_text(json.dumps(sample), encoding="utf-8")
            self.assertIsNone(loop.resolve_generated_job(
                1, (manifest_path, sample), generation_done=False,
            ))
            (task_dir / "failure.json").write_text(json.dumps({
                "failure_class": "GEN_SCHEMA", "message": "bad schema",
            }), encoding="utf-8")
            resolved = loop.resolve_generated_job(
                1, (manifest_path, sample), generation_done=False,
            )
            self.assertEqual(resolved["result"]["failure_code"], "GEN_SCHEMA")

    def test_interrupted_generation_does_not_build_uncommitted_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "round-01/generation/task/task-1"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text("{}")
            manifest_path = task_dir / "sample_manifest.json"
            manifest = {"batch_index": 1, "status": "generated", "sample_seed": 9}
            manifest_path.write_text(json.dumps(manifest))
            config = {"generate_count": 1, "threshold": 8, "max_concurrency": 1}
            with patch.object(loop, "build_one") as build:
                report = loop.run_round(
                    ROOT, root, config,
                    {"round": 1, "state": "running", "generation_started": True},
                )
            build.assert_not_called()
            self.assertEqual(report["jobs"][0]["result"]["failure_code"], "INFRA")

    def test_completed_generation_requires_matching_task_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "round-01/generation/task/task-1"
            task_dir.mkdir(parents=True)
            task_path = task_dir / "task.json"
            task_path.write_text("{}")
            manifest_path = task_dir / "sample_manifest.json"
            manifest = {
                "batch_index": 1, "status": "completed", "sample_seed": 9,
                "task_sha256": "0" * 64,
            }
            manifest_path.write_text(json.dumps(manifest))
            config = {"generate_count": 1, "threshold": 8, "max_concurrency": 1, "build_mode": "clean"}
            with patch.object(loop, "build_one") as build:
                rejected = loop.run_round(
                    ROOT, root, config,
                    {"round": 1, "state": "running", "generation_started": True},
                )
            build.assert_not_called()
            self.assertEqual(rejected["jobs"][0]["result"]["failure_code"], "INFRA")
            manifest["task_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(loop, "build_one", return_value={"passed": True, "score": 9}) as build:
                accepted = loop.run_round(
                    ROOT, root, config,
                    {"round": 1, "state": "running", "generation_started": True},
                )
            build.assert_called_once()
            self.assertEqual(accepted["jobs"][0]["state"], "complete")

    def test_one_job_exception_does_not_abort_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"generate_count": 0, "task_paths": ["one", "two"], "threshold": 8, "max_concurrency": 1, "build_mode": "clean"}
            with patch.object(loop, "build_one", side_effect=[ValueError("bad input"), {"passed": True, "score": 9}]):
                report = loop.run_round(ROOT, Path(tmp), config, {"round": 1, "state": "running"})
            self.assertEqual(report["summary"]["requested"], 2)
            self.assertEqual(report["summary"]["failures"], {"infrastructure": 1})

    def test_buildability_failure_is_not_mislabeled_as_luna_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "buildability.json").write_text(json.dumps({
                    "buildable": False,
                    "issues": [{"code": "EXTERNAL_CAPABILITY_UNAVAILABLE"}],
                }))
                return {"exit_code": 4, "timed_out": False, "seconds": .1}
            config = {"threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "task_quality")
            self.assertEqual(result["failure_code"], "EXTERNAL_CAPABILITY_UNAVAILABLE")

    def test_infrastructure_disconnect_retries_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}", encoding="utf-8")
            output = root / "sandbox"
            calls = []
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })

            def disconnected(command, *_args):
                calls.append(command)
                output.mkdir(exist_ok=True)
                (output / "status.json").write_text(json.dumps({
                    "status": "failed",
                    "failure_code": "INFRA", "failure_category": "infrastructure",
                    "failed_phase": "node_development",
                }), encoding="utf-8")
                return {"exit_code": 5, "timed_out": False, "seconds": .1}

            config = {
                "threshold": 8, "max_attempts": 2, "build_timeout": 10,
                "build_mode": "clean", "infrastructure_retries": 1,
            }
            with patch("env_factory.tasks.task_quality.score_file", return_value=quality), \
                    patch.object(loop, "run_process", side_effect=disconnected):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--resume", calls[0])
            self.assertIn("--resume", calls[1])
            self.assertEqual(result["failure_code"], "INFRA")

    def test_task_scoring_exception_counts_as_task_quality_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "task.json"
            task_path.write_text('{"environment_plan": []}', encoding="utf-8")
            result = loop.build_one(
                ROOT, task_path, root / "sandbox",
                {"threshold": 8, "build_mode": "clean", "sandbox_runtime": "none"},
            )
        self.assertEqual(result["failure_class"], "task_quality")
        self.assertEqual(result["task_score"]["score"], 0)
        self.assertFalse(result["task_score"]["eligible"])
        self.assertEqual(result["category"], "unknown")
        summary = loop.summarize([result], 8)
        self.assertEqual(summary["task_good_yield"], 0)
        self.assertTrue(summary["valid_quality_round"])

    def test_build_failure_uses_last_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })

            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "status.json").write_text(json.dumps({
                    "status": "failed", "phase": "failed",
                    "failed_phase": "runtime_validation",
                }))
                return {"exit_code": 5, "timed_out": False, "seconds": .1}

            config = {"threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "build")
            self.assertEqual(result["build_failed_phase"], "runtime_validation")
            self.assertEqual(result["failure_code"], "BUILD_RUNTIME_INTEGRITY")
            self.assertEqual(result["repair_target"], "platform_runtime_boundary")

    def test_scoring_requires_fresh_report_bound_to_current_sandbox(self):
        quality = SimpleNamespace(to_dict=lambda: {
            "eligible": True, "score": 9, "training_category": "simple_agentic",
        })
        for case in ("missing", "wrong_fingerprint", "bare_score"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task_path = root / "task.json"
                task_path.write_text("{}")
                output = root / "sandbox"
                output.mkdir()
                report_path = output / "score_summary.json"
                report_path.write_text('{"sandboxes":[{"passed":true,"score":10}]}')

                def process(command, cwd, log, timeout):
                    rendered = " ".join(command)
                    if "develop_sandbox_with_agent.sh" in rendered:
                        self.write_task_lineage(output, task_path)
                    if "score_sandbox_offline.py" in rendered:
                        self.assertFalse(report_path.exists())
                        if case == "wrong_fingerprint":
                            report = self.mock_sandbox_score(output)
                            report["evidence_fingerprint"] = "stale"
                            report_path.write_text(json.dumps({"sandboxes": [report]}))
                        if case == "bare_score":
                            report_path.write_text(json.dumps({"sandboxes": [{
                                "passed": True, "score": 10,
                                "model": loop.MODEL, "review_model": loop.MODEL,
                                "evidence_fingerprint": evidence_fingerprint(output, ROOT),
                            }]}))
                    return {"exit_code": 0, "timed_out": False, "seconds": .1}

                config = {"threshold": 8, "max_attempts": 2, "build_timeout": 10,
                          "score_timeout": 10, "build_mode": "clean", "validation": "offline"}
                with (
                    patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                    patch.object(loop, "run_process", side_effect=process),
                ):
                    result = loop.build_one(ROOT, task_path, output, config)
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure_class"], "offline_validation")
                if case == "missing":
                    self.assertFalse(report_path.exists())

    def test_agent_startup_failure_is_infrastructure_not_build_yield(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })

            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "status.json").write_text(json.dumps({
                    "status": "failed", "phase": "failed",
                    "failed_phase": "node_development",
                    "failure_category": "infrastructure", "failure_code": "INFRA",
                }))
                return {"exit_code": 5, "timed_out": False, "seconds": .1}

            config = {"threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "infrastructure")
            self.assertEqual(result["failure_code"], "INFRA")
            self.assertEqual(result["build_failed_phase"], "node_development")
            self.assertFalse(loop.summarize([result], 8)["valid_quality_round"])

    def test_live_build_runs_real_reward_counterfactual_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            output.mkdir()
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            commands = []

            def process(command, cwd, log, timeout):
                commands.append(command)
                if "develop_sandbox_with_agent.sh" in " ".join(command):
                    self.write_task_lineage(output, task_path)
                if "score_sandbox_offline.py" in " ".join(command):
                    (output / "score_summary.json").write_text(json.dumps({
                        "sandboxes": [self.mock_sandbox_score(output)],
                    }))
                elif "audit_data_governance.py" in " ".join(command):
                    (output / "data_governance.json").write_text(json.dumps({
                        "eligible_for_external_model_processing": True,
                    }))
                elif "run_live_rollout.py" in " ".join(command):
                    (output / "live_rollout.json").write_text(json.dumps({
                        "passed": True, "live_rollout_verified": True,
                        "quality_score": 1.0, "episodes": [],
                    }))
                elif "audit_trajectory_privacy.py" in " ".join(command):
                    (output / "trajectory_privacy.json").write_text(json.dumps({
                        "eligible_for_policy_training_export": True,
                    }))
                elif "validate_agentic_training_value.py" in " ".join(command):
                    (output / "agentic_training_value_live.json").write_text(json.dumps({
                        "curriculum_training_ready": True,
                        "validation_mode": "live_evaluator",
                        "runtime_execution": {
                            "mode": "docker_http",
                            "container_image_id": "sha256:" + "d" * 64,
                        },
                    }))
                return {"exit_code": 0, "timed_out": False, "seconds": .1}

            config = {
                "threshold": 8, "max_attempts": 2, "build_timeout": 10,
                "score_timeout": 10, "rollout_timeout": 10,
                "build_mode": "clean", "validation": "live",
                "sandbox_runtime": "docker",
                "rollout_episodes": 3, "rollout_steps": 20,
                "rollout_min_success_rate": 0,
            }
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=process),
                patch.object(loop, "start_rollout_container", return_value={
                    "started": True,
                    "container_id": "container-1",
                    "image_id": "sha256:" + "d" * 64,
                    "base_url": "http://127.0.0.1:49152",
                    "security": {},
                }),
                patch.object(loop, "stop_rollout_container", return_value={
                    "exit_code": 0, "timed_out": False, "seconds": .1,
                }),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertTrue(result["passed"])
            self.assertTrue(result["data_governance_verified"])
            self.assertTrue(result["trajectory_privacy_verified"])
            self.assertTrue(result["live_reward_calibration_verified"])
            build_command = commands[0]
            self.assertEqual(
                build_command[build_command.index("--runtime") + 1], "docker"
            )
            image_tag = build_command[build_command.index("--tag") + 1]
            cleanup_command = next(
                command for command in commands
                if command[:3] == ["docker", "image", "rm"]
            )
            self.assertEqual(cleanup_command[-1], image_tag)
            governance_index = next(
                index for index, command in enumerate(commands)
                if "audit_data_governance.py" in " ".join(command)
            )
            rollout_index = next(
                index for index, command in enumerate(commands)
                if "run_live_rollout.py" in " ".join(command)
            )
            self.assertLess(governance_index, rollout_index)
            rollout_command = commands[rollout_index]
            self.assertEqual(
                rollout_command[rollout_command.index("--base-url") + 1],
                "http://127.0.0.1:49152",
            )
            calibration = next(
                command for command in commands
                if "validate_agentic_training_value.py" in " ".join(command)
            )
            self.assertEqual(
                calibration[calibration.index("--evaluator-mode") + 1], "live"
            )
            self.assertEqual(
                calibration[calibration.index("--base-url") + 1],
                "http://127.0.0.1:49152",
            )
            self.assertEqual(
                calibration[calibration.index("--container-image-id") + 1],
                "sha256:" + "d" * 64,
            )

    def test_data_governance_failure_prevents_live_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            output.mkdir()
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            commands = []

            def process(command, cwd, log, timeout):
                commands.append(command)
                rendered = " ".join(command)
                if "develop_sandbox_with_agent.sh" in rendered:
                    self.write_task_lineage(output, task_path)
                if "score_sandbox_offline.py" in rendered:
                    (output / "score_summary.json").write_text(json.dumps({
                        "sandboxes": [self.mock_sandbox_score(output)],
                    }))
                    return {"exit_code": 0, "timed_out": False, "seconds": .1}
                if "audit_data_governance.py" in rendered:
                    (output / "data_governance.json").write_text(json.dumps({
                        "eligible_for_external_model_processing": False,
                        "credential_findings": [{"kind": "private_key", "path": "$.data"}],
                    }))
                    return {"exit_code": 1, "timed_out": False, "seconds": .1}
                return {"exit_code": 0, "timed_out": False, "seconds": .1}

            config = {
                "threshold": 8, "max_attempts": 2, "build_timeout": 10,
                "score_timeout": 10, "rollout_timeout": 10,
                "build_mode": "clean", "validation": "live",
                "rollout_episodes": 3, "rollout_steps": 20,
                "rollout_min_success_rate": 0,
            }
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=process),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertFalse(result["passed"])
            self.assertEqual(result["failure_class"], "data_governance")
            self.assertFalse(any(
                "run_live_rollout.py" in " ".join(command) for command in commands
            ))

    def test_production_profile_cannot_disable_container_build(self):
        completed = __import__("subprocess").run(
            [
                sys.executable,
                str(ROOT / "scripts/loop_experiment.py"),
                "--certification-profile", "production",
                "--sandbox-runtime", "none",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "production certification requires --sandbox-runtime docker",
            completed.stderr,
        )

    def test_production_profile_cannot_lower_material_score_threshold(self):
        completed = __import__("subprocess").run(
            [
                sys.executable,
                str(ROOT / "scripts/loop_experiment.py"),
                "--certification-profile", "production",
                "--threshold", "7.9",
                "--sandbox-runtime", "docker",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "production certification requires --threshold >= 8",
            completed.stderr,
        )

    def test_production_profile_requires_trusted_bundle_signing_keys(self):
        completed = __import__("subprocess").run(
            [
                sys.executable,
                str(ROOT / "scripts/loop_experiment.py"),
                "--certification-profile", "production",
                "--sandbox-runtime", "docker",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "production certification requires bundle signing private and trusted public keys",
            completed.stderr,
        )

    def test_holdout_requires_fresh_tasks_and_two_of_three_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for index in range(30):
                task = root / f"task-{index}.json"
                task.write_text(json.dumps({"task": index}))
                results.append({
                    "task_path": str(task), "sample_seed": 10_000 + index,
                    "score": 9, "passed": True, "category": "multi_step_agentic",
                    "task_score": {"eligible": True, "score": 9},
                    "sandbox_score": {"passed": True, "score": 9},
                    "live_rollout": {
                        "agent_success_rate": 2 / 3,
                        "all_episodes_environment_clean": True,
                        "all_episodes_fallback_free": True,
                    },
                })
            summary = loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3, previous_seeds={1, 2, 3},
            )
            self.assertTrue(summary["target_met"])
            results[0]["live_rollout"]["agent_success_rate"] = 1 / 3
            self.assertFalse(loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3,
            )["qualified_rollout_floor_met"])
            results[0]["sample_seed"] = 1
            self.assertFalse(loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3, previous_seeds={1},
            )["fresh_tasks_verified"])


class FakeApp:
    def __init__(self, *, correct=True, fallback=False):
        self.correct, self.fallback = correct, fallback
        self.changed = False

    def business_snapshot(self):
        return {"items": [{"id": 1, "done": self.changed}]}

    def handle(self, method, path, body=None, headers=None):
        if path == "/v1/reset":
            self.changed = False
            value = {}
        elif path == "/v1/tools":
            value = {"tools": [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}]}
        elif path == "/v1/state":
            value = {"business_state": self.business_snapshot()}
        elif path == "/v1/tools/finish":
            self.changed = self.correct
            value = {"updated": 1}
        elif path == "/v1/reward":
            value = {"reward": 1 if getattr(self, "answered", False) else 0}
        elif path == "/v1/agent_response":
            self.answered = True
            value = {"accepted": True}
        elif path == "/v1/user_simulator":
            value = {"user_query": "accepted", "should_end": True, "termination_reason": "completed"}
        elif path == "/v1/replay":
            value = {"events": [{"payload": {"used_fallback": self.fallback}}]}
        else:
            value = {}
        return 200, value, {}


class RolloutTest(unittest.TestCase):
    def test_http_sandbox_client_only_accepts_loopback_origins(self):
        self.assertEqual(
            rollout.HTTPSandboxClient("http://127.0.0.1:8000").base_url,
            "http://127.0.0.1:8000",
        )
        for value in (
            "https://127.0.0.1:8000",
            "http://sandbox.example:8000",
            "http://user:secret@127.0.0.1:8000",
            "http://127.0.0.1:8000/path",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rollout.HTTPSandboxClient(value)

    def test_release_gate_requires_two_of_three_clean_successes(self):
        episodes = [
            {"agent_success": True, "issues": []},
            {"agent_success": True, "issues": []},
            {"agent_success": False, "issues": []},
        ]
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["quality_score"], 1.0)
        episodes[1]["agent_success"] = False
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertTrue(report["environment_qualified"])
        self.assertFalse(report["agent_policy_qualified"])
        self.assertIsNone(report["failure_owner"])
        episodes[1]["issues"] = ["runtime_llm_fallback"]
        report = rollout.summarize_episodes(episodes, 0)
        self.assertFalse(report["all_episodes_fallback_free"])
        self.assertEqual(report["failure_owner"], "environment")

    def run_episode(self, app):
        task = {"task": "finish item", "acceptance_contract": {"secret": "do not show"},
                "public_input": {"initial_user_message": "finish the supplied item", "materials": [{
                    "name": "item", "mime_type": "application/json", "content": '{"id": 1}'
                }]},
                "task_spec": {"goal_contract": {"row_predicates": [{"table": "items", "where": {"id": 1},
                       "values": {"done": True}, "count": 1}], "requires_state_change": True}}}
        calls = []
        actions = iter([{"kind": "tool", "name": "finish", "arguments": {}}, {"kind": "respond", "content": "done"}])
        def chat(messages, **kwargs):
            calls.append(json.dumps(messages))
            return SimpleNamespace(
                content=json.dumps(next(actions)),
                model="actual-policy-snapshot",
                id="response-id",
                finish_reason="stop",
                usage={"total_tokens": 10},
            )
        with patch.dict(os.environ, {"SANDBOX_TRAINER_API_KEY": "test"}):
            result = rollout.episode(app, task, SimpleNamespace(chat=chat), 0, 4)
        self.assertTrue(all("do not show" not in call for call in calls))
        self.assertTrue(any("Public materials" in call and "application/json" in call for call in calls))
        return result

    def test_live_agent_uses_only_public_inputs_and_records_usage(self):
        result = self.run_episode(FakeApp())
        self.assertTrue(result["agent_success"])
        self.assertEqual(
            sum(item["token_usage"]["total_tokens"] for item in result["usage"]),
            20,
        )
        self.assertEqual(
            {item["response_model"] for item in result["usage"]},
            {"actual-policy-snapshot"},
        )
        self.assertEqual(result["schema_version"], "2.0")
        self.assertEqual(len(result["transitions"]), 2)
        self.assertEqual(result["transitions"][0]["action"]["kind"], "tool")
        self.assertIsInstance(result["transitions"][0]["next_observation"], dict)
        self.assertTrue(result["transitions"][-1]["terminated"])
        self.assertEqual(
            result["transitions"][-1]["result"],
            {"status": 200, "user_query": "accepted"},
        )
        self.assertTrue(
            result["transitions"][-1]["trainer_metadata"]["user_simulator"][
                "should_end"
            ]
        )

    def test_high_reward_with_wrong_state_is_rejected(self):
        result = self.run_episode(FakeApp(correct=False))
        self.assertFalse(result["agent_success"])
        self.assertIn("reward_state_disagreement", result["issues"])

    def test_runtime_fallback_is_not_live_success(self):
        result = self.run_episode(FakeApp(fallback=True))
        self.assertFalse(result["agent_success"])
        self.assertIn("runtime_llm_fallback", result["issues"])
