import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from env_factory.tasks.task_routing import TRAINING_CATEGORIES, training_contract
from env_factory.evidence.material_artifacts import (
    DOCKERIGNORE_SOURCE, docker_build_context_digest,
)
from env_factory.sandbox_scoring import SCORE_RUBRIC


ROOT = Path(__file__).parents[2]


def load_script(name):
    path = ROOT / "scripts" / name
    if not path.is_file():
        path = ROOT / "scripts" / "sandbox" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildWorkflowTest(unittest.TestCase):
    def test_empty_tools_only_allowed_for_direct_response_in_shared_contract_gate(self):
        from generation.test_answer_contract import answer_source
        from env_factory.generation.agent_authoring import compile_source
        from env_factory.generation.artifacts import write_task_artifact
        from env_factory.tasks.task import Task
        gate = load_script("validate_contract_and_tools.py")
        source, request = answer_source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = compile_source(source, root=root, request=request)
            task_path = write_task_artifact(root, Task(a["task"], a["environment"], a["metrics"],
                task_intent=a["task_intent"], artifacts=a), "direct_response")
            contract_path, tools_path = root / "BUILD_CONTRACT.json", root / "tools.json"
            tools_path.write_text("[]")
            original = json.loads(task_path.read_text())
            for category in ("direct_response", "simple_agentic", "multi_step_agentic"):
                task = {**original, "training_category": category}
                task_path.write_text(json.dumps(task))
                contract_path.write_text(json.dumps({k: v for k, v in task.items() if k != "actions"}))
                if category == "direct_response":
                    gate.validate(task_path, contract_path, tools_path)
                else:
                    with self.assertRaisesRegex(SystemExit, "不允许"):
                        gate.validate(task_path, contract_path, tools_path)

    def test_code_agent_contract_always_has_independent_builder_node(self):
        planner = load_script("generate_development_plan.py")
        plan = planner.build_plan({"artifacts": {"generation_pipeline": {"backend": "code_agent"}}})
        self.assertEqual([node["id"] for node in plan["nodes"]], ["business_integration"])
        self.assertIn("tests/business_integration", plan["nodes"][0]["validation"][0])
        self.assertEqual(planner.build_plan({})["nodes"], [])

    def test_zero_extension_reward_timeout_is_not_sent_to_task_agent(self):
        router = load_script("route_delivery_failure.py")
        timeout = "independent baseline conformance failed\n" \
                  "runtime endpoint unavailable GET /v1/reward: timed out"
        self.assertTrue(router.zero_extension_reward_timeout({"nodes": []}, timeout))
        self.assertFalse(router.zero_extension_reward_timeout(
            {"nodes": [{"id": "task_handlers"}]}, timeout,
        ))
        self.assertFalse(router.zero_extension_reward_timeout(
            {"nodes": []}, "scenario assertion failed",
        ))

    def test_protected_reward_contract_defect_returns_task_for_regeneration(self):
        gate = load_script("protected_contract_defects.py")
        self.assertTrue(gate.requires_task_regeneration([{
            "severity": "high", "file": "BUILD_CONTRACT.json",
            "fix_required": "Align the outcome evaluator criteria and success fixture with the task rule.",
        }]))
        self.assertFalse(gate.requires_task_regeneration([{
            "severity": "high", "file": "task_impl.py",
            "fix_required": "Fix the business handler filter.",
        }]))
        self.assertFalse(gate.requires_task_regeneration([{
            "severity": "high", "file": "BUILD_CONTRACT.json",
            "fix_required": "Expose a missing field through the existing task handler.",
        }]))

    def test_platform_owned_defect_does_not_consume_task_agent_retries(self):
        gate = load_script("protected_contract_defects.py")
        self.assertTrue(gate.requires_platform_repair([{
            "severity": "high", "file": "sandbox_runtime.py",
            "category": "evaluator_fallback",
        }]))
        self.assertFalse(gate.requires_platform_repair([{
            "severity": "high", "file": "task_impl.py",
            "category": "business_handler",
        }]))

    def test_buildability_rejects_incomplete_declarative_insert_before_agent(self):
        assessor = load_script("assess_task_buildability.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "schemas").mkdir(parents=True)
            (data / "rows").mkdir()
            (data / "schemas" / "requests.json").write_text(json.dumps({
                "table_name": "requests", "columns": [
                    {"name": "request_id", "type": "VARCHAR(32)", "nullable": False},
                    {"name": "item", "type": "VARCHAR(32)", "nullable": False},
                ], "primary_key": ["request_id"],
            }), encoding="utf-8")
            (data / "rows" / "requests.jsonl").write_text(
                json.dumps({"request_id": "REQ1", "item": "A"}) + "\n", encoding="utf-8",
            )
            (root / "task.json").write_text(json.dumps({
                "environment_plan": {"mode": "stateful"},
                "artifacts": {"data_manifest": {"root": "data", "tables": [{
                    "table_name": "requests", "schema_file": "schemas/requests.json",
                    "rows_file": "rows/requests.jsonl", "row_count": 1,
                    "primary_key": ["request_id"],
                }]}},
                "tools": [{"function": {"name": "create_request", "parameters": {
                    "properties": {"item": {"type": "string"}},
                }}}],
                "tool_implementations": [{
                    "tool_name": "create_request", "operation": "insert", "table": "requests",
                    "values": {"item": "item"}, "result_field": "records",
                }],
            }), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9})
            with patch.object(assessor, "score_file", return_value=quality):
                report = assessor.assess(root)
            self.assertIn("INSERT_ROW_INCOMPLETE", [item["code"] for item in report["issues"]])
            self.assertFalse(report["buildable"])

            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["tools"][0]["function"]["parameters"].update(
                properties={"request_id": {"type": "string"}, "item": {"type": "string"}},
                required=["item"],
            )
            task["tool_implementations"][0]["values"]["request_id"] = "request_id"
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            with patch.object(assessor, "score_file", return_value=quality):
                optional_report = assessor.assess(root)
            self.assertIn("MUTATION_ARGUMENT_OPTIONAL", [
                item["code"] for item in optional_report["issues"]
            ])

    def test_prompt_examples_are_written_without_running_shell_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sandbox"
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts/develop_sandbox_with_agent.sh"),
                 "--input", str(ROOT / "examples/clothing_materials_task.json"),
                 "--output", str(output), "--foreground", "--skip-auto-score"],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            )
            # This legacy example fails the buildability gate after prompt
            # creation. Prompt rendering itself must not execute its examples.
            self.assertEqual(completed.returncode, 4)
            prompt = (output / "TASK_PROMPT.md").read_text(encoding="utf-8")
            self.assertIn("`python3 -m pytest -q`", prompt)
            self.assertIn('"$PYTHON"', prompt)
            self.assertLess(len(prompt.encode("utf-8")), 6000)
            self.assertIn("The following files are platform-owned", prompt)
            self.assertIn("Do not add task-local tests", prompt)
            self.assertNotIn("Implement Trainer Bearer authentication", prompt)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["failed_phase"], "buildability")
            self.assertEqual(
                [item["phase"] for item in status["phase_timings"]],
                ["preparing", "buildability"],
            )
            self.assertTrue(all(item["seconds"] is not None for item in status["phase_timings"]))
            self.assertGreaterEqual(status["elapsed_seconds"], 0)

    def test_task_preparation_failure_closes_timing_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            task.write_text(json.dumps({"task": "bad", "metrics": None}), encoding="utf-8")
            output = root / "sandbox"
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts/develop_sandbox_with_agent.sh"),
                 "--input", str(task), "--output", str(output),
                 "--foreground", "--skip-auto-score"],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(completed.returncode, 0)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["failed_phase"], "preparing")
            self.assertEqual(status["phase_timings"][0]["phase"], "preparing")
            self.assertIsNotNone(status["phase_timings"][0]["seconds"])

    def test_training_categories_have_complete_task_and_sandbox_blueprints(self):
        profiles = set()
        for category in TRAINING_CATEGORIES:
            contract = training_contract(category)
            self.assertEqual(contract["category"], category)
            self.assertTrue(contract["allowed_environment_modes"])
            self.assertIn("goal_success", contract["required_scenarios"])
            self.assertIn("goal_failure", contract["required_scenarios"])
            self.assertTrue(contract["allowed_intents"])
            profiles.add(contract["sandbox_profile"])
        self.assertEqual(len(profiles), len(TRAINING_CATEGORIES))
        self.assertEqual(training_contract("direct_response")["business_tools"], {"min": 0, "max": 0})
        self.assertTrue(training_contract("multi_step_agentic")["dependency"]["required"])

    def test_scaffold_materializes_declared_sandbox_profile(self):
        scaffold = load_script("generate_sandbox_scaffold.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "tools": [],
                "training_category": "direct_response",
                "training_contract": training_contract("direct_response"),
            }
            (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
            scaffold.generate(root)
            profile = json.loads((root / "sandbox_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["sandbox_profile"], "direct_response")
            self.assertIn("test_sandbox_profile_matches_training_contract", (root / "tests/test_scaffold_contract.py").read_text(encoding="utf-8"))
            self.assertIn(
                '"task_input": self.contract.get("public_input", {})',
                (root / "task_impl.py").read_text(encoding="utf-8"),
            )
            self.assertTrue((root / "acceptance.sh").stat().st_mode & 0o111)
            self.assertTrue((root / "acceptance_runner.py").is_file())
            self.assertTrue((root / "IMPLEMENTATION_REPORT.md").is_file())
            self.assertEqual(
                (root / ".dockerignore").read_text(), DOCKERIGNORE_SOURCE
            )
            docker_run = (root / "docker_run.sh").read_text(encoding="utf-8")
            self.assertIn("--read-only", docker_run)
            self.assertIn("--cap-drop ALL", docker_run)
            self.assertIn("no-new-privileges:true", docker_run)
            self.assertIn("--pids-limit", docker_run)
            self.assertIn(
                "/app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700",
                docker_run,
            )

    def test_container_smoke_starts_service_with_writable_non_root_runtime(self):
        builder = (ROOT / "scripts/build_docker_sandbox_image.sh").read_text(
            encoding="utf-8"
        )
        loop = (ROOT / "scripts/loop_experiment.py").read_text(encoding="utf-8")
        runtime_mount = (
            "/app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700"
        )
        self.assertIn(runtime_mount, builder)
        self.assertIn(runtime_mount, loop)
        self.assertIn('docker exec "$smoke_cid"', builder)
        self.assertIn("service_health", builder)

    def test_sandbox_score_uses_ten_point_critical_gate_rubric(self):
        scorer = load_script("score_sandbox.py")
        checks = [
            scorer.Check("delivery", 4.0, True, "ok", True),
            scorer.Check("mutation", 3.0, False, "survived", True),
            scorer.Check("readiness", 3.0, True, "ok", True),
        ]
        report = scorer.score_checks(checks, threshold=8)
        self.assertEqual(report["score"], 0.0)
        self.assertEqual(report["raw_score"], 7.0)
        self.assertEqual(report["score_kind"], "gated_weighted_10_point")
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_critical_gates"], ["mutation"])

        high_raw = [
            scorer.Check("business", 9.0, True, "ok", True),
            scorer.Check("reward", 1.0, False, "wrong", True),
        ]
        self.assertEqual(scorer.score_checks(high_raw)["score"], 0.0)

        complete = [scorer.Check("all", 10.0, True, "ok", True)]
        self.assertEqual(scorer.score_checks(complete)["score"], 10.0)
        self.assertEqual(scorer.score_checks(
            complete, quality_factors={"all": 0.91},
        )["score"], 9.1)

    def test_build_finalization_can_score_pending_artifacts_without_marking_success(self):
        scorer = load_script("score_sandbox.py")
        pending = {"status": "pending", "success": False, "phase": "offline_scoring"}
        self.assertFalse(scorer.delivery_status_ok(pending))
        self.assertTrue(scorer.delivery_status_ok(pending, build_finalization=True))
        self.assertFalse(scorer.delivery_status_ok(
            {**pending, "phase": "acceptance"}, build_finalization=True,
        ))
        self.assertFalse(scorer.delivery_status_ok(
            {"status": "failed", "success": False, "phase": "offline_scoring"},
            build_finalization=True,
        ))
        self.assertTrue(scorer.delivery_status_ok({"status": "success", "success": True}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in scorer.REQUIRED_FILES:
                (root / name).write_text("fixture\n", encoding="utf-8")
            (root / "status.json").write_text(json.dumps(pending), encoding="utf-8")
            with patch.object(scorer, "contract_check", return_value=(True, "ok")), \
                 patch.object(scorer, "review_check", return_value=(True, "ok", 0.9)), \
                 patch.object(scorer, "evidence_fingerprint", return_value="fixture"):
                ordinary = scorer.evaluate(root, project=ROOT, execute=False, threshold=8)
                finalizing = scorer.evaluate(
                    root, project=ROOT, execute=False, threshold=8,
                    build_finalization=True,
                )
            self.assertFalse(ordinary["checks"][0]["passed"])
            self.assertTrue(finalizing["checks"][0]["passed"])
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn('"$output_path" --project "$project_dir" --build-finalization', workflow)

    def test_failed_scoring_preflight_skips_expensive_execution(self):
        scorer = load_script("score_sandbox.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            with patch.object(scorer, "contract_check", return_value=(False, "invalid task")), \
                 patch.object(scorer, "review_check", return_value=(False, "missing", 0.0)), \
                 patch.object(scorer, "run", side_effect=AssertionError("must not execute")):
                report = scorer.evaluate(root, project=ROOT, execute=True, threshold=8)
            self.assertEqual(report["score"], 0.0)
            self.assertEqual(len(report["checks"]), len(SCORE_RUBRIC))
            self.assertIn("contract_and_tool_identity", report["skipped_checks"])

    def test_standard_sandbox_score_cli_executes_checks_by_default(self):
        scorer = load_script("score_sandbox.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "score.json"
            with patch.object(scorer, "evaluate", return_value={"passed": True}) as evaluate, \
                 patch("sys.argv", ["score_sandbox.py", str(root), "--output", str(output)]):
                self.assertEqual(scorer.main(), 0)
            self.assertTrue(evaluate.call_args.kwargs["execute"])
            with patch.object(scorer, "evaluate", return_value={"passed": True}) as evaluate, \
                 patch("sys.argv", ["score_sandbox.py", str(root), "--output", str(output),
                                    "--reuse-evidence"]):
                self.assertEqual(scorer.main(), 0)
            self.assertFalse(evaluate.call_args.kwargs["execute"])

    def test_sandbox_contract_gate_rechecks_current_task_quality(self):
        scorer = load_script("score_sandbox.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"tools": []}
            for name, value in (
                ("task.json", task), ("BUILD_CONTRACT.json", task),
                ("tools.json", []),
            ):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            passed, evidence = scorer.contract_check(root, threshold=8)
            self.assertFalse(passed)
            self.assertIn("task quality gate failed", evidence)
            with patch.object(scorer, "score_file", return_value=SimpleNamespace(
                score=9.0, eligible=False, passed=False, eligibility_failures=("bad reward",),
            )):
                passed, evidence = scorer.contract_check(root, threshold=8)
            self.assertFalse(passed)
            self.assertIn("bad reward", evidence)
            with patch.object(scorer, "score_file", return_value=SimpleNamespace(
                score=9.0, eligible=True, passed=True, eligibility_failures=(),
            )) as score:
                passed, evidence = scorer.contract_check(root, threshold=8)
            self.assertTrue(passed, evidence)
            score.assert_called_once_with(root / "task.json", min_score=8)

    def test_semantic_review_is_bound_to_current_sandbox_sources(self):
        scorer = load_script("score_sandbox.py")
        from env_factory.sandbox_scoring import review_source_hashes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task_impl.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "extra_business_logic.py").write_text("LIMIT = 10\n", encoding="utf-8")
            (root / "data").mkdir()
            (root / "data" / "rows.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
            report = {
                "status": "pass", "score": 0.98, "findings": [],
                "review_run_id": "review-1", "reviewed_at": "2026-09-29T00:00:00Z",
                "checked_modules": ["business_tools", "reward", "user_simulator", "runtime_contract"],
                "required_repairs": [],
                "source_hashes": review_source_hashes(root),
            }
            (root / "review_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(scorer.review_check(root)[0])
            report["score"] = 0.79
            (root / "review_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(scorer.review_check(root)[0])
            report["score"] = 0.98
            del report["checked_modules"]
            (root / "review_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(scorer.review_check(root)[0])
            report["checked_modules"] = ["business_tools", "reward", "user_simulator", "runtime_contract"]
            (root / "review_report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "task_impl.py").write_text("VALUE = 2\n", encoding="utf-8")
            passed, evidence, _ = scorer.review_check(root)
            self.assertFalse(passed)
            self.assertIn("source hashes differ", evidence)
            (root / "task_impl.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "data" / "rows.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
            self.assertFalse(scorer.review_check(root)[0])
            (root / "data" / "rows.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
            (root / "sandbox_score.json").write_text('{"score": 10}\n', encoding="utf-8")
            self.assertTrue(scorer.review_check(root)[0])
            (root / "extra_business_logic.py").write_text("LIMIT = 11\n", encoding="utf-8")
            self.assertFalse(scorer.review_check(root)[0])

    def test_sandbox_loop_is_pinned_to_high_value_tasks_and_luna(self):
        loop = load_script("run_sandbox_build_loop.py")
        command = loop.build_command(ROOT, 45, ROOT / "output/example", 3)
        self.assertEqual(loop.DEFAULT_TASK_IDS, (45, 78, 92, 175))
        self.assertIn("gpt-6-luna", command)
        self.assertIn(str(ROOT / "output/task/task-45/task.json"), command)

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            legacy = project / "output/task_artifacts/task-45/task.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("{}", encoding="utf-8")
            self.assertIn(str(legacy), loop.build_command(project, 45, project / "sandbox", 3))
            current = project / "output/task/task-45/task.json"
            current.parent.mkdir(parents=True)
            current.write_text("{}", encoding="utf-8")
            self.assertIn(str(current), loop.build_command(project, 45, project / "sandbox", 3))
        history = {"rounds": [{"round": 3, "tasks": [{
            "task_id": "task-78", "score": {
                "score": 10.0, "passed": True, "model": "gpt-5.6-luna",
                "review_model": "gpt-5.6-luna",
            }
        }]}]}
        self.assertIsNone(loop.reusable_result(history, 78))  # stale/unexecuted evidence cannot be reused

        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "task-92"
            seed.mkdir()
            (seed / "task_impl.py").write_text("# implementation\n", encoding="utf-8")
            seed_history = {"rounds": [{"round": 6, "tasks": [{
                "task_id": "task-92", "output": str(seed), "score": {"score": 4.5}
            }]}]}
            self.assertEqual(loop.latest_seed(seed_history, 92), seed)
            resumed = loop.build_command(ROOT, 92, ROOT / "output/example", 3, resume=True)
            self.assertIn("--resume", resumed)

    def test_build_workflow_always_captures_terminal_status(self):
        workflow = (ROOT / "scripts" / "develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn('export PYTHONPATH="$project_dir/src', workflow)
        self.assertIn('export PATH="$project_dir/.venv/bin:$PATH"', workflow)
        self.assertIn("if run_agent_and_finalize_impl; then", workflow)
        self.assertIn('write_status "failed" "$exit_code"', workflow)
        self.assertNotIn("set +e\n  run_agent_and_finalize_impl", workflow)
        self.assertIn(
            "validate_agentic_training_value || return $?\n  # Mutation runs execute acceptance.sh",
            workflow,
        )
        self.assertIn("restore_final_acceptance_evidence || return $?\n  validate_dockerfile_security", workflow)
        self.assertIn("from env_factory.tasks.task_portability import prepare_sandbox_task", workflow)
        self.assertIn("prepare_sandbox_task(", workflow)

    def test_training_readiness_enables_deterministic_evaluator_mock(self):
        validator = (ROOT / "scripts" / "sandbox" / "validate_training_readiness.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("SANDBOX_EVALUATOR_MOCK", "1")', validator)

    def test_runtime_validator_distinguishes_template_literals_from_authored_scores(self):
        generator = load_script("generate_sandbox_scaffold.py")
        validator = load_script("validate_sandbox_runtime.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BUILD_CONTRACT.json").write_text(json.dumps({
                "task_spec": {"id": "test"},
                "metrics": [{"id": "event"}], "tools": [],
            }))
            generator.generate(root)
            for name in ("sandbox_runtime.py", "runtime_llm.py"):
                (root / name).write_bytes((ROOT / "src" / "env_factory" / name).read_bytes())
            with patch.object(sys, "argv", ["validator", "--root", str(root)]):
                self.assertEqual(validator.main(), 0)
                implementation = root / "task_impl.py"
                implementation.write_text(
                    implementation.read_text() + '\ndef authored_scores():\n    return {"event": 1.0}\n'
                )
                with self.assertRaisesRegex(SystemExit, "hard-codes generated metric id: event"):
                    validator.main()

    def test_metric_labels_in_business_data_are_not_scoring_overrides(self):
        validator = load_script("validate_sandbox_runtime.py")
        source = 'AUDIT_LABEL = "event"\ndef handler():\n    return {"event": 1.0}\n'
        self.assertEqual(validator.hardcoded_metric_scores(source, {"event"}), [])
        for source in (
            'def reward():\n    return {"event": 1.0}\n',
            'def scores(result):\n    result["event"] = 1.0\n',
            'def evaluate(metric):\n    return metric == "event"\n',
        ):
            self.assertEqual(validator.hardcoded_metric_scores(source, {"event"}), ["event"])

    def test_runtime_validator_rejects_trace_derived_tool_preconditions(self):
        validator = load_script("validate_sandbox_runtime.py")
        unsafe = '''
def dependency_error():
    return SandboxError("PRECONDITION_FAILED", "read first", 400)
def successful_calls(store):
    return store.replay()["events"]
def handler(store, arguments):
    if not successful_calls(store):
        raise dependency_error()
    return arguments
'''
        self.assertEqual(
            validator.trace_precondition_functions(unsafe), ["handler"]
        )
        safe = '''
def observe(store):
    return store.replay()
def handler(data, arguments):
    if not data.exists(arguments["id"]):
        raise SandboxError("PRECONDITION_FAILED", "missing business row", 400)
    return data.read(arguments["id"])
'''
        self.assertEqual(validator.trace_precondition_functions(safe), [])

    def test_agentic_value_validator_rejects_placeholders_and_empty_collections(self):
        validator = load_script("validate_agentic_training_value.py")
        for invalid in (float("nan"), float("inf"), -float("inf"), 1.1, -1.1, 10**1000, True):
            self.assertIsNone(validator.reward_of({"history": [{
                "operation": "reward", "body": {"reward": invalid},
            }]}))
        self.assertEqual(validator.reward_of({"history": [{
            "operation": "reward", "body": {"reward": -1.0},
        }]}), -1.0)
        readiness = load_script("validate_training_readiness.py")
        self.assertIsNone(readiness.reward_from_run({"history": [{
            "operation": "reward", "body": {"reward": float("nan")},
        }]}))
        semantic = {"metrics": [{"category": "outcome", "evaluation_inputs": ["final_agent_response"],
                                 "evaluator": {"kind": "external_llm_judge"}}]}
        self.assertTrue(validator.semantic_terminal_requires_live(semantic))
        semantic["metrics"][0]["evaluator"]["kind"] = "numeric_targets"
        self.assertFalse(validator.semantic_terminal_requires_live(semantic))
        step = {"tool_name": "lookup", "arguments": {"selector": {"$ref": "row_id"}, "date": "2026-01-01"}}
        tools = [{"function": {"name": "lookup", "parameters": {
            "required": ["selector", "date"],
        }}}]
        self.assertEqual(validator.argument_probe_keys(step, tools), ["selector", "date"])
        self.assertTrue(validator.has_placeholder({"source": "fixture-value"}))
        self.assertTrue(validator.empty_critical_collection({"categories": []}))
        self.assertFalse(validator.empty_critical_collection({"categories": ["glass"]}))
        self.assertTrue(validator.meaningful_result({"records": [{"id": 1}], "count": 1}))
        self.assertFalse(validator.meaningful_result({"records": [], "count": 0}))
        replay = {"events": [
            {"event": "tool_call", "payload": {"tool_name": "lookup", "noise": False,
                                              "business_data_reads": ["inventory"]}},
            {"event": "tool_call", "payload": {"tool_name": "noise", "noise": True,
                                              "business_data_reads": ["orders"]}},
        ]}
        self.assertEqual(validator.business_data_reads_from_replay(
            replay, {"inventory", "orders"},
        ), {"inventory"})
        self.assertEqual(validator.business_tool_reads_from_replay(
            replay, {"inventory", "orders"},
        ), [{"tool_name": "lookup", "tables_read": ["inventory"]}])

    def test_reference_data_causality_rejects_read_then_constant_result(self):
        validator = load_script("validate_agentic_training_value.py")
        self.assertTrue(validator._answer_mentions_value("库存共计 120 件", 120))
        self.assertFalse(validator._answer_mentions_value("2026-01-20 库存", 20))

        class ProbeApp:
            def __init__(self, frozen):
                self.frozen = frozen
                self.rows = [{"id": "item-1", "value": "actual"}]
                self.events = []

            def business_snapshot(self):
                return {"inventory": [dict(row) for row in self.rows]}

            def mutate_business_state(self, mutation):
                for row in self.rows:
                    if row["id"] == mutation["selector"]["id"]:
                        row.update(mutation["changes"])

            def handle(self, method, path, body=None, headers=None):
                if path == "/v1/reset":
                    self.rows = [{"id": "item-1", "value": "actual"}]
                    self.events = []
                    return 200, {"episode_id": "probe"}, {}
                if path == "/v1/tools/lookup":
                    self.events.append({"event": "tool_call", "payload": {
                        "tool_name": "lookup", "business_data_reads": ["inventory"],
                    }})
                    value = "actual" if self.frozen else self.rows[0]["value"]
                    return 200, {"value": value}, {}
                if path == "/v1/replay":
                    return 200, {"events": self.events}, {}
                raise AssertionError(path)

        scenario = {"steps": [
            {"operation": "reset", "body": {"episode_id": "probe"}},
            {"operation": "tool_call", "tool_name": "lookup", "arguments": {}},
        ]}
        manifest = {"tables": [{"table_name": "inventory", "primary_key": ["id"]}]}
        self.assertFalse(validator.probe_reference_data_causality(
            ProbeApp(True), {}, scenario, manifest,
        )["proved"])
        self.assertTrue(validator.probe_reference_data_causality(
            ProbeApp(False), {}, scenario, manifest,
        )["proved"])

    def test_causality_prefers_field_stated_in_answer(self):
        validator = load_script("validate_agentic_training_value.py")

        class ProbeApp:
            def __init__(self):
                self.rows = [{"id": "item-1", "quantity": 12, "status": "需要复核"}]
                self.events = []

            def business_snapshot(self):
                return {"inventory": [dict(row) for row in self.rows]}

            def mutate_business_state(self, mutation):
                self.rows[0].update(mutation["changes"])

            def handle(self, method, path, body=None, headers=None):
                if path == "/v1/reset":
                    self.rows = [{"id": "item-1", "quantity": 12, "status": "需要复核"}]
                    self.events = []
                    return 200, {"episode_id": "probe"}, {}
                if path == "/v1/tools/lookup":
                    self.events.append({"event": "tool_call", "payload": {
                        "tool_name": "lookup", "business_data_reads": ["inventory"],
                    }})
                    return 200, dict(self.rows[0]), {}
                if path == "/v1/replay":
                    return 200, {"events": self.events}, {}
                raise AssertionError(path)

        scenario = {"steps": [
            {"operation": "reset", "body": {"episode_id": "probe"}},
            {"operation": "tool_call", "tool_name": "lookup", "arguments": {}},
            {"operation": "agent_response", "content": "item-1 的状态是需要复核。"},
        ]}
        manifest = {"tables": [{"table_name": "inventory", "primary_key": ["id"]}]}
        result = validator.probe_reference_data_causality(ProbeApp(), {}, scenario, manifest)
        self.assertTrue(result["proved"])
        self.assertEqual(result["field"], "status")
        self.assertTrue(result["answer_value_mentioned"])

    def test_numeric_answer_counterfactual_changes_calculation_result(self):
        from env_factory.contracts.reward_contract import numeric_answer_counterfactual
        answer = '{"total": 42, "tax": 3.5}'
        altered = numeric_answer_counterfactual({"task_intent": "calculate"}, answer)
        self.assertEqual(altered, '{"total": 42, "tax": 4.5}')
        answer = "2025-04-01 计划200份，库存0.8公斤；结论：采购金额252元。"
        self.assertEqual(
            numeric_answer_counterfactual({"task_intent": "calculate"}, answer),
            "2025-04-01 计划200份，库存0.8公斤；结论：采购金额253元。",
        )
        answer = "结论：总金额252元；核算日期2025-04-01。"
        self.assertEqual(
            numeric_answer_counterfactual({"task_intent": "calculate"}, answer),
            "结论：总金额253元；核算日期2025-04-01。",
        )
        answer = (
            "兽皮总张数为440张。明细：120 + 85 + 95 + 140 = 440。"
            "未计入记录60张；该440张可安排生产。"
        )
        task = {"task_intent": "calculate", "metric_implementations": [{
            "operator": "numeric_targets", "expected": {"targets": [{"label": "总张数"}]},
        }]}
        self.assertEqual(
            numeric_answer_counterfactual(task, answer),
            answer.replace("总张数为440", "总张数为441", 1),
        )
        self.assertIsNone(numeric_answer_counterfactual({"task_intent": "explain"}, answer))

    def test_private_probe_reaches_late_tool_and_combines_partial_outcomes(self):
        validator = load_script("validate_agentic_training_value.py")

        class ProbeApp:
            def __init__(self, frozen=False):
                self.frozen = frozen
                self.reset()

            def reset(self):
                self.tables = {
                    "index": [{"id": 1, **{f"note_{i}": f"note-{i}" for i in range(30)}}],
                    "details": [{"id": 2, **{f"note_{i}": f"detail-{i}" for i in range(30)}}],
                    "facts": [{"id": 3, "left": "A", "right": "B"}],
                }
                self.events = []

            def business_snapshot(self):
                return json.loads(json.dumps(self.tables))

            def mutate_business_state(self, mutation):
                self.tables[mutation["table"]][0].update(mutation["changes"])

            def handle(self, method, path, body=None, headers=None):
                if path == "/v1/reset":
                    self.reset()
                    return 200, {}, {}
                if path.startswith("/v1/tools/"):
                    name = path.rsplit("/", 1)[-1]
                    self.events.append({"event": "tool_call", "payload": {
                        "tool_name": name, "business_data_reads": [name]}})
                    return 200, {"records": self.business_snapshot()[name]}, {}
                if path == "/v1/replay":
                    return 200, {"events": self.events}, {}
                if path == "/v1/agent_response":
                    return 200, {}, {}
                if path == "/v1/reward":
                    facts = self.tables["facts"][0]
                    reward = 1.0 if self.frozen else (
                        .2 + .4 * (facts["left"] == "A") + .4 * (facts["right"] == "B"))
                    return 200, {"reward": reward}, {}
                raise AssertionError(path)

        scenario = {"steps": [{"operation": "reset"}, *[
            {"operation": "tool_call", "tool_name": name, "arguments": {}}
            for name in ("index", "details", "facts")],
            {"operation": "agent_response", "content": "A B"}, {"operation": "reward"}]}
        manifest = {"tables": [{"table_name": name, "primary_key": ["id"]}
                               for name in ("index", "details", "facts")]}
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                app = ProbeApp(frozen)
                causality = validator.probe_reference_data_causality(app, {}, scenario, manifest)
                mutations = causality["_fallback_mutations"]
                self.assertTrue(any(item["table"] == "facts" for item in mutations))
                result = validator.probe_reference_data_reward_candidates(app, {}, scenario, mutations, .8)
                self.assertEqual(result["proved"], not frozen)
                if not frozen:
                    self.assertTrue(all(abs(value - .6) < 1e-9 for value in result["changed_rewards"]))
                    self.assertEqual(result["required_drop"], .4)

    def test_private_data_reward_probe_rejects_stale_answer_credit(self):
        validator = load_script("validate_agentic_training_value.py")

        class ProbeApp:
            def __init__(self, sensitive):
                self.sensitive = sensitive
                self.amount = 2
                self.changed_reward_calls = 0
                self.baseline_reward_calls = 0
                self.tool_read_calls = 0

            def business_snapshot(self):
                return {"inventory": [{"id": 1, "amount": self.amount}]}

            def mutate_business_state(self, mutation):
                amount = mutation["changes"]["amount"]
                if self.sensitive == "positive_threshold" and amount <= 0:
                    raise ValueError("amount must be positive")
                self.amount = amount

            def handle(self, method, path, body=None, headers=None):
                if path == "/v1/reset":
                    self.amount = 2
                    return 200, {"episode_id": "probe"}, {}
                if path == "/v1/tools/read_inventory":
                    self.tool_read_calls += 1
                    result = {"amount": self.amount}
                    if self.sensitive == "flaky_tool":
                        result["read_sequence"] = self.tool_read_calls
                    return 200, result, {}
                if path == "/v1/agent_response":
                    return 200, {"accepted": True}, {}
                if path == "/v1/reward":
                    if self.sensitive == "flaky_baseline" and self.amount == 2:
                        self.baseline_reward_calls += 1
                        return 200, {"reward": float(self.baseline_reward_calls % 2 == 1)}, {}
                    if self.sensitive == "flaky" and self.amount != 2:
                        self.changed_reward_calls += 1
                        return 200, {"reward": float(self.changed_reward_calls % 2 == 0)}, {}
                    reward = 0.0 if (
                        self.sensitive == "threshold" and self.amount == 0
                        or self.sensitive == "positive_threshold" and self.amount == 1
                        or self.sensitive is True and self.amount != 2
                    ) else 1.0
                    return 200, {"reward": reward}, {}
                raise AssertionError(path)

        scenario = {"steps": [
            {"operation": "reset", "body": {"episode_id": "probe"}},
            {"operation": "tool_call", "tool_name": "read_inventory", "arguments": {}},
            {"operation": "agent_response", "content": "库存为 2"},
            {"operation": "reward"},
        ]}
        mutation = {"table": "inventory", "selector": {"id": 1},
                    "changes": {"amount": 3}}
        stale = validator.probe_reference_data_reward_sensitivity(
            ProbeApp(False), {}, scenario, mutation, 0.8,
        )
        self.assertFalse(stale["proved"])
        self.assertTrue(stale["tool_results_changed"])
        sensitive = validator.probe_reference_data_reward_sensitivity(
            ProbeApp(True), {}, scenario, mutation, 0.8,
        )
        self.assertTrue(sensitive["proved"])
        self.assertEqual((sensitive["baseline_reward"], sensitive["changed_reward"]),
                         (1.0, 0.0))
        self.assertEqual(sensitive["changed_rewards"], [0.0, 0.0])
        self.assertEqual(sensitive["baseline_rewards"], [1.0, 1.0])
        self.assertTrue(sensitive["baseline_tool_results_stable"])
        derived = validator.probe_reference_data_reward_sensitivity(
            ProbeApp("threshold"), {}, scenario, mutation, 0.8,
        )
        self.assertTrue(derived["proved"])
        self.assertEqual(derived["changed_reward"], 0.0)
        self.assertEqual(derived["changed_rewards"], [0.0, 0.0])
        bounded = validator.probe_reference_data_reward_sensitivity(
            ProbeApp("positive_threshold"), {}, scenario, mutation, 0.8,
        )
        self.assertTrue(bounded["proved"])
        self.assertEqual(bounded["changed_rewards"], [0.0, 0.0])
        flaky = validator.probe_reference_data_reward_sensitivity(
            ProbeApp("flaky"), {}, scenario, mutation, 0.8,
        )
        self.assertFalse(flaky["proved"])
        self.assertEqual(flaky["changed_reward"], 1.0)
        unstable_baseline = validator.probe_reference_data_reward_sensitivity(
            ProbeApp("flaky_baseline"), {}, scenario, mutation, 0.8,
        )
        self.assertFalse(unstable_baseline["proved"])
        self.assertEqual(unstable_baseline["baseline_rewards"], [1.0, 0.0])
        unstable_tool = validator.probe_reference_data_reward_sensitivity(
            ProbeApp("flaky_tool"), {}, scenario, mutation, 0.8,
        )
        self.assertFalse(unstable_tool["proved"])
        self.assertFalse(unstable_tool["baseline_tool_results_stable"])

    def test_agentic_value_reorders_a_real_dependency_not_parallel_producers(self):
        validator = load_script("validate_agentic_training_value.py")
        steps = [
            {"operation": "tool_call", "tool_name": "read_features"},
            {"operation": "tool_call", "tool_name": "read_rules"},
            {"operation": "tool_call", "tool_name": "decide"},
        ]
        dag = {"edges": [
            {"from_tool": "read_features", "to_tool": "decide"},
            {"from_tool": "read_rules", "to_tool": "decide"},
        ]}
        self.assertEqual(validator.dependency_swap_positions(steps, dag), (0, 2))

    def test_offline_scorer_requires_executable_business_semantics(self):
        scorer = load_script("score_sandbox_offline.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "tools": [{"category": "business", "function": {
                    "name": "lookup", "parameters": {
                        "type": "object", "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }}],
                "noise_tools": [],
                "acceptance_contract": {"executable_scenarios": [
                    {"kind": "goal_success", "steps": [
                        {"operation": "tool_call", "tool_name": "lookup", "arguments": {"id": "x"}},
                        {"operation": "agent_response", "content": "done"},
                        {"operation": "reward"},
                    ]},
                    {"kind": "goal_failure", "steps": [{"operation": "reward"}]},
                ]},
            }
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            evidence = {name: {"passed": True} for name in (
                "contract_and_tool_identity", "business_acceptance", "runtime_genericity",
                "mutation_resistance", "training_readiness", "agentic_training_value",
            )}
            passed, message = scorer.offline_semantic_check(root, evidence)
            self.assertTrue(passed, message)
            task["acceptance_contract"]["executable_scenarios"][0]["steps"][0]["arguments"] = {}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            self.assertFalse(scorer.offline_semantic_check(root, evidence)[0])

            task["tools"][0]["function"]["parameters"]["required"] = []
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            self.assertTrue(scorer.offline_semantic_check(root, evidence)[0])

    def test_offline_scorer_requires_review_and_executable_semantics(self):
        scorer = load_script("score_sandbox_offline.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = [
                scorer.rubric_check(name, True, "review passed")
                for name, _, _ in SCORE_RUBRIC
            ]
            base = {
                "checks": [item._asdict() for item in checks],
                "evidence_fingerprint": "frozen",
                "model": "builder", "review_model": "reviewer",
            }
            with patch.object(scorer, "evaluate", return_value=base), \
                 patch.object(scorer, "offline_semantic_check", return_value=(True, "executed")), \
                 patch.object(scorer, "sandbox_quality_factors", return_value={
                     "semantic_business_fidelity": 0.95,
                     "declared_training_policy": 0.8,
                 }):
                passing = scorer.evaluate_offline(root, project=ROOT, threshold=8)
                self.assertTrue(passing["passed"])
                self.assertEqual(passing["score_scope"], "offline_sandbox_qualification")
                self.assertFalse(passing["delivery_verified"])
                self.assertEqual(scorer.evaluate_offline(
                    root, project=ROOT, threshold=8,
                )["score"], 9.57)
                base["checks"][2]["passed"] = False
                rejected = scorer.evaluate_offline(root, project=ROOT, threshold=8)
                self.assertEqual(rejected["score"], 0.0)
                self.assertIn("semantic_business_fidelity", rejected["failed_critical_gates"])
                base["checks"][2]["passed"] = True
            with patch.object(scorer, "evaluate", return_value=base), \
                 patch.object(scorer, "offline_semantic_check", return_value=(False, "invalid")):
                rejected = scorer.evaluate_offline(root, project=ROOT, threshold=8)
                self.assertEqual(rejected["score"], 0.0)

    def test_outer_mutation_probe_detects_platform_owned_markers(self):
        mutation = load_script("run_mutation_tests.py")
        task = {
            "acceptance_contract": {"argument_probes": [{
                "tool_name": "update_item", "arguments": {"value": "a"}
            }]},
            "tools": [{"function": {"name": "update_item"}}],
        }
        original = mutation.request
        try:
            mutation.request = lambda base, method, path, body=None, key=None: (
                (200, {"mutation": "constant_tool_result"})
                if path.startswith("/v1/tools/") else (200, {})
            )
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "constant_tool_result"))
            task["acceptance_contract"] = {"executable_scenarios": [
                {"steps": [{"operation": "tool_call", "tool_name": "update_item", "arguments": {"bad": True}}]},
                {"kind": "goal_success", "steps": [{"operation": "tool_call", "tool_name": "update_item", "arguments": {}}]},
            ]}
            mutation.request = lambda base, method, path, body=None, key=None: (
                (400, {}) if path.startswith("/v1/tools/") and body else
                (200, {"mutation": "constant_tool_result"}) if path.startswith("/v1/tools/") else (200, {}))
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "constant_tool_result"))
            task["acceptance_contract"] = {"executable_scenarios": [{"steps": [{
                "operation": "tool_call", "tool_name": "update_item", "arguments": {}
            }]}]}
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "constant_tool_result"))
            mutation.request = lambda base, method, path, body=None, key=None: (200, {})
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "bypass_trainer_auth"))
        finally:
            mutation.request = original

    def test_generated_acceptance_failure_requires_independent_witness(self):
        mutation = load_script("run_mutation_tests.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("# runtime placeholder\n", encoding="utf-8")
            (root / "task.json").write_text(json.dumps({
                "requirements": {"runtime_interface": {
                    "mutation_testing": {"modes": ["constant_reward"]},
                }},
            }), encoding="utf-8")
            with patch.object(sys, "argv", ["run_mutation_tests.py", "--root", str(root)]), \
                    patch.object(mutation, "run_acceptance", side_effect=[(0, "ok"), (1, "killed")]) as acceptance, \
                    patch.object(mutation, "free_port", return_value=49152), \
                    patch.object(mutation.subprocess, "Popen") as runtime, \
                    patch.object(mutation, "wait_health", return_value=True), \
                    patch.object(mutation, "run_outer", return_value=(0, "ok")) as outer, \
                    patch.object(mutation, "argument_sensitivity", return_value=False), \
                    patch.object(mutation, "direct_mutation_probe", return_value=False), \
                    patch.object(mutation, "stop"):
                with self.assertRaisesRegex(SystemExit, "surviving mutants"):
                    mutation.main()
            self.assertEqual(acceptance.call_count, 2)
            self.assertEqual(runtime.call_count, 2)
            self.assertEqual(outer.call_count, 2)

    def test_direct_response_omits_only_tool_mutants_and_still_checks_reward_and_auth(self):
        mutation = load_script("run_mutation_tests.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("# runtime placeholder\n")
            (root / "task.json").write_text(json.dumps({
                "training_category": "direct_response", "tools": [],
                "requirements": {"runtime_interface": {"mutation_testing": {"modes": [
                    "constant_tool_result", "ignore_tool_arguments", "skip_business_write",
                    "constant_reward", "bypass_trainer_auth"]}}}}))
            with patch.object(sys, "argv", ["run_mutation_tests.py", "--root", str(root)]), \
                    patch.object(mutation, "run_acceptance", return_value=(0, "ok")), \
                    patch.object(mutation, "free_port", return_value=49152), \
                    patch.object(mutation.subprocess, "Popen") as runtime, \
                    patch.object(mutation, "wait_health", return_value=True), \
                    patch.object(mutation, "run_outer", return_value=(0, "ok")), \
                    patch.object(mutation, "argument_sensitivity", return_value=False), \
                    patch.object(mutation, "direct_mutation_probe", return_value=True) as probe, \
                    patch.object(mutation, "stop"):
                self.assertEqual(mutation.main(), 0)
            self.assertEqual(runtime.call_count, 3)  # baseline plus reward/auth
            self.assertEqual([call.args[-1] for call in probe.call_args_list], ["constant_reward", "bypass_trainer_auth"])

    def test_mutation_reuses_unchanged_outer_baseline_only(self):
        mutation = load_script("run_mutation_tests.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "app.py"
            app.write_text("# validated runtime\n", encoding="utf-8")
            (root / "task.json").write_text(json.dumps({
                "requirements": {"runtime_interface": {
                    "mutation_testing": {"modes": ["constant_reward"]},
                }},
            }), encoding="utf-8")
            (root / "acceptance_result.json").write_text(json.dumps({
                "business_acceptance": "passed", "http_conformance": "skipped",
                "http_skip_reason": "in-process acceptance",
            }), encoding="utf-8")
            digest = docker_build_context_digest(root)
            self.assertTrue(mutation.reusable_baseline(root, digest))
            with patch.object(sys, "argv", [
                "run_mutation_tests.py", "--root", str(root),
                "--baseline-context-digest", digest,
            ]), patch.object(mutation, "run_acceptance", return_value=(1, "killed")) as acceptance, \
                 patch.object(mutation, "free_port", return_value=49152), \
                 patch.object(mutation.subprocess, "Popen"), \
                 patch.object(mutation, "wait_health", return_value=True), \
                 patch.object(mutation, "run_outer", return_value=(0, "ok")), \
                 patch.object(mutation, "argument_sensitivity", return_value=False), \
                 patch.object(mutation, "direct_mutation_probe", return_value=True), \
                 patch.object(mutation, "stop"):
                self.assertEqual(mutation.main(), 0)
            self.assertEqual(acceptance.call_count, 1)
            app.write_text("# changed runtime\n", encoding="utf-8")
            self.assertFalse(mutation.reusable_baseline(root, digest))

    def test_in_process_mutation_probe_handles_runtime_json_logs(self):
        mutation = load_script("run_mutation_tests.py")
        task = {"acceptance_contract": {"argument_probes": [{
            "tool_name": "query_item", "arguments": {"item_id": "A1"},
        }]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text(
                "import os\n"
                "class App:\n"
                "    def handle(self, method, path, body, headers=None):\n"
                "        print('{\"event\":\"request_completed\"}')\n"
                "        if path == '/v1/reset': return 200, {}, {}\n"
                "        return 200, {'mutation': os.getenv('SANDBOX_MUTATION_MODE')}, {}\n"
                "def create_app(db_path=None): return App()\n",
                encoding="utf-8",
            )
            env = dict(__import__("os").environ)
            env["SANDBOX_MUTATION_MODE"] = "constant_tool_result"
            self.assertTrue(mutation.in_process_constant_tool_probe(task, root, env, "key"))
            env["SANDBOX_MUTATION_MODE"] = "disabled"
            self.assertFalse(mutation.in_process_constant_tool_probe(task, root, env, "key"))

    def test_outer_conformance_accepts_compiled_process_metric(self):
        outer = load_script("generate_outer_conformance.py")
        task = {
            "actions": [{"name": "lookup"}],
            "reward_key_steps": [{
                "step_id": "step-1", "action_name": "lookup",
                "rationale": "required lookup", "required_for_goal": True,
            }],
            "metrics": [{
                "id": "process_lookup", "category": "process", "type": "rule-based",
                "target_action": "lookup", "weight": 1.0,
                "evaluator": {"kind": "trajectory_rule", "source": "runtime_rule",
                              "score_mapping": {"pass": 1, "fail": 0}},
            }],
            "reward_formula": {"score_range": [-1, 1]},
        }
        steps, metrics = outer.check_rewards(task)
        self.assertEqual(steps[0]["step_id"], "step-1")
        self.assertEqual(metrics[0]["id"], "process_lookup")

    def test_outer_conformance_accepts_typed_dynamic_object_fields(self):
        outer = load_script("generate_outer_conformance.py")
        outer.check_schema({
            "type": "object", "description": "Measurements by name",
            "additionalProperties": {"type": "number"},
        }, "measurements")
        with self.assertRaises(SystemExit):
            outer.check_schema({
                "type": "object", "description": "Measurements by name",
                "additionalProperties": "number",
            }, "measurements")

    def test_sandbox_builder_help_exposes_model_selection(self):
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts" / "develop_sandbox_with_agent.sh"), "--help"],
            text=True, capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--model NAME", completed.stdout)
        self.assertIn("--review-model NAME", completed.stdout)
        self.assertIn("--skip-auto-score", completed.stdout)

    def test_successful_build_triggers_offline_scoring(self):
        workflow = (ROOT / "scripts" / "develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn('auto_score="true"', workflow)
        self.assertIn('scripts/sandbox/score_sandbox_offline.py', workflow)
        self.assertIn('offline_sandbox_score.json', workflow)
        loop = load_script("run_sandbox_build_loop.py")
        self.assertIn("--skip-auto-score", loop.build_command(ROOT, 45, ROOT / "output/example", 3))

    def test_scaffold_generator_creates_compilable_thin_composition(self):
        generator = load_script("generate_sandbox_scaffold.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = [{"type": "function", "function": {"name": "lookup"}}]
            (root / "BUILD_CONTRACT.json").write_text(
                json.dumps({"tools": tools}), encoding="utf-8"
            )
            generator.generate(root)
            app = (root / "app.py").read_text(encoding="utf-8")
            implementation = (root / "task_impl.py").read_text(encoding="utf-8")
            compile(app, "app.py", "exec")
            compile(implementation, "task_impl.py", "exec")
            self.assertIn("SandboxApplication", app)
            self.assertIn("class TaskHooks", implementation)
            self.assertEqual(json.loads((root / "tools.json").read_text()), tools)
            self.assertIn("USER sandbox", (root / "Dockerfile").read_text())
            requirements = (root / "requirements-dev.txt").read_text().splitlines()
            self.assertIn("pytest==9.1.1", requirements)
            self.assertTrue(all("==" in item for item in requirements))
            self.assertTrue((root / "docker_build.sh").stat().st_mode & 0o111)
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"], cwd=root,
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
    def test_readiness_helpers_preserve_business_fields_and_normalize_replay_envelopes(self):
        readiness = load_script("validate_training_readiness.py")
        value = {"request_id": "business-key", "nested": {
            "timestamp": 1, "created_at": 2, "value": 2,
        }}
        self.assertEqual(readiness.canonical(value), value)
        replay = {"episode_id": "episode", "trace_hash": "digest", "events": [{
            "event": "tool_call", "timestamp": 3,
            "payload": {"tool_call_id": "random", "timestamp": 4,
                        "arguments": {"timestamp": 5}},
            "result": {"created_at": 6, "timestamp": 7},
        }, {"event": "business_update", "timestamp": 8,
            "payload": {"timestamp": 9}}]}
        normalized = readiness.canonical(replay)
        self.assertNotIn("trace_hash", normalized)
        self.assertNotIn("timestamp", normalized["events"][0])
        self.assertEqual(normalized["events"][0]["payload"]["arguments"]["timestamp"], 5)
        self.assertEqual(normalized["events"][0]["result"], replay["events"][0]["result"])
        self.assertEqual(normalized["events"][1]["payload"]["timestamp"], 9)
        changed = json.loads(json.dumps(replay))
        changed["events"][0]["result"]["created_at"] = 8
        self.assertNotEqual(readiness.canonical(replay), readiness.canonical(changed))
        self.assertEqual(
            readiness.forbidden_paths({"public": {"ground_truth": "secret"}}),
            ["$.public.ground_truth"],
        )

    def test_development_plan_is_deterministic_and_valid(self):
        generator = load_script("generate_development_plan.py")
        validator = load_script("validate_development_plan.py")
        contract = {
            "tools": [{"function": {"name": "lookup"}}],
            "metrics": [{"id": "success"}],
            "artifacts": {"data_manifest": {"tables": [{"table_name": "items"}]}},
        }
        first = generator.build_plan(contract)
        second = generator.build_plan(contract)
        self.assertEqual(first, second)
        self.assertEqual(validator.validate(first), [])
        self.assertEqual(first["authority"], "env_factory_outer_workflow")
        self.assertEqual(first["version"], "2.0")
        self.assertEqual(
            [node["id"] for node in first["nodes"]],
            ["task_handlers", "metric_extensions"],
        )
        self.assertEqual(first["nodes"][0]["scope"]["tables"], ["items"])

    def test_model_judge_is_not_a_custom_development_node(self):
        generator = load_script("generate_development_plan.py")
        plan = generator.build_plan({"tools": [], "metrics": [
            {"id": "semantic_quality", "evaluator": {"kind": "external_llm_judge"}},
            {"id": "hybrid_quality", "evaluator": {"kind": "hybrid_outcome"}},
        ]})
        self.assertNotIn("metric_extensions", [node["id"] for node in plan["nodes"]])
        self.assertEqual(plan["nodes"], [])
        self.assertEqual(load_script("validate_development_plan.py").validate(plan), [])

    def test_builder_supports_a_fully_declarative_zero_node_plan(self):
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn('node_ids=("")', workflow)
        self.assertIn('[[ -z "$node_id" ]] && continue', workflow)

    def test_builder_restores_platform_assets_changed_by_task_agent(self):
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn("platform_owned_files=(", workflow)
        self.assertIn("sandbox_runtime.py", workflow)
        self.assertIn("restore_and_reject_platform_changes", workflow)
        self.assertIn("Code Agent 修改了平台资产，已恢复并拒绝本轮", workflow)

    def test_semantic_reviewer_cannot_read_its_live_transcript(self):
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn("envfactory-review-stderr", workflow)
        self.assertIn('2>"$review_stderr_tmp"', workflow)
        self.assertIn('cp "$review_stderr_tmp" "$review_stderr"', workflow)

    def test_outer_mutation_evidence_is_authoritative_for_semantic_review(self):
        workflow = (ROOT / "scripts/develop_sandbox_with_agent.sh").read_text(encoding="utf-8")
        self.assertIn('mutation_log="$output_path/mutation_report.log"', workflow)
        self.assertIn('"mutation testing: ok" in mutation_report', workflow)
        self.assertIn("acceptance.sh is a\nsingle baseline/probe entry point", workflow)

    def test_runtime_validator_accepts_modular_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"metrics": [{"id": "goal"}]}
            (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "app.py").write_text(
                "from tool_registry import ToolRegistry\nfrom reward_evaluator import RewardEvaluator\n",
                encoding="utf-8",
            )
            (root / "tool_registry.py").write_text(
                "class ToolRegistry:\n    def execute(self):\n        return None\n",
                encoding="utf-8",
            )
            (root / "reward_evaluator.py").write_text(
                "class RewardEvaluator:\n"
                "    def __init__(self, contract):\n"
                "        self.metrics = contract.get('metrics', [])\n"
                "    def evaluate(self, metric):\n"
                "        evaluator = metric['evaluator']\n"
                "        return evaluator\n"
                "from runtime_llm import RuntimeLLMClient\n"
                "def call(client): return client.json_chat([])\n",
                encoding="utf-8",
            )
            (root / "user_simulator.py").write_text(
                "class UserSimulator:\n"
                "    def __init__(self):\n"
                "        self.profiles = []; self.scripts = []\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["python3", str(ROOT / "scripts" / "sandbox" / "validate_sandbox_runtime.py"), "--root", str(root)],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_runtime_validator_allows_support_components_in_shared_runtime(self):
        validator = ROOT / "scripts" / "sandbox" / "validate_sandbox_runtime.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "metrics": [{"id": "goal", "evaluator": {}}],
                "requirements": {"runtime_interface": {"shared_runtime": {
                    "required_components": ["AcceptanceScenarioRunner", "validate_json_schema"]
                }}},
            }
            (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "sandbox_runtime.py").write_text(
                "class AcceptanceScenarioRunner: pass\n"
                "def validate_json_schema(): pass\n",
                encoding="utf-8",
            )
            (root / "app.py").write_text(
                "from sandbox_runtime import ContractToolRegistry, ContractRewardAggregator, SandboxApplication, ContractUserSimulator, DeclarativeMetricEvaluator\n"
                "from runtime_llm import RuntimeLLMClient\n"
                "contract = {}; metrics = contract.get('metrics', [])\n"
                "def call(client): return client.json_chat([])\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(validator), "--root", str(root)],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
