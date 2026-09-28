import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from env_factory.tasks.task_routing import TRAINING_CATEGORIES, training_contract
from env_factory.evidence.material_artifacts import DOCKERIGNORE_SOURCE


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
        self.assertEqual(report["score"], 7.0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_critical_gates"], ["mutation"])

        complete = [scorer.Check("all", 10.0, True, "ok", True)]
        self.assertEqual(scorer.score_checks(complete)["score"], 10.0)

    def test_sandbox_loop_is_pinned_to_high_value_tasks_and_luna(self):
        loop = load_script("run_sandbox_build_loop.py")
        command = loop.build_command(ROOT, 45, ROOT / "output/example", 3)
        self.assertEqual(loop.DEFAULT_TASK_IDS, (45, 78, 92, 175))
        self.assertIn("gpt-6-luna", command)
        self.assertIn(str(ROOT / "output/task_artifacts/task-45/task.json"), command)
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
            "validate_agentic_training_value\n  # Mutation runs execute acceptance.sh",
            workflow,
        )
        self.assertIn("restore_final_acceptance_evidence\n  validate_dockerfile_security", workflow)
        self.assertIn("from env_factory.tasks.task_portability import prepare_sandbox_task", workflow)
        self.assertIn("prepare_sandbox_task(", workflow)

    def test_training_readiness_enables_deterministic_evaluator_mock(self):
        validator = (ROOT / "scripts" / "sandbox" / "validate_training_readiness.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("SANDBOX_EVALUATOR_MOCK", "1")', validator)

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
        self.assertTrue(validator.has_placeholder({"source": "fixture-value"}))
        self.assertTrue(validator.empty_critical_collection({"categories": []}))
        self.assertFalse(validator.empty_critical_collection({"categories": ["glass"]}))
        self.assertTrue(validator.meaningful_result({"records": [{"id": 1}], "count": 1}))
        self.assertFalse(validator.meaningful_result({"records": [], "count": 0}))

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
            task["acceptance_contract"] = {"executable_scenarios": [{"steps": [{
                "operation": "tool_call", "tool_name": "update_item", "arguments": {}
            }]}]}
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "constant_tool_result"))
            mutation.request = lambda base, method, path, body=None, key=None: (200, {})
            self.assertTrue(mutation.direct_mutation_probe(task, "http://sandbox", "key", "bypass_trainer_auth"))
        finally:
            mutation.request = original

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
    def test_readiness_helpers_remove_dynamic_fields_and_find_leaks(self):
        readiness = load_script("validate_training_readiness.py")
        value = {"request_id": "a", "nested": {"timestamp": 1, "value": 2}}
        self.assertEqual(readiness.canonical(value), {"nested": {"value": 2}})
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
