import copy
import tempfile
import json
import os
import subprocess
import sys
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock

from env_factory.generation.spec_pipeline import instantiate, compile_spec, generate, verify_execution
from env_factory.generation.task_generator import TaskGenerator, TaskGenerationError
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
from env_factory.task_pipeline import PipelineGenerationError
from env_factory.tasks.task import Task
from examples.generate_task import _write_task_artifact
from scripts.sandbox.assess_task_buildability import assess


class SpecPipelineTest(unittest.TestCase):
    def _assert_original_gate(self, prototype):
        from scripts.sandbox.generate_sandbox_scaffold import generate as scaffold
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = generate(seed=42, artifact_dir=root, prototype=prototype)
            path = _write_task_artifact(root, Task(a['task'], a['environment'], a['metrics'],
                                        task_intent=a['task_intent'], artifacts=a), 'multi_step_agentic')
            contract = json.loads(path.read_text())
            contract.pop('actions')
            (root / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
            scaffold(root)
            for name in ('sandbox_runtime.py', 'runtime_llm.py'):
                shutil.copy2(project / 'src/env_factory' / name, root / name)
            report = root / 'gate.json'
            result = subprocess.run([sys.executable, str(project / 'scripts/sandbox/validate_agentic_training_value.py'),
                '--root', str(root), '--output', str(report)], capture_output=True, text=True,
                env={**os.environ, 'PYTHONPATH': str(project / 'src'),
                     'SANDBOX_TRAINER_API_KEY': 'test-spec-key', 'SANDBOX_EVALUATOR_MOCK': '1'}, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-2000:])
            evidence = json.loads(report.read_text())
            self.assertEqual(evidence['failures'], [])
            if prototype == 'lookup_join_sum':
                self.assertTrue(evidence['evidence']['business_data_reward_sensitivity']['proved'])

    def test_original_sandbox_gate_handles_aggregate_without_quoted_inputs(self):
        self._assert_original_gate("lookup_join_sum")

    def test_original_sandbox_gate_accepts_both_write_prototypes(self):
        for prototype in ("lookup_update", "constraint_create"):
            with self.subTest(prototype=prototype):
                self._assert_original_gate(prototype)

    def test_thirty_seeds_pass_original_gate_and_runtime_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in range(90):
                seed, prototype = case % 30, ("lookup_join_sum", "lookup_update", "constraint_create")[case // 30]
                with self.subTest(seed=seed, prototype=prototype):
                    root = Path(directory) / str(case)
                    a = generate(seed=seed, artifact_dir=root, prototype=prototype)
                    _write_task_artifact(root, Task(a['task'], a['environment'], a['metrics'],
                                         task_intent=a['task_intent'], artifacts=a), 'multi_step_agentic')
                    self.assertTrue(assess(root)['buildable'])
                    self.assertTrue(all(a['generation_pipeline']['verification']['execution_checks'].values()))
                    if prototype == 'lookup_update':
                        goal = a['task_spec']['goal_contract']
                        self.assertEqual(len(goal['expected_delta']), 1)
                        self.assertEqual(goal['expected_delta'][0]['where'], {'name': instantiate(seed=seed, prototype=prototype)['selector']})
                        self.assertTrue(goal['forbidden_deltas'])
                    scripts = json.loads((root / 'data/user_simulation/user_scripts.json').read_text())
                    for script in scripts:
                        for transition in script['transitions']:
                            if transition['outcome_category'] == 'goal_satisfied':
                                self.assertTrue(transition['should_end'])
                                self.assertEqual(transition['to_state'], 'done')
                    public = str(a['public_input']) + str(a['tools'])
                    for row in instantiate(seed=seed, prototype=prototype)['tables'][0]['rows']:
                        self.assertNotIn(str(row.get('account_id', row.get('id'))), public)

    def test_spec_backend_does_not_call_graph_or_model(self):
        graph, llm = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            task = TaskGenerator(graph, llm, generation_backend='spec').generate(
                seed=17, artifact_dir=directory, task_intent="calculate", training_category='multi_step_agentic')
        self.assertEqual(task.task_intent, 'calculate')
        graph.random_scene_event_path.assert_not_called()
        self.assertEqual(llm.mock_calls, [])
        with self.assertRaisesRegex(TaskGenerationError, 'SPEC_UNSUPPORTED'):
            TaskGenerator(graph, llm, generation_backend='spec').generate(task_intent='diagnose')

    def test_constant_reward_is_rejected_by_runtime_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            a = generate(seed=42, artifact_dir=Path(directory))
            for metric in a['metric_implementations']:
                metric['score_mapping'] = {'pass': 1, 'fail': 1}
            with self.assertRaisesRegex(PipelineGenerationError, 'SPEC_EXECUTION_FAILED'):
                verify_execution(a, Path(directory))

    def test_dynamic_join_tracks_current_mapping_and_not_fixed_fixture_id(self):
        evaluator = DeclarativeMetricEvaluator()
        state = {'accounts': [{'name': 'A', 'id': 31}],
                 'entries': [{'id': 31, 'q': 5}, {'id': 72, 'q': 19}]}
        expression = {'aggregate': {'table': 'entries', 'field': 'q', 'op': 'sum',
                      'where': {'id': {'lookup': {'table': 'accounts', 'field': 'id', 'where': {'name': 'A'}}}}}}
        self.assertEqual(evaluator._numeric_expression(expression, state), 5)
        state['accounts'][0]['id'] = 72
        self.assertEqual(evaluator._numeric_expression(expression, state), 19)
        state['entries'].append({'id': 72, 'q': 3})
        self.assertEqual(evaluator._numeric_expression(expression, state), 22)
        malformed = copy.deepcopy(expression)
        malformed['aggregate']['where']['id'] = {'execute': 'arbitrary code'}
        with self.assertRaises(SandboxError):
            evaluator._numeric_expression(malformed, state)

    def test_seed_reproducibility_and_unsupported_spec(self):
        self.assertEqual(instantiate(seed=31), instantiate(seed=31))
        self.assertNotEqual(instantiate(seed=31), instantiate(seed=32))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PipelineGenerationError, 'SPEC_UNSUPPORTED'):
                compile_spec({'version': '1.0', 'prototype': 'unknown'}, artifact_dir=Path(directory))

    def test_ambiguous_supplier_constraints_are_rejected_before_materialization(self):
        spec = instantiate(seed=7, prototype='constraint_create')
        request = next(r for r in spec['tables'][0]['rows'] if r['name'] == spec['selector'])
        suppliers = spec['tables'][1]['rows']
        for supplier in suppliers:
            supplier['capacity'] = request['quantity'] + 1
            supplier['unit_price'] = request['max_price'] - 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PipelineGenerationError, 'exactly one candidate'):
                compile_spec(spec, artifact_dir=Path(directory))
            self.assertFalse((Path(directory) / 'spec_verification.json').exists())

    def test_already_satisfied_update_is_rejected_before_construction(self):
        spec = instantiate(seed=7, prototype='lookup_update')
        next(r for r in spec['tables'][0]['rows'] if r['name'] == spec['selector'])['status'] = '已复核'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PipelineGenerationError, 'declared initial'):
                compile_spec(spec, artifact_dir=Path(directory))
