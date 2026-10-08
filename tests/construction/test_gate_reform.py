"""Regression tests for gate correctness, evidence invalidation and review routing."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    path = ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GateReformTest(unittest.TestCase):
    def test_determinism_ignores_only_failed_http_envelope_request_id(self):
        checker = script('sandbox/validate_training_readiness.py')
        body = {'error': {'code': 'RESPONSE_LOST', 'message': 'lost', 'details': None, 'request_id': 'a'}}
        a = {'history': [{'operation': 'tool_call', 'status': 504, 'body': body}],
             'step_results': {'write': body}, 'variables': {}}
        b = copy.deepcopy(a)
        b['history'][0]['body']['error']['request_id'] = 'b'
        self.assertEqual(checker.canonical(a), checker.canonical(b))
        b['history'][0]['body']['error']['code'] = 'OTHER_FAILURE'
        self.assertNotEqual(checker.canonical(a), checker.canonical(b))
        self.assertNotEqual(checker.canonical({'request_id': 'business-a'}),
                            checker.canonical({'request_id': 'business-b'}))

    def test_readiness_goal_uses_decision_checkpoint_not_initial_or_final_stock(self):
        from types import SimpleNamespace
        import env_factory.sandbox_runtime as runtime
        baseline = {'stock': [{'name': 'target', 'available': 20, 'reserved': 0}]}
        checkpoint = {'stock': [{'name': 'target', 'available': 16, 'reserved': 0}]}
        current = {'stock': [{'name': 'target', 'available': 4, 'reserved': 12}]}
        store = SimpleNamespace(get_state=lambda key, default: {'read_stock': checkpoint})
        app = SimpleNamespace(episode_store=store, business_snapshot=lambda: current)
        predicate = {'table': 'stock', 'where': {'name': 'target'}, 'count': 1,
            'value_expressions': {'available': {'op': 'sub', 'args': [
                {'at_tool': {'name': 'read_stock', 'expression': {'lookup': {
                    'table': 'stock', 'field': 'available', 'where': {'name': 'target'}}}}}, {'literal': 12}]}}}
        with patch.dict('sys.modules', {'sandbox_runtime': runtime}):
            checker = script('sandbox/validate_training_readiness.py')
            state = checker.goal_evaluation_state(app, baseline)
            self.assertTrue(runtime.BusinessGoalEvaluator.evaluate([predicate], state, baseline))
            current['stock'][0]['available'] = 5
            self.assertFalse(runtime.BusinessGoalEvaluator.evaluate([predicate],
                checker.goal_evaluation_state(app, baseline), baseline))

    def test_reset_evidence_allows_only_clean_protocol_initialization(self):
        from env_factory.contracts.episode_evidence import initial_replay_events
        event = {'sequence': 1, 'event': 'user_protocol', 'timestamp': 1,
                 'payload': {'script_id': 'script-1', 'required_stages': ['approval']}, 'result': {}}
        self.assertEqual(initial_replay_events([event]), initial_replay_events([{**event, 'timestamp': 2}]))
        self.assertEqual(initial_replay_events([]), [])
        for invalid in ([event, event], [{**event, 'event': 'agent_response'}],
                        [{**event, 'result': {'stale': True}}],
                        [{**event, 'payload': {**event['payload'], 'previous_episode': 'A'}}]):
            self.assertIsNone(initial_replay_events(invalid))

    def test_mutation_probe_completes_dialogue_and_preserves_idempotency(self):
        mutation = script('sandbox/run_mutation_tests.py')
        task = {'acceptance_contract': {'executable_scenarios': [{'kind': 'goal_success', 'steps': [
            {'operation': 'dialogue_turn', 'content': 'Authorize writing?'},
            {'operation': 'tool_call', 'tool_name': 'write', 'arguments': {}, 'idempotency_key': 'write-1'}]}]}}
        approved = False
        def request(base, method, path, body=None, *, key=None, headers=None):
            nonlocal approved
            if path == '/v1/reset':
                approved = False
                return 200, {}
            if path == '/v1/user_simulator':
                self.assertEqual(headers['Authorization'], 'Bearer trainer')
                approved = True
                return 200, {'user_query': 'Approved'}
            if path.startswith('/v1/tools/'):
                if not approved:
                    return 403, {}
                self.assertEqual(headers, {'Idempotency-Key': 'write-1'})
                return 200, {'mutation': 'skip_business_write'}
            return 200, {}
        with patch.object(mutation, 'request', side_effect=request):
            self.assertTrue(mutation.direct_mutation_probe(task, 'http://sandbox', 'trainer', 'skip_business_write'))
        with patch.object(mutation, 'request', return_value=(403, {})):
            self.assertFalse(mutation.direct_mutation_probe(task, 'http://sandbox', 'trainer', 'skip_business_write'))

    def test_outer_acceptance_accepts_executed_interactive_reference(self):
        from tests.generation.test_interaction_contract import interactive_source
        from env_factory.generation.agent_authoring import compile_source
        source, request = interactive_source()
        with tempfile.TemporaryDirectory() as directory:
            task = compile_source(source, root=Path(directory), request=request)
        outer = script('sandbox/generate_outer_conformance.py')
        names = [item['function']['name'] for item in task['tools']]
        accepted = outer.check_acceptance_contract(task, names, task['metrics'])
        self.assertTrue(any(step['operation'] == 'dialogue_turn'
            for scenario in accepted['executable_scenarios'] for step in scenario['steps']))
        malformed = copy.deepcopy(task)
        malformed['acceptance_contract']['executable_scenarios'][0]['steps'][0]['operation'] = 'invented_operation'
        with self.assertRaises(SystemExit):
            outer.check_acceptance_contract(malformed, names, task['metrics'])

    def test_stateful_final_answer_has_its_declared_weight(self):
        from env_factory.contracts.reward_contract import terminal_outcome_weight
        task = {"environment_plan": {"mode": "stateful"},
                "metrics": [{"id": "answer", "category": "outcome", "weight": .4},
                            {"id": "state", "category": "outcome", "weight": .6}],
                "metric_implementations": [{"metric_id": "answer", "source": "final_agent_response"}]}
        self.assertEqual(terminal_outcome_weight(task), .4)
        task["metric_implementations"] = []
        self.assertEqual(terminal_outcome_weight(task), 0)

    def test_schema_probe_only_violates_additional_properties(self):
        from env_factory.sandbox_runtime import validate_json_schema, SandboxError
        outer = script('sandbox/generate_outer_conformance.py')
        schema = {'type':'object','properties':{'name':{'type':'string'}},'required':['name'],'additionalProperties':False}
        task = {'tools':[{'function':{'name':'lookup','parameters':schema}}]}
        case = next(c for c in outer.invalid_schema_cases(task) if c['case'] == 'unexpected_property')
        with self.assertRaisesRegex(SandboxError, 'unexpected properties'):
            validate_json_schema(schema, case['request'])
        permissive = {**schema, 'additionalProperties':True}
        validate_json_schema(permissive, case['request'])

    def test_typed_scoped_delta_does_not_match_other_rows(self):
        gate = script('sandbox/validate_training_readiness.py')
        rows = {'items':[{'id':1,'value':0},{'id':2,'value':4}]}
        delta = {'table':'items','where':{'id':1},'field':'value','before':0,'after':1}
        self.assertTrue(gate.valid_delta_precondition(delta,rows))
        self.assertFalse(gate.valid_delta_precondition({**delta,'before':4},rows))
        self.assertFalse(gate.valid_delta_precondition({**delta,'before':False},rows))
        self.assertFalse(gate.valid_delta_precondition({'before':0,'after':1},rows))

    def test_unreproduced_review_is_quarantined_and_not_repaired(self):
        review = script('sandbox/adjudicate_review.py')
        extract = script('sandbox/extract_delivery_defects.py')
        from env_factory.sandbox_scoring import review_source_hashes
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'task_impl.py').write_text('pass\n')
            report={'status':'fail','score':.9,'review_run_id':'test',
                    'checked_modules':sorted(review.REQUIRED_MODULES),
                    'findings':[{'severity':'high','evidence':'unsupported','category':'reward'}],
                    'source_hashes':review_source_hashes(root)}
            report['adjudication']=review.adjudicate(root,report)
            self.assertEqual(report['adjudication']['status'],'review_unresolved')
            (root/'review_report.json').write_text(json.dumps(report))
            self.assertEqual(extract.extract(root,'semantic_review'),[])
            with patch.object(review,'replay_probe',return_value={'status':'confirmed','evidence':{'reward':1}}):
                report['findings'][0]['reproduction']={'kind':'no_tool_reward','steps':[]}
                self.assertEqual(review.adjudicate(root,report)['status'],'confirmed_defect')

    def test_gate_cache_invalidates_on_output_input_and_checker_changes(self):
        from env_factory.evidence.gate_cache import GATES,ARTIFACTS,save,load
        with tempfile.TemporaryDirectory() as d:
            project=Path(d);root=project/'sandbox';root.mkdir()
            (project/'src/env_factory').mkdir(parents=True);(project/'scripts').mkdir()
            code=project/'scripts/check.py';code.write_text('pass\n')
            app=root/'app.py';app.write_text('pass\n')
            for name in ARTIFACTS:(root/name).write_text('{}')
            results={name:[True,'executed'] for name in GATES}
            save(root,project,results);self.assertIsNotNone(load(root,project))
            (root/'training_readiness.json').write_text('{"changed":true}')
            self.assertIsNone(load(root,project))
            save(root,project,results);app.write_text('x=1\n');self.assertIsNone(load(root,project))
            save(root,project,results);code.write_text('x=2\n');self.assertIsNone(load(root,project))

    def test_live_environment_and_agent_outcomes_are_separate(self):
        live=script('rollout/run_live_rollout.py')
        report=live.summarize_episodes([{'agent_success':False,'issues':[]}],2/3)
        self.assertTrue(report['environment_qualified'])
        self.assertFalse(report['agent_policy_qualified'])
        self.assertFalse(report['task_solvability_witness'])
        self.assertIsNone(report['failure_owner'])
        self.assertTrue(report['passed'])
        self.assertEqual(report['agent_failure_owner'],'agent')

    def test_clean_review_is_not_blocked_by_bare_subjective_score(self):
        review = script("sandbox/adjudicate_review.py")
        with tempfile.TemporaryDirectory() as directory:
            report = {"status": "pass", "score": 0.79, "findings": [],
                      "checked_modules": sorted(review.REQUIRED_MODULES)}
            result = review.adjudicate(Path(directory), report)
            self.assertEqual(result["status"], "resolved")

    def test_one_success_in_three_is_environment_evidence_not_policy_threshold(self):
        live = script("rollout/run_live_rollout.py")
        episodes = [{"agent_success": value, "issues": []} for value in (True, False, False)]
        report = live.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertTrue(report["task_solvability_witness"])
        self.assertFalse(report["agent_policy_qualified"])
        episodes[1]["issues"] = ["unstable_reward_read"]
        self.assertFalse(live.summarize_episodes(episodes, 2 / 3)["passed"])

    def test_fixed_goal_uses_verified_completion_and_incomplete_goal_still_calls_model(self):
        from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            store=EpisodeStore(Path(d)/'episodes.sqlite3');store.reset(episode_id='goal',seed=1)
            render=Mock(return_value={'user_query':'请完成任务。','match_status':'unmatched','outcome_category':'agent_premature_completion','reason_code':'not_done'})
            checker=Mock(return_value=False)
            simulator=ContractUserSimulator(store,profiles=[{'profile_id':'p'}],scripts=[{
                'script_id':'s','initial_state':'start','variables':{},
                'states':[{'state_id':'start','terminal':False},{'state_id':'done','terminal':True}],
                'transitions':[{'transition_id':'finish','from_state':'start','to_state':'done','outcome_category':'goal_satisfied','should_end':True,'condition':'goal verified','updates':{}}],
            }],renderer=render,completion_check=checker)
            simulator.reset();store.set_state('final_agent_response','wrong')
            result=simulator.turn([{'role':'assistant','content':'wrong'}]);self.assertFalse(result['should_end']);render.assert_called_once()
            simulator.reset();store.set_state('final_agent_response','correct');checker.return_value=True;render.reset_mock()
            result=simulator.turn([{'role':'assistant','content':'correct'}])
            self.assertTrue(result['should_end']);self.assertEqual(result['outcome_category'],'goal_satisfied');render.assert_not_called()
            self.assertEqual(store.replay()['events'][-1]['payload']['decision_source'],'executable_goal')


def test_answer_partial_credit_does_not_reduce_policy_quality():
    from env_factory.sandbox_scoring import sandbox_quality_factors
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'review_report.json').write_text(json.dumps({'score': .9}))
        (root / 'task.json').write_text(json.dumps({
            'environment_plan': {'mode': 'stateful'},
            'metrics': [{'id': 'answer', 'category': 'outcome', 'weight': .4}],
            'metric_implementations': [{'metric_id': 'answer', 'source': 'final_agent_response'}]}))
        (root / 'agentic_training_value.json').write_text(json.dumps({'evidence': {'counterfactuals': {
            'goal_success': {'reward': 1}, 'wrong_final_answer': {'reward': .6},
            'no_tools': {'reward': 0}}}}))
        assert sandbox_quality_factors(root)['declared_training_policy'] == 1
