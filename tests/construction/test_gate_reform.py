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
