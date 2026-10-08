import copy
import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.dynamic_choice_coverage import coverage, observe, requirements
from env_factory.generation.deterministic_preflight import check_all_routes
from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore
from .test_bounded_corrections import bounded_source


@pytest.mark.parametrize('choices,refuse,stale,covered', [
    (['西区', '南区'], True, False, True),
    (['西区', '南区'], False, False, False),
    (['东区', '西区'], True, False, False),
    ([], True, False, False),
    (['西区', '南区'], True, True, False),
])
def test_only_executed_unavailable_preference_branch_counts(tmp_path, choices, refuse, stale, covered):
    source, request = bounded_source()
    compile_source(source, root=tmp_path, request=request, structural_preview=True)
    scripts = json.loads((tmp_path / 'data/user_simulation/user_scripts.json').read_text())
    script = scripts[0]
    script['variables']['refuse'] = refuse
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    store.reset(seed=17)
    profiles = json.loads((tmp_path / 'data/user_simulation/user_profiles.json').read_text())
    user = ContractUserSimulator(store, scripts=[script], profiles=profiles)
    user.reset()
    stages = script['interaction_protocol']['stages']
    conversation = [{'role': 'assistant', 'content': stages[0]['reference_response']}]
    reply = user.turn(conversation)
    conversation.append({'role': 'user', 'content': reply['user_query']})
    store.event('tool_call', {'tool_name': 'lookup_zone', 'arguments': {'name': '青松库'}},
                {'records': [{'id': 821, 'regions': '、'.join(choices)}]})
    envelope = json.loads(stages[1]['reference_response'])
    if not stale:
        envelope['request']['options'] = choices
    conversation.append({'role': 'assistant', 'content': json.dumps(envelope, ensure_ascii=False)})
    user.turn(conversation)
    observations = observe(store.replay(), [script])
    assert bool(observations) is covered
    if covered:
        assert observations[0]['observed_options'] == choices
        assert observations[0]['branch'] == 'decline'


def test_requirements_deduplicate_variants_but_preserve_distinct_bindings():
    def script(tool):
        return {'interaction_protocol': {'stages': [{'id': 'choice',
            'options_from': {'tool': tool, 'path': '$.rows[*].id'}}]}}
    required = requirements([script('a'), script('a'), script('b')])
    assert len(required) == 2
    report = coverage({'task_spec': {'interaction_complexity': {
        'dynamic_choice_requirements': required}}}, required[:1])
    assert not report['passed']
    assert report['missing'] == required[1:]


def test_missing_case_fails_before_any_semantic_judge(tmp_path):
    source, request = bounded_source()
    request['require_dynamic_choice_coverage'] = True
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        with pytest.raises(ValueError, match='DYNAMIC_CHOICE_UNCOVERED'):
            compile_source(source, root=tmp_path, request=request)
    model.assert_not_called()
    report = json.loads((tmp_path / 'deterministic_preflight_all_routes.json').read_text())
    assert report['training_ready'] is False
    assert report['dynamic_choice_coverage']['missing'][0]['stage_id'] == 'choice'
    assert 'actionable_feedback' in (tmp_path / 'execution_progress.jsonl').read_text()


def test_unexecuted_model_dependent_routes_defer_missing_evidence(tmp_path):
    required = {'stage_id': 'choice', 'options_from': {'tool': 'catalog', 'path': '$.options'}}
    artifacts = {'task_spec': {'interaction_complexity': {'dynamic_choice_requirements': [required]}}}
    def verify(*args, **kwargs):
        return {'findings': [], 'deferred': [{'reason': 'reward-dependent trajectory'}]}
    report = check_all_routes(artifacts, tmp_path, verify, None)
    assert report['dynamic_choice_coverage']['deferred']
    assert not report['dynamic_choice_coverage']['passed']
    assert not report['training_ready']


def test_probe_observations_are_required_for_final_coverage(tmp_path):
    from env_factory.generation.validation_batch import verify_independent_routes
    required = {'stage_id': 'choice', 'options_from': {'tool': 'catalog', 'path': '$.options'}}
    artifacts = {'task_spec': {'interaction_complexity': {'dynamic_choice_requirements': [required]}},
                 'execution_route_probes': [{'route_id': 'unavailable', 'initial_changes': []}]}
    def prepare(artifacts, root, probe):
        directory = root / 'route_probes' / probe['route_id']
        directory.mkdir(parents=True, exist_ok=True)
        return {'probe': True}, directory
    def verify(candidate, root, **kwargs):
        return {'passed': True, 'checks': {}, 'realism_observations': [],
                'execution_route_coverage': {'references': [], 'probes': []},
                'dynamic_choice_observations': [required] if candidate.get('probe') else []}
    report = verify_independent_routes(artifacts, tmp_path, verify, prepare)
    assert report['dynamic_choice_coverage']['passed']
    missing = copy.deepcopy(artifacts)
    missing['execution_route_probes'] = []
    with pytest.raises(ValueError, match='DYNAMIC_CHOICE_UNCOVERED'):
        verify_independent_routes(missing, tmp_path, verify, prepare)
