from review_fixtures import with_requirement_reviews
import copy
import json
from unittest.mock import Mock

import pytest

from env_factory_v2.challenge import _case_scenario, _execute_case
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple
from env_factory_v2.expressions import ContractError


def setup_case():
    source = simple()
    compiled = compile_contract(source)
    case = {'id': 'independent', 'scenario_id': 'correct', 'rule_ids': ['size_matches'],
            'reason': 'Independent trajectory fixture', 'mutations_json': '[]',
            'answer_json': '{}', 'user_overrides_json': '{}', 'expect_success': True,
            'steps_json': 'null'}
    return compiled, case, {s['id']: s for s in compiled['source']['scenarios']}


def test_independent_trajectory_can_add_missing_clock_read_without_editing_contract():
    compiled, case, scenarios = setup_case()
    source = copy.deepcopy(compiled['source'])
    source['success'] = [{'id': 'done', 'public': 'Read time and retrieve results.',
        'when': {'literal': True}, 'required_tools': ['find_items'], 'required_stages': ['size'],
        'required_platform_observations': [{'capability': 'time', 'freshness': 'current'}]}]
    compiled = compile_contract(source)
    before = copy.deepcopy(compiled)
    agent = Mock()
    missing = _execute_case(compiled, case, scenarios, agent)
    assert not missing['passed']
    case['steps_json'] = json.dumps(scenarios['correct']['steps'] + [{'kind': 'time'}])
    result = _execute_case(compiled, case, scenarios, agent)
    assert result['passed']
    assert any(turn['kind'] == 'platform_observation' for turn in result['first']['trace'])
    assert compiled == before


def test_explicit_empty_steps_is_not_inherited_trajectory():
    compiled, case, scenarios = setup_case()
    case.update(steps_json='[]', answer_json=json.dumps({'ids': {'literal': ['I1']}}), expect_success=False)
    result = _execute_case(compiled, case, scenarios, Mock())
    assert result['passed']
    assert result['first']['trace'] == []
    assert result['first']['outcome']['reward'] == 0


def test_independent_steps_do_not_inherit_expected_runtime_error():
    compiled, case, scenarios = setup_case()
    scenarios['correct']['expect_error'] = 'undisclosed'
    case.update(steps_json=json.dumps(scenarios['correct']['steps'][1:]), expect_success=False)
    scenario = _case_scenario(compiled, case, scenarios)
    assert 'expect_error' not in scenario
    result = _execute_case(compiled, case, scenarios, Mock())
    assert result['first']['error'] is not None
    assert not result['passed']


@pytest.mark.parametrize('steps', [
    {}, [{'kind': 'call', 'id': 'unknown', 'tool': 'invented', 'arguments': {}}],
    [{'kind': 'user', 'stage': 'invented', 'question': 'Invented?'}],
    [{'kind': 'shell', 'command': 'anything'}],
    [{'kind': 'wait', 'seconds': 86401}],
    [{'kind': 'wait', 'seconds': True}],
])
def test_independent_steps_must_use_existing_grammar_and_interface(steps):
    compiled, case, scenarios = setup_case()
    case['steps_json'] = json.dumps(steps)
    with pytest.raises(ContractError):
        _execute_case(compiled, case, scenarios, Mock())


def test_full_review_executes_independent_steps_and_preserves_original_identity():
    from env_factory_v2.challenge import challenge

    compiled, case, scenarios = setup_case()
    case['steps_json'] = json.dumps(scenarios['correct']['steps'] + [{'kind': 'time'}])
    negative = {**case, 'id': 'no_actions', 'steps_json': '[]', 'expect_success': False,
                'answer_json': json.dumps({'ids': {'literal': ['I1']}})}
    agent = Mock()
    agent.ask.return_value = (with_requirement_reviews(compiled['source'], {'business_consistent': True, 'reason': 'Unit test, not model evidence',
                              'cases': [case, negative]}), {'unit_test_only': True})
    report = challenge(compiled, {'goal': compiled['source']['task']}, agent)
    assert report['passed']
    assert report['contract_identity'] == compiled['identity']
    assert len(report['records']) == 2
    assert all(r['passed'] for r in report['records'])
    agent.ask.assert_called_once()


def test_compact_steps_preserve_exact_trajectory_and_execution():
    from env_factory_v2.verification import replay_identity
    compiled, case, scenarios = setup_case()
    steps = scenarios['correct']['steps']
    before = copy.deepcopy(scenarios)
    full = {**case, 'steps_json': json.dumps(steps + [{'kind': 'time'}])}
    compact = {**case, 'steps_json': json.dumps([{'$step': i} for i in range(len(steps))] + [{'kind': 'time'}])}
    assert _case_scenario(compiled, compact, scenarios) == _case_scenario(compiled, full, scenarios)
    first = _execute_case(compiled, full, scenarios, Mock())
    second = _execute_case(compiled, compact, scenarios, Mock())
    assert first['passed'] and second['passed']
    assert replay_identity(first['first']) == replay_identity(second['first'])
    assert scenarios == before


@pytest.mark.parametrize('reference', [-1, 10000, True, '0', None])
def test_invalid_compact_step_reference_fails_closed(reference):
    compiled, case, scenarios = setup_case()
    case['steps_json'] = json.dumps([{'$step': reference}])
    with pytest.raises(ContractError, match='step reference'):
        _case_scenario(compiled, case, scenarios)


def test_compact_reference_cannot_smuggle_edits_or_bypass_dependencies():
    compiled, case, scenarios = setup_case()
    case['steps_json'] = json.dumps([{'$step': 0, 'kind': 'wait', 'seconds': 1}])
    with pytest.raises(ContractError, match='step reference'):
        _case_scenario(compiled, case, scenarios)
    case.update(steps_json=json.dumps([{'$step': 1}]), expect_success=False)
    result = _execute_case(compiled, case, scenarios, Mock())
    assert result['first']['error'] is not None
    assert not result['passed']  # A missing prerequisite is not a valid negative case.
