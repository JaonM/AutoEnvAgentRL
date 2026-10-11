import copy
from unittest.mock import Mock

import pytest

from env_factory_v2.business_definition import compile_definition
from env_factory_v2.examples import direct, lit
from env_factory_v2.expressions import ContractError
from env_factory_v2.verification import run_scenario


def definition():
    source = direct()
    environment = {key: copy.deepcopy(source[key]) for key in ['rules', 'tables', 'tools', 'time']}
    environment.update(version='1.0', id='environment', communications=[{
        'id': 'explain_fact', 'rule_id': 'fixed_fact', 'obligation': 'Explain the current public temperature.',
        'when': {'rule': {'id': 'fixed_fact', 'arguments': {}}}}])
    task = {key: copy.deepcopy(value) for key, value in source.items()
            if key not in {'rules', 'tables', 'tools', 'time', 'version', 'task'}}
    task.update(version='1.0', request=source['task'], messages=[{
        'id': 'reason', 'meaning': 'Explanation', 'obligations': ['explain_fact'],
        'context_fields': ['temperature'], 'weight': 0.25}])
    task['answers'][0]['weight'] = 0.75
    for case in task['scenarios']:
        case['answer']['reason'] = lit('The reported temperature is 18 degrees Celsius.')
    return environment, task


def judge_result(payload):
    return {'criteria': {c['id']: True for c in payload['field']['criteria']},
            'provenance': {'unit_test_only': True}}


def test_message_derives_semantic_criterion_and_dynamic_facts():
    environment, task = definition()
    compiled, receipt = compile_definition(environment, task)
    assert receipt['derived_messages'] == ['reason']
    assert compiled['source']['explanations'][0]['criteria'] == environment['communications']
    assert [a['id'] for a in compiled['source']['answers']] == ['temperature']
    judge = Mock(side_effect=judge_result)
    result = run_scenario(compiled, task['scenarios'][0], judge=judge)
    assert result['outcome']['success']
    assert judge.call_args.args[0]['authoritative_facts'] == {'temperature': 18}
    assert judge.call_args.args[0]['text'] == task['scenarios'][0]['answer']['reason']['literal']
    # A semantic pass cannot override a wrong structured fact.
    assert not run_scenario(compiled, task['scenarios'][1], judge=judge)['outcome']['success']


def test_inapplicable_communication_does_not_require_model_or_menu_recitation():
    environment, task = definition()
    environment['rules'][0]['predicate'] = lit(False)
    compiled, _ = compile_definition(environment, task)
    case = copy.deepcopy(task['scenarios'][0])
    case['answer']['reason'] = lit('')
    judge = Mock(side_effect=AssertionError('inactive obligation must not call model'))
    result = run_scenario(compiled, case, judge=judge)
    assert result['outcome']['success']
    judge.assert_not_called()
    environment['rules'][0]['predicate'] = lit(True)
    compiled, _ = compile_definition(environment, task)
    rejecting_judge = Mock(return_value={'criteria': {'explain_fact': False}, 'provenance': {'unit_test_only': True}})
    assert not run_scenario(compiled, case, judge=rejecting_judge)['outcome']['success']
    rejecting_judge.assert_called_once()


def test_only_applicable_criteria_are_submitted_and_required():
    environment, task = definition()
    environment['communications'].append({'id': 'other_branch', 'rule_id': 'fixed_fact',
        'obligation': 'Explain the unavailable branch.', 'when': lit(False)})
    task['messages'][0]['obligations'].append('other_branch')
    compiled, _ = compile_definition(environment, task)
    judge = Mock(side_effect=judge_result)
    assert run_scenario(compiled, task['scenarios'][0], judge=judge)['outcome']['success']
    assert [c['id'] for c in judge.call_args.args[0]['field']['criteria']] == ['explain_fact']


@pytest.mark.parametrize('condition', [{'ref': 'answer.temperature'}, {'ref': 'evidence.result'}])
def test_applicability_cannot_depend_on_submitted_or_policy_controlled_evidence(condition):
    environment, task = definition()
    environment['communications'][0]['when'] = condition
    with pytest.raises(ContractError):
        compile_definition(environment, task)


def test_task_cannot_rewrite_shared_obligation():
    environment, task = definition()
    task['messages'][0]['criteria'] = []
    with pytest.raises(ContractError, match='fields mismatch'):
        compile_definition(environment, task)


def test_non_boolean_applicability_cannot_silently_skip_requirement():
    environment, task = definition()
    environment['communications'][0]['when'] = lit(0)
    compiled, _ = compile_definition(environment, task)
    result = run_scenario(compiled, task['scenarios'][0])
    assert 'reasoning applicability must produce boolean' in result['error']
