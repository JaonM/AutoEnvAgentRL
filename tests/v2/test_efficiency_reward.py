import pytest

from env_factory_v2.examples import simple, retrieval
from env_factory_v2.runtime import Sandbox
from env_factory_v2.reward import grade, assess_efficiency


def world():
    sandbox = Sandbox(simple())
    sandbox.interact('size', question='尺码？')
    return sandbox


def test_business_success_survives_redundancy_with_only_small_auxiliary_change():
    sandbox = world()
    for i in range(4):
        sandbox.call('find_items', {'size': 'M'}, call_id=f'q{i}')
    result = grade(sandbox, {'ids': ['I1']})
    assert result['success'] and result['business_reward'] == 1
    assert result['efficiency']['redundant_reads'] == 3
    assert result['reward'] == pytest.approx(0.9625)


def test_efficient_wrong_answer_gets_no_efficiency_credit():
    sandbox = world()
    sandbox.call('find_items', {'size': 'M'}, call_id='q')
    result = grade(sandbox, {'ids': []})
    assert not result['success'] and result['reward'] == 0
    assert result['efficiency']['score'] == 1


def test_correct_guess_without_evidence_gets_no_efficiency_credit():
    result = grade(world(), {'ids': ['I1']})
    assert result['reward'] == 0 and not result['success']


def test_all_required_pagination_is_useful():
    sandbox = Sandbox(retrieval(rows=512))
    from env_factory_v2.verification import run_scenario
    source = sandbox.compiled
    run = run_scenario(source, source['source']['scenarios'][0])
    assert run['error'] is None
    assert run['outcome']['efficiency']['redundant_reads'] == 0
    assert run['outcome']['reward'] == 1


def test_equivalent_read_path_can_satisfy_business_evidence():
    import copy
    source = simple()
    alternate = copy.deepcopy(source['tools'][0])
    alternate['id'] = 'alternative_search'
    alternate['selection_role'] = 'distractor'
    source['tools'].append(alternate)
    sandbox = Sandbox(source)
    sandbox.interact('size', question='尺码？')
    sandbox.call('alternative_search', {'size': 'M'}, call_id='q')
    assert grade(sandbox, {'ids': ['I1']})['reward'] == 1


def test_partial_view_cannot_replace_required_full_evidence():
    from env_factory_v2.tool_interference import augment
    source, _ = augment(simple())
    sandbox = Sandbox(source)
    sandbox.interact('size', question='尺码？')
    sandbox.call('find_items_view_id', {'size': 'M'}, call_id='q')
    assert grade(sandbox, {'ids': ['I1']})['reward'] == 0
