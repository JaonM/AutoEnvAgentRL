import pytest
import json
from unittest.mock import Mock

from env_factory.sandbox_runtime import interaction_phrase_matches


def test_structured_request_branch_reads_question_not_option_metadata(tmp_path):
    from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore
    simulator = object.__new__(ContractUserSimulator)
    simulator.episode_store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    simulator.episode_store.reset(seed=17)
    simulator._execution_route = Mock(return_value=None)
    simulator._select_branch = Mock(return_value=None)
    stage = {'id': 'choice', 'kind': 'option_selection', 'before_tool': 'lookup',
             'bind_to': {'argument': 'region'}, 'options': ['east', 'west'],
             'user_reply': 'east', 'retry_reply': 'Choose a region'}
    script = {'interaction_protocol': {'request_mode': 'structured_v1', 'stages': [stage]}}
    state = {'script_id': 'test', 'turn_index': 0}
    request = {'question': 'Which area should I use?', 'request': {
        'intent': 'request_input', 'tool': 'lookup', 'arguments': ['region'], 'options': ['east', 'west']}}
    result = simulator._protocol_turn(script, state, [{'role': 'assistant', 'content': json.dumps(request)}])
    simulator._select_branch.assert_called_once_with(stage, 'Which area should I use?')
    assert result['fsm_transition_applied']


@pytest.mark.parametrize('change', ['tool', 'arguments', 'intent', 'options'])
def test_structured_request_cannot_bypass_bound_business_fields(change):
    from env_factory.sandbox_runtime import interaction_request_matches, structured_interaction_request
    stage = {'kind': 'option_selection', 'before_tool': 'lookup',
             'bind_to': {'argument': 'region'}, 'options': ['east', 'west']}
    request = structured_interaction_request(stage)
    request[change] = ['wrong'] if change in {'arguments', 'options'} else 'wrong'
    assert not interaction_request_matches(stage, json.dumps({
        'question': 'Which area should I use?', 'request': request}), 'structured_v1')


@pytest.mark.parametrize('text', ['请批准预留一套。', '请批准预留 1 套。', '请批准预留１套。'])
def test_quantity_typography_matches_same_requirement(text):
    assert interaction_phrase_matches(text, '预留一套')


@pytest.mark.parametrize('text', ['请批准预留两套。', '请批准预留11套。', '请批准预留十一套。',
                                  '请批准查询一套。'])
def test_different_quantity_or_action_does_not_match(text):
    assert not interaction_phrase_matches(text, '预留一套')


def test_whitespace_does_not_remove_required_approval_word():
    assert interaction_phrase_matches('请您批准 预留一套', '批准预留一套')
    assert not interaction_phrase_matches('请查询库存', '批准预留一套')
