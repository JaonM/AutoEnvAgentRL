import copy

import pytest

from env_factory.sandbox_runtime import CompleteRetrieval


def contract():
    return {'tools': [{'tool_name': 'read', 'table': 'items', 'cursor_argument': 'cursor', 'result_field': 'records', 'page_size': 2}],
            'process_bindings': {'read': [{'tool_name': 'read', 'arguments': {'group': 'A'}}]}}


def page(token='', following='p2', count=2, total=5, group='A'):
    return {'event': 'tool_call', 'payload': {'tool_name': 'read', 'arguments': {
        'group': group, **({'cursor': token} if token else {})}},
        'result': {'records': list(range(count)), 'count': count, 'total_count': total,
                   'has_more': bool(following), 'next_cursor': following}}


def test_complete_chain_counts_distinct_pages_and_retry_is_not_progress():
    first, second, last = page(), page('p2', 'p3'), page('p3', '', count=1)
    result = CompleteRetrieval.observe(contract(), [first, first, second, second, last, last])
    assert result == {'retrieval_complete': True, 'retrieval_chains': [
        {'tool_name': 'read', 'complete': True, 'pages': 3, 'rows': 5, 'total_count': 5}]}
    assert not CompleteRetrieval.observe(contract(), [first] * 3)['retrieval_complete']


@pytest.mark.parametrize('case', ['omit_middle', 'omit_last', 'wrong_order', 'other_query', 'changed_total', 'bad_count', 'cycle'])
def test_incomplete_or_unrelated_chain_never_satisfies_target(case):
    events = [page(), page('p2', 'p3'), page('p3', '', count=1)]
    if case == 'omit_middle': events.pop(1)
    if case == 'omit_last': events.pop()
    if case == 'wrong_order': events[1:] = events[:0:-1]
    if case == 'other_query':
        events = [events[0], page(group='B'), page('p2', 'p3', group='B'), page('p3', '', count=1, group='B')]
    if case == 'changed_total': events[-1]['result']['total_count'] = 4
    if case == 'bad_count': events[-1]['result']['count'] = 2
    if case == 'cycle': events[1]['result']['next_cursor'] = 'p2'
    assert not CompleteRetrieval.observe(contract(), events)['retrieval_complete']


def test_empty_query_finishes_and_inactive_route_does_not_require_it():
    assert CompleteRetrieval.observe(contract(), [page(following='', count=0, total=0)])['retrieval_complete']
    assert CompleteRetrieval.observe(contract(), [], required_tools=set())['retrieval_complete']
    assert not CompleteRetrieval.observe(contract(), [])['retrieval_complete']


def test_query_start_must_match_actual_captured_producer_not_a_separate_lookalike():
    spec = contract()
    spec['process_bindings'] = {
        'lookup': [{'tool_name': 'lookup', 'arguments': {'name': 'chosen'}}],
        'read': [{'tool_name': 'read', 'arguments': {'group': {'$ref': 'group'}},
                  'captures': [{'name': 'group', 'tool_name': 'lookup', 'path': '$.group'}]}]}
    producer = {'event': 'tool_call', 'payload': {'tool_name': 'lookup', 'arguments': {'name': 'wrong'}}, 'result': {'group': 'A'}}
    pages = [page(), page('p2', 'p3'), page('p3', '', count=1)]
    assert not CompleteRetrieval.observe(spec, [producer, *pages])['retrieval_complete']
    producer['payload']['arguments']['name'] = 'chosen'
    assert CompleteRetrieval.observe(spec, [producer, *pages])['retrieval_complete']


def test_ablation_rebinds_cursor_but_preserves_original_business_decision():
    from env_factory.contracts.trajectory_probes import bind_success_steps, restore_pagination_refs
    steps = [{'operation': 'tool_call', 'tool_name': 'read', 'arguments': {'group': 'A'}, 'capture': {'next': '$.next_cursor', 'id': '$.id'}},
             {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {'group': {'$ref': 'id'}, 'cursor': {'$ref': 'next'}}}]
    history = [{'operation': 'tool_call', 'status': 200, 'body': {'next_cursor': 'old-episode-token', 'id': 'observed-A'}},
               {'operation': 'tool_call', 'status': 200, 'body': {}}]
    bound = bind_success_steps(steps, history)
    restored = restore_pagination_refs(bound, steps, [{'tool_name': 'read', 'pagination': {'cursor_argument': 'cursor'}}])
    assert restored[1]['arguments'] == {'group': 'observed-A', 'cursor': {'$ref': 'next'}}
    assert steps[1]['arguments']['group'] == {'$ref': 'id'}


def test_cross_episode_signature_only_normalizes_declared_transport_not_business_values():
    from scripts.sandbox.validate_agentic_training_value import tool_evidence_signature
    steps = [{'operation': 'tool_call', 'tool_name': 'read'}] * 2
    implementations = [{'tool_name': 'read', 'pagination': {'cursor_argument': 'cursor'}}]
    first = {'history': [{'operation': 'tool_call', 'status': 200,
                          'body': {'records': [{'next_cursor': 'business-value'}], 'next_cursor': 'token-a', 'count': 1}}] * 2}
    second = copy.deepcopy(first)
    for item in second['history']: item['body']['next_cursor'] = 'token-b'
    signature = lambda run: tool_evidence_signature(run, steps=steps, implementations=implementations)
    assert signature(first) == signature(second)
    assert tool_evidence_signature(first) != tool_evidence_signature(second)
    second['history'][0]['body']['records'][0]['next_cursor'] = 'changed-business-value'
    assert signature(first) != signature(second)
    second = copy.deepcopy(first)
    second['history'][1] = copy.deepcopy(second['history'][1])
    second['history'][1]['body']['next_cursor'] = 'different-token'
    assert signature(first) != signature(second)
