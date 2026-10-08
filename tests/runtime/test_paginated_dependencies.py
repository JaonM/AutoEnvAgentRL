"""Business dependencies survive changed page counts; traversal is still mandatory."""
import copy

import pytest

from env_factory.sandbox_runtime import ContractRewardGate, ContractToolRegistry, DeclarativeToolCompiler
from env_factory.tasks.task_spec import _capability_edges
from tests.runtime.test_pagination import dataset


def reference():
    return [
        {'operation': 'tool_call', 'tool_name': 'lookup', 'arguments': {'name': 'group1'},
         'capture': {'group': '$.records[0].name'}},
        {'operation': 'tool_call', 'tool_name': 'search', 'arguments': {'group': {'$ref': 'group'}},
         'capture': {'p2': '$.next_cursor'}},
        {'operation': 'tool_call', 'tool_name': 'search',
         'arguments': {'group': {'$ref': 'group'}, 'cursor': {'$ref': 'p2'}},
         'capture': {'p3': '$.next_cursor'}},
        {'operation': 'tool_call', 'tool_name': 'search',
         'arguments': {'group': {'$ref': 'group'}, 'cursor': {'$ref': 'p3'}}},
    ]


def retrieval(tmp_path, rows):
    data, spec, _ = dataset(tmp_path, count=360, page_size=5)
    data.replace_table('items', [r for r in data.table('items') if r['group_id'] == 1][:rows])
    lookup = {'tool_name': 'lookup', 'operation': 'select', 'table': 'groups',
              'filters': [{'argument': 'name', 'column': 'name', 'operator': 'eq'}]}
    tools = [{'type': 'function', 'function': {'name': name, 'parameters': {
        'type': 'object', 'properties': {k: {'type': 'string'} for k in keys},
        'required': [keys[0]], 'additionalProperties': False}}}
        for name, keys in [('lookup', ['name']), ('search', ['group', 'cursor'])]]
    registry = ContractToolRegistry(tools, DeclarativeToolCompiler(data).compile_all([lookup, spec]),
                                    event_recorder=data.episode_store.event)
    edges = _capability_edges([{'kind': 'goal_success', 'steps': reference()}], [lookup, spec])
    contract = {'tools': [{'tool_name': 'search', 'cursor_argument': 'cursor',
                          'result_field': 'records', 'page_size': 5}], 'process_bindings': {
        'lookup': [{'tool_name': 'lookup', 'arguments': {'name': 'group1'}}],
        'search': [{'tool_name': 'search', 'arguments': {'group': {'$ref': 'group'}},
                    'captures': [{'name': 'group', 'tool_name': 'lookup', 'path': '$.records[0].name'}]}]}}
    gate = ContractRewardGate({'training_contract': {'category': 'multi_step_agentic'},
        'capability_dag': {'nodes': ['lookup', 'search'], 'edges': edges}, 'retrieval_contract': contract},
        [{'id': 'outcome', 'category': 'outcome', 'score_range': [0, 1]}])
    return data.episode_store, registry, gate, edges


def execute(registry, *, explicit_empty=False):
    group = registry.execute('lookup', {'name': 'group1'})['records'][0]['name']
    args = {'group': group, **({'cursor': ''} if explicit_empty else {})}
    while True:
        page = registry.execute('search', args)
        if not page['has_more']:
            break
        args['cursor'] = page['next_cursor']


@pytest.mark.parametrize('rows', [0, 1, 5, 6, 11, 17])
@pytest.mark.parametrize('explicit_empty', [False, True])
def test_runtime_query_length_does_not_require_reference_page_count(tmp_path, rows, explicit_empty):
    store, registry, gate, edges = retrieval(tmp_path, rows)
    execute(registry, explicit_empty=explicit_empty)
    diagnostic = {}
    assert gate.apply({'outcome': 1}, {'trajectory': store.replay()}, diagnostics=diagnostic)['outcome'] == 1
    assert diagnostic['dependency_path_passed'] and diagnostic['retrieval_complete']
    assert diagnostic['retrieval_chains'][0]['rows'] == rows
    assert len(edges) == 1
    assert [call['tool_name'] for call in edges[0]['capture_plan']] == ['lookup']


@pytest.mark.parametrize('failure', ['omit_middle', 'omit_last', 'repeat_first', 'wrong_group',
                                    'missing_producer', 'wrong_producer', 'reordered', 'changed_business_argument'])
def test_wrong_dataflow_or_incomplete_pages_still_get_zero(tmp_path, failure):
    store, registry, gate, edges = retrieval(tmp_path, 11)
    execute(registry)
    events = copy.deepcopy([e for e in store.replay()['events'] if e['event'] == 'tool_call'])
    assert len(events) == 4
    if failure == 'omit_middle': del events[2]
    if failure == 'omit_last': events.pop()
    if failure == 'repeat_first': events = [events[0], *[events[1]] * 3]
    if failure == 'wrong_group': events[2]['payload']['arguments']['group'] = 'group2'
    if failure == 'missing_producer': events.pop(0)
    if failure == 'wrong_producer': events[0]['payload']['arguments']['name'] = 'group2'
    if failure == 'reordered': events[0], events[1] = events[1], events[0]
    if failure == 'changed_business_argument':
        for event in events[1:]: event['payload']['arguments']['extra_business_key'] = 'unplanned'
    assert gate.apply({'outcome': 1}, {'trajectory': {'events': events}})['outcome'] == 0


def test_later_page_result_needed_by_business_action_retains_its_cursor_ancestors():
    calls = reference()
    calls[-1]['capture'] = {'chosen': '$.records[0].id'}
    calls.append({'operation': 'tool_call', 'tool_name': 'write', 'arguments': {'id': {'$ref': 'chosen'}}})
    spec = {'tool_name': 'search', 'operation': 'select', 'pagination': {'cursor_argument': 'cursor', 'page_size': 5}}
    edges = _capability_edges([{'kind': 'goal_success', 'steps': calls}], [spec])
    downstream = next(edge for edge in edges if edge['to_tool'] == 'write')
    assert [p['tool_name'] for p in downstream['capture_plan']] == ['lookup', 'search', 'search', 'search']
    assert downstream['capture_plan'][-1]['arguments']['cursor'] == {'$ref': 'p3'}
    assert 'consumer_transport_arguments' not in downstream
