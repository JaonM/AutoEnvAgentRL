"""Delivery previews exercise production cursor and episode semantics."""
from copy import deepcopy

import pytest

from env_factory.generation.contract_compiler import ContractCompiler
from env_factory.generation.pipeline_errors import PipelineGenerationError


def preview(steps, count=3):
    return ContractCompiler._preview_success_tool_results(
        scenarios=[{'kind': 'goal_success', 'steps': steps}],
        data_tables=[{'table_name': 'items', 'columns': [
            {'name': 'id', 'type': 'integer', 'nullable': False}],
            'primary_key': ['id'], 'rows': [{'id': i} for i in range(count)]}],
        tool_implementations=[
            {'tool_name': 'search', 'operation': 'select', 'table': 'items',
             'result_field': 'records', 'projection': ['id'], 'order_by': ['id'],
             'pagination': {'cursor_argument': 'cursor', 'page_size': 2}},
            {'tool_name': 'detail', 'operation': 'select', 'table': 'items',
             'result_field': 'records', 'filters': [
                 {'argument': 'id', 'column': 'id', 'operator': 'eq'}]},
            {'tool_name': 'insert', 'operation': 'insert', 'table': 'items',
             'result_field': 'records', 'values': {'id': 'id'}}],
        environment_mode='stateful')


def call(name, arguments=None, capture=None):
    step = {'operation': 'tool_call', 'tool_name': name, 'arguments': arguments or {}}
    if capture:
        step['capture'] = capture
    return step


@pytest.mark.parametrize('count', [0, 1, 2, 3, 5])
def test_preview_paginates_to_exhaustion_and_preserves_later_page_dependencies(count):
    steps = [{'operation': 'reset', 'body': {}}]
    for page in range(max(1, (count + 1) // 2)):
        steps.append(call('search', {'cursor': {'$ref': 'next'}} if page else {},
                          {'next': '$.next_cursor'}))
    if count:
        steps[-1]['capture']['last'] = '$.records[0].id'
        steps.append(call('detail', {'id': {'$ref': 'last'}}))
    result = preview(steps, count)
    pages = [x['result'] for x in result if x['tool_name'] == 'search']
    assert [row['id'] for page in pages for row in page['records']] == list(range(count))
    assert pages[-1]['has_more'] is False
    assert pages[-1]['next_cursor'] == ''
    if count:
        assert result[-1]['result']['records'] == pages[-1]['records'][:1]


@pytest.mark.parametrize('change,error', [('reset', 'another query or episode'),
                                         ('insert', 'query data changed')])
def test_preview_rejects_cursor_after_reset_or_business_mutation(change, error):
    steps = [call('search', capture={'next': '$.next_cursor'}),
             {'operation': 'reset', 'body': {}} if change == 'reset'
             else call('insert', {'id': 10}),
             call('search', {'cursor': {'$ref': 'next'}})]
    with pytest.raises(PipelineGenerationError, match=error):
        preview(steps)


def test_previews_are_isolated_and_still_reject_invalid_writes():
    steps = [call('insert', {'id': 8}), call('search')]
    assert preview(steps)[1]['result']['total_count'] == 4
    assert preview([call('search')])[0]['result']['total_count'] == 3
    wrong = deepcopy(steps)
    wrong[0]['arguments']['id'] = 0
    with pytest.raises(PipelineGenerationError, match='duplicate primary key'):
        preview(wrong)
