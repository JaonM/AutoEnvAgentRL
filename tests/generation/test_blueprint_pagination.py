import copy

import pytest

from env_factory.generation.business_blueprint import validate_blueprint, validate_implementation
from .test_business_blueprint import blueprint, implementation, request


def fixture():
    plan, source, req = blueprint(), implementation(), request()
    paging = {'cursor_argument': 'cursor', 'page_size': 100}
    plan['tools'][1]['pagination'] = paging
    plan['tools'][1]['inputs']['cursor'] = {'kind': 'pagination_cursor'}
    source['tools'][1]['function']['parameters']['properties']['cursor'] = {'type': 'string'}
    source['tool_implementations'][1]['pagination'] = copy.deepcopy(paging)
    steps = source['scenarios'][0]['steps']
    steps[-1]['capture'] = {'page2': '$.next_cursor'}
    steps.append({'operation': 'tool_call', 'tool_name': 'quote',
        'arguments': {'id': {'$ref': 'id'}, 'limit': 10, 'cursor': {'$ref': 'page2'}}})
    return plan, source, req


def test_cursor_origin_does_not_fabricate_a_business_dependency():
    plan, source, req = fixture()
    evidence = validate_implementation(plan, source, req)
    assert evidence['dependencies'] == {'lookup': [], 'quote': ['lookup']}


@pytest.mark.parametrize('change', ['wrong_capture', 'literal', 'renamed_cursor', 'page_size', 'missing_pagination'])
def test_materialization_cannot_change_paging_or_invent_a_cursor(change):
    plan, source, req = fixture()
    if change == 'wrong_capture': source['scenarios'][0]['steps'][-1]['arguments']['cursor'] = {'$ref': 'id'}
    if change == 'literal': source['scenarios'][0]['steps'][-1]['arguments']['cursor'] = 'invented'
    if change == 'renamed_cursor': source['tool_implementations'][1]['pagination']['cursor_argument'] = 'page'
    if change == 'page_size': source['tool_implementations'][1]['pagination']['page_size'] = 99
    if change == 'missing_pagination': source['tool_implementations'][1].pop('pagination')
    with pytest.raises(ValueError, match='BLUEPRINT_DRIFT: pagination'):
        validate_implementation(plan, source, req)


@pytest.mark.parametrize('change', ['oversize', 'mutation', 'not_declared', 'user_stage', 'nested_origin'])
def test_blueprint_requires_a_real_transport_origin(change):
    plan, _, req = fixture()
    tool = plan['tools'][1]
    if change == 'oversize': tool['pagination']['page_size'] = 101
    if change == 'mutation': tool['operation'] = 'update'
    if change == 'not_declared': tool.pop('pagination')
    if change == 'user_stage': plan['interactions'][1]['argument'] = 'cursor'
    if change == 'nested_origin': tool['inputs']['id'] = {'kind': 'derived', 'sources': [{'kind': 'pagination_cursor'}]}
    with pytest.raises(ValueError, match='BLUEPRINT:'):
        validate_blueprint(plan, req)
