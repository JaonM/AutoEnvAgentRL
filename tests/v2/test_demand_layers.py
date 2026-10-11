import copy
import pytest
from env_factory_v2.requirements.demand import layered_demand, validate_demand
from env_factory_v2.contracts.contract import compile_contract, ContractError
from env_factory_v2.examples import simple


def test_public_context_preserves_user_wording_and_excludes_internal_graph():
    source = simple()
    source['demand'] = layered_demand(source['task'], '只查对应尺码。', document_id='business.json', graph_evidence={'internal_node': 'n123'})
    compiled = compile_contract(source)
    assert compiled['public']['agent_context']['user_request'] == source['task']
    assert 'internal_node' not in str(compiled['public'])


def test_hidden_requirement_or_changed_wording_cannot_compile():
    for change in ['hidden', 'wording']:
        source = simple()
        source['demand'] = layered_demand(source['task'], '只查对应尺码。', document_id='business.json')
        if change == 'hidden':
            source['demand']['task_spec']['requirements'][1]['text'] = '隐藏额外条件'
        else:
            source['demand']['user_request'] = '不同的需求'
        with pytest.raises(ContractError):
            compile_contract(source)


def granular():
    return layered_demand('先查库存；缺货就不要下单。',
        '必须先授权。拒绝时不写入；成功后核验状态。', document_id='business.json')


def test_clause_citations_preserve_every_public_character_and_do_not_leak_graph():
    demand = granular()
    requirements = demand['task_spec']['requirements']
    assert len(requirements) == 5
    for source, text in demand['agent_context'].items():
        assert ''.join(r['text'] for r in requirements if r['source'] == source) == text
        for requirement in requirements:
            if requirement['source'] == source:
                citation = requirement['citation']
                assert text[citation['start']:citation['end']] == requirement['text']
    assert validate_demand(demand, demand['user_request']) == demand['agent_context']


@pytest.mark.parametrize('change', ['omit_clause', 'invent_text', 'duplicate_id', 'boolean_offset',
                                   'negative_offset', 'outside_range', 'wrong_source'])
def test_citation_validation_rejects_missing_or_untraceable_obligations(change):
    demand = granular()
    requirements = demand['task_spec']['requirements']
    if change == 'omit_clause':
        requirements.pop()
    elif change == 'invent_text':
        requirements[-1]['text'] = '必须调用所有工具。'
    elif change == 'duplicate_id':
        requirements[-1]['id'] = requirements[0]['id']
    elif change == 'boolean_offset':
        requirements[0]['citation']['start'] = False
    elif change == 'negative_offset':
        requirements[0]['citation']['start'] = -1
    elif change == 'outside_range':
        requirements[0]['citation']['end'] = 10000
    else:
        requirements[0]['source'] = 'business_policy'
    with pytest.raises(ContractError):
        validate_demand(demand, demand['user_request'])


def test_legacy_whole_document_sources_still_validate_without_rewriting_frozen_assets():
    demand = granular()
    demand['task_spec']['requirements'] = [
        {'id': source, 'source': source, 'text': text, 'visibility': 'public'}
        for source, text in demand['agent_context'].items()]
    assert validate_demand(demand, demand['user_request']) == demand['agent_context']
