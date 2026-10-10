import copy
import pytest
from env_factory_v2.demand import layered_demand
from env_factory_v2.contract import compile_contract, ContractError
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
