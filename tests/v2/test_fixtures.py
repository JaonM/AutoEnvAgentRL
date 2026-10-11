import copy

import pytest

from env_factory_v2.contracts.contract import ContractError, compile_contract
from env_factory_v2.examples import retrieval, long_chain
from env_factory_v2.contracts.fixtures import review_contract
from env_factory_v2.validation.verification import qualify


def generated_retrieval():
    source = retrieval(512)
    items = source['tables'][1]
    items.pop('rows')
    items['generator'] = {'count': 1024, 'seed': 0, 'columns': {
        'id': {'kind': 'sequence', 'start': 0, 'step': 1, 'prefix': 'I', 'width': 4},
        'brand_id': {'kind': 'foreign_key', 'table': 'brands', 'column': 'id'},
        'display': {'kind': 'constant', 'value': '青松'}, 'size': {'kind': 'constant', 'value': 'M'},
        'active': {'kind': 'constant', 'value': True}, 'stock': {'kind': 'sequence', 'start': 1, 'step': 1}}}
    return source


def test_bulk_fixture_materializes_all_rows_and_preserves_full_retrieval():
    compiled, report = qualify(generated_retrieval())
    assert len(compiled['source']['tables'][1]['rows']) == 1024
    assert report['gates']['joint_execution_and_counterexamples']
    correct = next(r for r in report['records'] if r['id'] == 'correct')['first']
    assert correct['outcome']['business_attempts'] == 33
    assert len(correct['answer']['ids']) == 1024


def test_generator_plan_is_exact_and_recompilation_rejects_changed_rows():
    compiled = compile_contract(generated_retrieval())
    assert compile_contract(compiled['source'])['identity'] == compiled['identity']
    changed = copy.deepcopy(compiled['source'])
    changed['tables'][1]['rows'][0]['stock'] = 88
    with pytest.raises(ContractError, match='differs from its declared generator'):
        compile_contract(changed)


def test_review_uses_complete_plan_with_concrete_previews_instead_of_bulk_tokens():
    compiled = compile_contract(generated_retrieval())
    review, evidence = review_contract(compiled['source'])
    assert 'rows' not in review['tables'][1]
    assert review['tables'][1]['generator']['count'] == 1024
    assert evidence['items']['row_count'] == 1024
    assert evidence['items']['last_rows'][-1]['id'] == 'I1023'


def test_same_seed_replays_and_changed_seed_changes_declared_cycle():
    source = generated_retrieval()
    source['tables'][1]['generator']['columns']['size'] = {'kind': 'cycle', 'values': ['M', 'L']}
    first = compile_contract(source)
    assert first['identity'] == compile_contract(source)['identity']
    source['tables'][1]['generator']['seed'] = 1
    second = compile_contract(source)
    assert first['source']['tables'][1]['rows'][0]['size'] == 'M'
    assert second['source']['tables'][1]['rows'][0]['size'] == 'L'


def test_repeated_same_tool_consumes_only_earlier_real_results():
    source = long_chain()
    source['tools'] = source['tools'][:2]
    source['tools'][1]['result_bindings'][0]['tool'] = ['node_0', 'node_1']
    for scenario in source['scenarios']:
        for step in scenario['steps'][1:]:
            step['tool'] = 'node_1'
    _, report = qualify(source)
    assert report['gates']['joint_execution_and_counterexamples']
    assert next(r for r in report['records'] if r['id'] == 'correct')['first']['outcome']['business_attempts'] == 12
