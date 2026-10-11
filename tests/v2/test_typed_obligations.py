import copy
import pytest

from env_factory_v2.contracts.business_definition import compile_definition
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.examples import direct, simple, retrieval, transaction, long_chain, complex_transaction
from test_business_definition import split, report_definition


def typed(task):
    task = copy.deepcopy(task)
    obligations = []
    for section, kind in [('answers', 'answer'), ('reports', 'report'), ('messages', 'communication'),
                          ('goals', 'state'), ('success', 'process')]:
        for item in task.pop(section, []):
            definition = {k: v for k, v in item.items() if k != 'id'}
            obligations.append({'id': item['id'], 'kind': kind,
                'citations': [{'source': 'user_request', 'quote': task['request']}],
                'definition': definition})
    task['obligations'] = obligations
    return task


@pytest.mark.parametrize('factory', [direct, simple, retrieval, transaction, long_chain, complex_transaction])
def test_typed_lowering_preserves_existing_contract_across_task_difficulties(factory):
    environment, task = split(factory())
    before, _ = compile_definition(environment, task)
    converted = typed(task)
    after, manifest = compile_definition(environment, converted)
    assert before['identity'] == after['identity']
    assert len(manifest['obligation_bindings']) == len(converted['obligations'])
    assert all(entry['citations'] for entry in manifest['obligation_bindings'])


def test_derived_report_still_uses_shared_query_rules():
    environment, task = report_definition()
    before, _ = compile_definition(environment, task)
    after, manifest = compile_definition(environment, typed(task))
    assert before['identity'] == after['identity']
    assert manifest['derived_reports'] == ['ids']


@pytest.mark.parametrize('mutation', ['duplicate_owner', 'fabricated_citation', 'missing_citation', 'wrong_kind'])
def test_typed_obligations_reject_ambiguous_or_ungrounded_definitions(mutation):
    environment, task = split(simple())
    task = typed(task)
    if mutation == 'duplicate_owner':
        task['answers'] = []
    elif mutation == 'fabricated_citation':
        task['obligations'][0]['citations'][0]['quote'] = 'not in original request'
    elif mutation == 'missing_citation':
        task['obligations'][0]['citations'] = []
    else:
        task['obligations'][0]['kind'] = 'state'
    with pytest.raises(ContractError):
        compile_definition(environment, task)
