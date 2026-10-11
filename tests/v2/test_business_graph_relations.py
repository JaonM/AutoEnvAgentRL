import copy
import pytest
from env_factory_v2.requirements.business_graph import validate_relations
from env_factory_v2.contracts.expressions import ContractError


def dossier():
    return {'policy': '申请的入口指向审批节点；终点可为空。', 'tables': [
        {'id': 'applications', 'rows': [{'id': 'A', 'entry_id': 'N'}]},
        {'id': 'nodes', 'rows': [{'id': 'N', 'next_id': None}]}], 'relations': [
        {'id': 'entry', 'source_table': 'applications', 'source_column': 'entry_id',
         'target_table': 'nodes', 'target_column': 'id', 'policy_text': '申请的入口指向审批节点', 'nullable': False},
        {'id': 'next', 'source_table': 'nodes', 'source_column': 'next_id',
         'target_table': 'nodes', 'target_column': 'id', 'policy_text': '终点可为空', 'nullable': True}]}


def test_explicit_relations_validate_without_mutating_dossier():
    business = dossier()
    before = copy.deepcopy(business)
    assert validate_relations(business) == business['relations']
    assert business == before


@pytest.mark.parametrize('change', ['missing_record', 'missing_column', 'unproven_policy', 'duplicate_target', 'forbidden_null'])
def test_invalid_business_relation_rejected(change):
    business = dossier()
    if change == 'missing_record': business['tables'][0]['rows'][0]['entry_id'] = 'unknown'
    if change == 'missing_column': business['relations'][0]['source_column'] = 'unknown'
    if change == 'unproven_policy': business['relations'][0]['policy_text'] = 'invented relation'
    if change == 'duplicate_target': business['tables'][1]['rows'].append({'id': 'N', 'next_id': None})
    if change == 'forbidden_null': business['relations'][1]['nullable'] = False
    with pytest.raises(ContractError): validate_relations(business)


def test_relation_checks_full_generated_rows_and_preserves_generator():
    business = dossier()
    business['tables'][0] = {'id': 'applications', 'generator': {'count': 3, 'seed': 0,
        'columns': {'id': {'kind': 'sequence', 'start': 1, 'step': 1, 'prefix': 'A'},
                    'entry_id': {'kind': 'cycle', 'values': ['N', 'N', 'missing']}}}}
    before = copy.deepcopy(business)
    with pytest.raises(ContractError, match='missing record'):
        validate_relations(business)
    assert business == before
    business['tables'][0]['generator']['columns']['entry_id']['values'] = ['N']
    assert len(validate_relations(business)) == 2
