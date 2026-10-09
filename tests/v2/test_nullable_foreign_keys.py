import copy

import pytest

from env_factory_v2.contract import ContractError, validate_data


def chain(nullable=True):
    return {'tables': [{'id': 'nodes', 'primary_key': 'id',
                        'schema': {'type': 'object', 'additionalProperties': False,
                                   'required': ['id', 'next_id'],
                                   'properties': {'id': {'type': 'integer'},
                                                  'next_id': {'type': 'integer', 'nullable': nullable}}},
                        'foreign_keys': [{'column': 'next_id', 'target_table': 'nodes',
                                          'target_column': 'id'}]}]}


def test_nullable_terminal_preserves_real_nonnull_relationship_validation():
    data = {'nodes': [{'id': 1, 'next_id': 2}, {'id': 2, 'next_id': None}]}
    validate_data(chain(), data)
    dangling = copy.deepcopy(data)
    dangling['nodes'][0]['next_id'] = 3
    with pytest.raises(ContractError, match='foreign key violation'):
        validate_data(chain(), dangling)
    with pytest.raises(ContractError, match='expected integer'):
        validate_data(chain(False), data)


def test_empty_target_allows_absent_relationship_but_not_fabricated_one():
    contract = chain()
    contract['tables'].append(copy.deepcopy(contract['tables'][0]))
    contract['tables'][1]['id'] = 'targets'
    contract['tables'][1]['foreign_keys'] = []
    contract['tables'][0]['foreign_keys'][0]['target_table'] = 'targets'
    data = {'nodes': [{'id': 1, 'next_id': None}], 'targets': []}
    validate_data(contract, data)
    data['nodes'][0]['next_id'] = 1
    with pytest.raises(ContractError, match='foreign key violation'):
        validate_data(contract, data)
