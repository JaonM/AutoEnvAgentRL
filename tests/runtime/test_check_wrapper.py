import pytest

from env_factory.generation.contract_compiler import ContractCompiler
from env_factory.generation.pipeline_errors import PipelineGenerationError
from env_factory.sandbox_runtime import EpisodeStore, ManifestDataStore, SandboxError


@pytest.mark.parametrize('expression', ['quantity >= 0 AND quantity <= 10',
    'CHECK (quantity >= 0 AND quantity <= 10)', ' check (quantity >= 0 AND quantity <= 10) '])
def test_check_wrapper_means_same_predicate_in_planning_fixture_and_runtime(tmp_path, expression):
    table = {'table_name': 'inventory', 'description': 'Stock', 'primary_key': ['id'],
        'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'quantity', 'type': 'integer', 'nullable': False}],
        'constraints': [{'type': 'CHECK', 'name': 'valid_quantity', 'expression': expression}],
        'rows': [{'id': 1, 'quantity': 10}], 'foreign_keys': [], 'indexes': []}
    ContractCompiler._validate_table_definitions([table])
    ContractCompiler._validate_relational_data([table])
    manifest = ContractCompiler._materialize_business_data([table], '', tmp_path / 'data')
    store = EpisodeStore(tmp_path / 'state.sqlite3')
    data = ManifestDataStore(manifest, tmp_path / 'data', store)
    data.reset()
    assert data.update('inventory', {'id': 1}, {'quantity': 4}) == 1
    for invalid in (-1, 11):
        with pytest.raises(SandboxError, match='CHECK constraint failed'):
            data.update('inventory', {'id': 1}, {'quantity': invalid})
        assert data.table('inventory')[0]['quantity'] == 4
        table['rows'][0]['quantity'] = invalid
        with pytest.raises(PipelineGenerationError, match='violates CHECK'):
            ContractCompiler._validate_relational_data([table])


@pytest.mark.parametrize('expression', ['CHECK (quantity >= 0) garbage', 'CHECK (abs(quantity) > 0)',
    'CHECK (unknown >= 0)', 'CHECK (quantity >= 0); DROP TABLE inventory'])
def test_wrapper_does_not_accept_unsupported_or_unknown_predicates(expression):
    table = {'table_name': 'inventory', 'columns': [{'name': 'quantity', 'type': 'integer'}],
             'primary_key': ['quantity'], 'constraints': [expression]}
    with pytest.raises(PipelineGenerationError):
        ContractCompiler._validate_table_definitions([table])
