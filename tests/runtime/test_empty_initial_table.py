import pytest

from env_factory.generation.contract_compiler import ContractCompiler
from env_factory.generation.pipeline_errors import PipelineGenerationError
from env_factory.sandbox_runtime import (
    BusinessGoalEvaluator, DeclarativeToolCompiler, EpisodeStore, ManifestDataStore, SandboxError,
)


def test_empty_destination_materializes_and_requires_real_valid_insert(tmp_path):
    parent = {'table_name': 'customers', 'description': 'Customers', 'primary_key': ['id'],
              'columns': [{'name': 'id', 'type': 'integer', 'nullable': False}],
              'rows': [{'id': 7}], 'foreign_keys': [], 'indexes': [], 'constraints': []}
    child = {'table_name': 'bookings', 'description': 'Bookings', 'primary_key': ['id'],
             'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                         {'name': 'customer', 'type': 'integer', 'nullable': False}],
             'rows': [], 'foreign_keys': [{'column': 'customer', 'ref_table': 'customers', 'ref_column': 'id'}],
             'indexes': [], 'constraints': []}
    ContractCompiler._validate_relational_data([parent, child])
    manifest = ContractCompiler._materialize_business_data([parent, child], '', tmp_path / 'data')
    store = EpisodeStore(tmp_path / 'state.sqlite3')
    data = ManifestDataStore(manifest, tmp_path / 'data', store)
    data.reset()
    assert data.table('bookings') == []
    goal = [{'table': 'bookings', 'where': {'id': 1}, 'values': {'customer': 7}, 'count': 1}]
    assert not BusinessGoalEvaluator.evaluate(goal, data._all(), data.baseline)
    create = DeclarativeToolCompiler(data).compile({'tool_name': 'create', 'operation': 'insert',
        'table': 'bookings', 'values': {'id': 'id', 'customer': 'customer'}})
    with pytest.raises(SandboxError): create({'id': 1, 'customer': 9})
    assert data.table('bookings') == []
    assert not BusinessGoalEvaluator.evaluate(goal, data._all(), data.baseline)
    assert create({'id': 1, 'customer': 7})['record'] == {'id': 1, 'customer': 7}
    assert BusinessGoalEvaluator.evaluate(goal, data._all(), data.baseline)
    with pytest.raises(SandboxError): create({'id': 1, 'customer': 7})
    assert data.table('bookings') == [{'id': 1, 'customer': 7}]
    # Empty tables do not exempt declared foreign keys or missing fixture data.
    parent['rows'] = []
    child['rows'] = [{'id': 1, 'customer': 7}]
    with pytest.raises(PipelineGenerationError, match='missing foreign values'):
        ContractCompiler._validate_relational_data([parent, child])
    child['rows'] = []
    ContractCompiler._validate_relational_data([parent, child])
    child.pop('rows')
    with pytest.raises(PipelineGenerationError, match='explicit rows list'):
        ContractCompiler._validate_relational_data([parent, child])
