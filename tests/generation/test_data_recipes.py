import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.data_recipes import expand_data_recipes
from env_factory.generation.contract_compiler import ContractCompiler
from .test_answer_fields import source as field_source


def source():
    value, request = field_source()
    table = value['tables'][0]
    table['row_recipe'] = {'count': 10000, 'columns': {
        'id': {'sequence': {'start': 1000, 'step': 1}},
        'name': {'string_sequence': {'prefix': '其他库区-', 'start': 0, 'step': 1, 'suffix': ''}},
        'temperature': {'integer': {'min': -10, 'max': 40}},
        'current': {'seeded_choice': [True, False]}}}
    request['seed'] = 73
    return value, request


def test_recipe_is_reproducible_seeded_and_preserves_explicit_business_rows():
    value, request = source(); original = copy.deepcopy(value)
    a = expand_data_recipes(value, seed=request['seed'])
    b = expand_data_recipes(value, seed=request['seed'])
    c = expand_data_recipes(value, seed=74)
    assert value == original and a == b and a != c
    assert len(a['tables'][0]['rows']) == 10002
    assert a['tables'][0]['rows'][-2:] == value['tables'][0]['rows']
    assert a['_data_recipe_provenance']['zones']['expanded_rows_sha256'] != c['_data_recipe_provenance']['zones']['expanded_rows_sha256']
    ContractCompiler._validate_relational_data(a['tables'])


def test_compiler_materializes_ten_thousand_rows_with_compact_recipe_provenance(tmp_path):
    value, request = source()
    artifact = compile_source(value, root=tmp_path, request=request, structural_preview=True)
    manifest = artifact['data_manifest']
    assert manifest['tables'][0]['row_count'] == 10002
    provenance = manifest['row_recipes']['zones']
    assert provenance['seed'] == 73 and provenance['expanded_row_count'] == 10002
    assert len(json.dumps(provenance)) < 2000
    rows = [json.loads(line) for line in (tmp_path / manifest['root'] / manifest['tables'][0]['rows_file']).read_text().splitlines()]
    assert rows == expand_data_recipes(value, seed=73)['tables'][0]['rows']
    from env_factory.sandbox_runtime import DataManifestValidator, SandboxError
    DataManifestValidator.validate(manifest, tmp_path / manifest['root'])
    row_path = tmp_path / manifest['root'] / manifest['tables'][0]['rows_file']
    rows[0]['temperature'] += 1
    row_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    with pytest.raises(SandboxError, match='row recipe data identity differs'):
        DataManifestValidator.validate(manifest, tmp_path / manifest['root'])


@pytest.mark.parametrize('change', ['oversize', 'bool_count', 'missing_column', 'code', 'nan', 'empty_choice', 'zero_step', 'spoof_provenance', 'total_cap'])
def test_invalid_recipes_fail_before_expansion(change):
    value, _ = source(); recipe = value['tables'][0]['row_recipe']
    if change == 'oversize': recipe['count'] = 100001
    if change == 'bool_count': recipe['count'] = True
    if change == 'missing_column': recipe['columns'].pop('current')
    if change == 'code': recipe['columns']['temperature'] = {'python': 'import os'}
    if change == 'nan': recipe['columns']['temperature'] = {'literal': float('nan')}
    if change == 'empty_choice': recipe['columns']['current'] = {'seeded_choice': []}
    if change == 'zero_step': recipe['columns']['id']['sequence']['step'] = 0
    if change == 'spoof_provenance': value['_data_recipe_provenance'] = {}
    if change == 'total_cap': recipe['count'] = 100000
    with pytest.raises(ValueError, match='DATA_RECIPE_INVALID'): expand_data_recipes(value, seed=73)


def test_expanded_primary_key_collision_still_fails_normal_schema_validation(tmp_path):
    value, request = source()
    value['tables'][0]['row_recipe']['columns']['id'] = {'literal': 821}
    with pytest.raises(ValueError, match='duplicate|primary'):
        compile_source(value, root=tmp_path, request=request, structural_preview=True)
