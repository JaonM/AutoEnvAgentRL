import json
import pytest

from env_factory import sandbox_runtime as runtime


def data(tmp_path, value=None, nullable=True, parent_values=('p1',), target='id'):
    schemas = {
        'parents': {'table_name':'parents','primary_key':['id'], 'columns':[{'name':'id','type':'text','nullable':False}]},
        'children': {'table_name':'children','primary_key':['id'], 'columns':[
            {'name':'id','type':'text','nullable':False}, {'name':'parent_id','type':'text','nullable':nullable}],
            'foreign_keys':[{'column':'parent_id','ref_table':'parents','ref_column':target}]}}
    rows = {'parents':[{'id':v} for v in parent_values], 'children':[{'id':'c1','parent_id':value}]}
    manifest = {'tables':[]}
    for name, schema in schemas.items():
        (tmp_path/f'{name}.schema.json').write_text(json.dumps(schema))
        (tmp_path/f'{name}.rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows[name]))
        manifest['tables'].append({'table_name':name,'schema_file':f'{name}.schema.json','rows_file':f'{name}.rows.jsonl','row_count':len(rows[name])})
    return manifest


@pytest.mark.parametrize('parents',[('p1',), ()])
def test_explicit_nullable_foreign_key_accepts_null(tmp_path, parents):
    manifest=data(tmp_path,parent_values=parents)
    assert runtime.DataManifestValidator.validate(manifest,tmp_path)['tables']['children'][0]['parent_id'] is None


@pytest.mark.parametrize('value',[None,'missing',''])
def test_nonnullable_or_dangling_reference_rejected(tmp_path,value):
    manifest=data(tmp_path,value=value,nullable=value is not None)
    with pytest.raises(runtime.SandboxError,match='foreign key'):
        runtime.DataManifestValidator.validate(manifest,tmp_path)


def test_nonnull_reference_and_schema_still_checked(tmp_path):
    manifest=data(tmp_path,value='p1')
    assert runtime.DataManifestValidator.validate(manifest,tmp_path)['tables']['children'][0]['parent_id']=='p1'
    manifest=data(tmp_path,target='missing')
    with pytest.raises(runtime.SandboxError,match='foreign key'):
        runtime.DataManifestValidator.validate(manifest,tmp_path)
