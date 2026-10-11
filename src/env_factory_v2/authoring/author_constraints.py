"""Check supplied author inputs; contains no evaluator answers or business oracle."""
import copy

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.contracts.fixtures import materialize


def check_author_inputs(compiled, request):
    source = compiled['source']
    if 'goal' in request:
        require(source['task'] == request['goal'], 'author task must preserve the original request verbatim')
    if 'category' in request:
        require(source['category'] == request['category'], 'author task category changed')
    if 'private_user_fixture' in request:
        require(canonical(source['user']['facts']) == canonical(request['private_user_fixture']['facts']),
                'author private facts differ from supplied simulator facts')
    business = request.get('business_evidence')
    if not business or 'tables' not in business:
        return
    actual = {table['id']: table for table in source['tables']}
    require(set(actual) == {table['id'] for table in business['tables']}, 'author business table identities changed')
    supplied = []
    for table in business['tables']:
        original = copy.deepcopy(actual[table['id']])
        original.pop('rows', None)
        original.pop('generator', None)
        if 'rows' in table:
            original['rows'] = copy.deepcopy(table['rows'])
        else:
            original['generator'] = copy.deepcopy(table['generator'])
        supplied.append(original)
    materialize(supplied)
    for table in supplied:
        require(sorted(canonical(row) for row in actual[table['id']]['rows']) ==
                sorted(canonical(row) for row in table['rows']),
                f"author business rows changed: {table['id']}")
    for relation in business.get('relations', []):
        require(any(fk['column'] == relation['source_column'] and
                    fk['target_table'] == relation['target_table'] and
                    fk['target_column'] == relation['target_column']
                    for fk in actual[relation['source_table']].get('foreign_keys', [])),
                'author omitted a supplied business relation')
