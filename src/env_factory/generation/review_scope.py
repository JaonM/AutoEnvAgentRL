"""Check reviewer assumptions against the finite candidate, not its verdict."""
import copy
import json
import re
from .repair_feedback import _location


def counterexample_schema():
    def obj(properties):
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    text={'type':'string'}
    return obj({
        'basis':{'type':'string','enum':['public_contract','declared_script','business_data']},
        'script_assumptions':{'type':'array','items':obj({'variant_index':{'type':'integer'},'stage_id':text,'value_json':text})},
        'data_mutations':{'type':'array','items':obj({'table':text,'row_index':{'type':'integer'},'column':text,'value_json':text})},
        'public_paths':{'type':'array','items':text}})


def validate_review_scope(report, bundle):
    """Validate grounding only; passing this is not execution or semantic proof."""
    from .semantic_review import SemanticReviewUnavailable
    def reject(reason):
        raise SemanticReviewUnavailable('SOURCE_REVIEW_INVALID: counterexample scope: '+reason)
    def parse(value):
        if not isinstance(value,str):reject('value_json must be a JSON string')
        try:
            return json.loads(value,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        except (ValueError,TypeError):reject('invalid assumption JSON')
    variants=(bundle.get('source',{}).get('interaction_contract') or {}).get('variants',[])
    tables={table['table_name']:table for table in bundle.get('source',{}).get('tables',[])}
    for finding in report.get('findings',[]):
        example=finding.get('counterexample')
        if not isinstance(example,dict) or set(example)!={'basis','script_assumptions','data_mutations','public_paths'}:
            reject('each blocking finding requires a structured counterexample')
        basis=example['basis']
        if basis not in {'public_contract','declared_script','business_data'}:reject('unsupported basis')
        if any(not isinstance(example[key],list) for key in ('script_assumptions','data_mutations','public_paths')):
            reject('assumptions, mutations and public paths must be lists')
        for assumption in example['script_assumptions']:
            if not isinstance(assumption,dict) or set(assumption)!={'variant_index','stage_id','value_json'}:
                reject('invalid script assumption')
            index=assumption['variant_index']
            if type(index) is not int or not 0<=index<len(variants):reject('unknown script variant')
            stages=[stage for stage in variants[index]['stages'] if stage['id']==assumption['stage_id']]
            if len(stages)!=1:reject('unknown or ambiguous disclosure stage')
            actual=stages[0].get('private_fact');claimed=parse(assumption['value_json'])
            if type(actual) is not type(claimed) or actual!=claimed:
                reject('assumed disclosure is not implemented in the selected script')
        if len({a['variant_index'] for a in example['script_assumptions']}) > 1:
            reject('one counterexample cannot mix different script episodes')
        if basis == 'declared_script' and example['data_mutations']:
            reject('data mutations require the business_data basis')
        mutated = {name: copy.deepcopy(table.get('rows', [])) for name, table in tables.items()}
        touched = set()
        for mutation in example['data_mutations']:
            if not isinstance(mutation,dict) or set(mutation)!={'table','row_index','column','value_json'}:
                reject('invalid business mutation')
            table=tables.get(mutation['table']);index=mutation['row_index']
            if table is None or type(index) is not int or not 0<=index<len(table.get('rows',[])):
                reject('unknown business row; user scripts are not business data')
            if mutation['column'] not in {column['name'] for column in table['columns']}:
                reject('unknown business column')
            address = (mutation['table'], index, mutation['column'])
            if address in touched:reject('duplicate business mutation')
            touched.add(address)
            mutated[mutation['table']][index][mutation['column']] = parse(mutation['value_json'])
        if example['data_mutations']:
            from ..sandbox_runtime import ManifestDataStore, SandboxError
            # Validation uses only schemas and supplied rows; no store or episode is opened.
            validator = object.__new__(ManifestDataStore)
            validator.schemas = tables
            try:
                validator._validate_tables(mutated)
            except SandboxError as exc:
                reject('business mutation violates runtime data constraints: '+str(exc))
        for path in example['public_paths']:
            # The shared resolver accepts both list-index spellings. Apply the
            # same spelling rule before checking the public-root boundary.
            if isinstance(path, str):
                path = re.sub(r'^compiled_tools\.(\d+)(?=\.|\[|$)', r'compiled_tools[\1]', path)
            allowed = ('compiled_public_input.', 'compiled_tools[', 'source.description.',
                       'original_goal.public_input.')
            if not isinstance(path, str) or not path.startswith(allowed):
                reject('public evidence path has an unsupported root: ' + json.dumps(path, ensure_ascii=False)
                       + '; use a concrete path under ' + ', '.join(allowed))
            if not _location(bundle, path)['resolved']:
                # A bounded report repair needs the actual bad address and
                # available shape, not a generic error that repeats unchanged.
                parent = path
                hint = ''
                while parent:
                    shorter = re.sub(r'(?:\.[^.\[\]]+|\[(?:\d+|\*)\])$', '', parent)
                    if shorter == parent:
                        break
                    parent = shorter
                    found = _location(bundle, parent)
                    if found['resolved']:
                        value = found['value']
                        if isinstance(value, dict):
                            hint = '; nearest resolved parent ' + parent + ' has keys ' + json.dumps(list(value)[:20])
                        elif isinstance(value, list):
                            hint = '; nearest resolved parent ' + parent + ' has length ' + str(len(value))
                        break
                    if '.' not in parent and '[' not in parent:
                        break
                reject('public evidence path does not resolve: ' + json.dumps(path, ensure_ascii=False) + hint
                       + '; inspect the unchanged candidate and cite an existing public field; do not guess aliases')
        if basis=='public_contract' and not example['public_paths']:reject('public conflict needs public evidence')
        if basis=='declared_script' and not example['script_assumptions']:reject('script counterexample needs an actual disclosure')
        if basis=='business_data' and not example['data_mutations']:reject('business counterexample needs explicit data changes')
        if basis == 'business_data' and all(
                mutated[name] == table.get('rows', []) for name, table in tables.items()):
            reject('business mutations leave every row unchanged; this does not establish an alternative '
                   'business state. Reassess the claim using actual lifecycle events and script choices, '
                   'or supply a genuinely changed, schema-valid counterexample. Do not invent a change '
                   'merely to preserve the failed verdict.')
