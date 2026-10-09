"""Validate and compile a single versioned business/user/reward contract."""
from __future__ import annotations

import copy
import hashlib
import math
from datetime import datetime

from .expressions import ContractError, canonical, refs, validate_expr


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), 'expected object')
    require(set(required) <= set(value) <= set(required) | set(optional),
            f'fields mismatch: required={required}, received={list(value)}')


def named(items, label):
    require(isinstance(items, list), f'{label} must be a list')
    result = {}
    for item in items:
        require(isinstance(item, dict) and isinstance(item.get('id'), str)
                and item['id'] and '.' not in item['id'], f'invalid {label} id')
        require(item['id'] not in result, f'duplicate {label}: {item["id"]}')
        result[item['id']] = item
    return result


def validate_schema(schema, depth=0):
    require(depth < 16, 'schema too deep')
    fields(schema, ['type'], ['properties', 'required', 'additionalProperties', 'items', 'enum', 'nullable'])
    require(schema['type'] in {'string', 'integer', 'number', 'boolean', 'object', 'array'}, 'unsupported type')
    if schema['type'] == 'object':
        require(schema.get('additionalProperties') is False, 'objects must forbid extra fields')
        require(isinstance(schema.get('properties'), dict) and isinstance(schema.get('required'), list), 'object schema incomplete')
        require(set(schema['required']) <= set(schema['properties']), 'unknown required field')
        require(len(set(schema['required'])) == len(schema['required'])
                and all(isinstance(key, str) and key and '.' not in key for key in schema['properties']), 'invalid schema field names')
        for item in schema['properties'].values():
            validate_schema(item, depth + 1)
    if schema['type'] == 'array':
        require('items' in schema, 'array items required')
        validate_schema(schema['items'], depth + 1)


def validate_value(value, schema, label='value'):
    if value is None and schema.get('nullable') is True:
        return
    kind = schema['type']
    valid = {'string': isinstance(value, str), 'integer': type(value) is int,
             'number': type(value) in {int, float}, 'boolean': type(value) is bool,
             'object': isinstance(value, dict), 'array': isinstance(value, list)}[kind]
    require(valid, f'{label}: expected {kind}')
    if kind == 'number':
        require(math.isfinite(value), f'{label}: finite number required')
    if 'enum' in schema:
        require(any(canonical(value) == canonical(item) for item in schema['enum']), f'{label}: outside enum')
    if kind == 'object':
        require(set(schema['required']) <= set(value) <= set(schema['properties']), f'{label}: fields mismatch')
        for key, item in value.items():
            validate_value(item, schema['properties'][key], f'{label}.{key}')
    if kind == 'array':
        require(len(value) <= 10000, f'{label}: array limit')
        for item in value:
            validate_value(item, schema['items'], label)


def validate_data(contract, data):
    tables = named(contract['tables'], 'table')
    require(set(data) == set(tables), 'state table identity mismatch')
    for name, table in tables.items():
        rows = data[name]
        require(isinstance(rows, list) and len(rows) <= 10000, 'table row limit')
        keys = set()
        for row in rows:
            validate_value(row, table['schema'], name)
            key = canonical(row[table['primary_key']])
            require(row[table['primary_key']] is not None and key not in keys, f'{name}: duplicate/null primary key')
            keys.add(key)
        for fk in table.get('foreign_keys', []):
            targets = {canonical(row[fk['target_column']]) for row in data[fk['target_table']]}
            for row in rows:
                # Schema validation above already rejects undeclared nulls. A
                # nullable FK means no relationship, not a reference to a null PK.
                if row[fk['column']] is None:
                    continue
                require(canonical(row[fk['column']]) in targets, f'{name}: foreign key violation')


def validate_rule_links(expression, rules, *, shared_filters=False):
    if not isinstance(expression, dict) or set(expression) == {'literal'}:
        return
    if 'rule' in expression:
        invocation = expression['rule']
        require(invocation['id'] in rules, 'unknown rule in expression')
        arguments = {ref.split('.')[1] for ref in refs(rules[invocation['id']]['predicate']) if ref.startswith('args.')}
        require(set(invocation['arguments']) == arguments, 'public rule argument mismatch')
    if 'filter' in expression and shared_filters:
        def shared(predicate):
            if set(predicate) == {'rule'}:
                return True
            if set(predicate) <= {'and', 'or', 'not'}:
                return all(shared(item) for values in predicate.values() for item in values)
            return False
        require(shared(expression['filter']['value']), 'reward eligibility filters must invoke named public rules')
    for child in expression_children(expression):
        validate_rule_links(child, rules, shared_filters=shared_filters)


def expression_children(expression):
    op, value = next(iter(expression.items()))
    if op in {'literal', 'ref'}:
        return []
    if op == 'rule':
        return value['arguments'].values()
    return value.values() if isinstance(value, dict) else value


def contains_rule(expression):
    return 'rule' in expression or any(contains_rule(child) for child in expression_children(expression))


def compile_contract(source):
    source = copy.deepcopy(source)
    try:
        require(len(canonical(source).encode()) <= 4 * 1024 * 1024, 'contract size limit')
    except (ValueError, TypeError) as exc:
        raise ContractError('contract must contain finite JSON data') from exc
    fields(source, ['version', 'id', 'category', 'task', 'rules', 'tables', 'tools', 'user',
                    'answers', 'scenarios', 'probes', 'limits', 'time'], ['explanations', 'invariants', 'events', 'success', 'goals'])
    require(source['version'] == '2.0', 'unsupported contract version')
    require(isinstance(source['id'], str) and source['id'], 'contract id required')
    require(source['category'] in {'direct_response', 'simple_agentic', 'multi_step_agentic'}, 'unsupported category')
    require(isinstance(source['task'], str) and source['task'].strip(), 'public task required')
    fields(source['user'], ['facts', 'stages'])
    require(isinstance(source['user']['facts'], dict), 'private facts must be an object')
    named(source['user']['stages'], 'user stage')
    rules = named(source['rules'], 'rule')
    for rule in rules.values():
        fields(rule, ['id', 'public', 'predicate'])
        require(isinstance(rule['public'], str) and rule['public'].strip(), 'public rule required')
        validate_expr(rule['predicate'], roots={'row', 'args', 'data', 'initial', 'user', 'time'})
        require(not contains_rule(rule['predicate']), 'public predicates cannot recursively invoke other rules')
    tables = named(source['tables'], 'table')
    for table in tables.values():
        fields(table, ['id', 'schema', 'primary_key'], ['rows', 'generator', 'foreign_keys'])
        require('rows' in table or 'generator' in table, 'fixture rows or generator required')
        validate_schema(table['schema'])
        require(table['schema']['type'] == 'object' and table['primary_key'] in table['schema']['required'], 'required primary key missing')
        for fk in table.get('foreign_keys', []):
            fields(fk, ['column', 'target_table', 'target_column'])
            require(fk['target_table'] in tables and fk['column'] in table['schema']['properties']
                    and fk['target_column'] in tables[fk['target_table']]['schema']['properties'], 'invalid foreign key')
    from .fixtures import materialize
    materialize(source['tables'])
    validate_data(source, {key: table['rows'] for key, table in tables.items()})
    require(len(canonical(source).encode()) <= 4 * 1024 * 1024, 'materialized contract size limit')
    tools = named(source['tools'], 'tool')
    for tool in tools.values():
        common = ['id', 'kind', 'description', 'parameters', 'rule_ids']
        optional = ['user_bindings', 'result_bindings', 'authorization', 'complete_retrieval']
        if tool.get('kind') == 'query':
            fields(tool, common + ['table', 'columns', 'order_by', 'page_size'], optional)
            require(tool['table'] in tables, 'unknown query table')
            cols = tables[tool['table']]['schema']['properties']
            require(isinstance(tool['columns'], list) and len(set(tool['columns'])) == len(tool['columns'])
                    and set(tool['columns']) <= set(cols), 'unknown/duplicate projection')
            require(isinstance(tool['order_by'], list), 'order_by must be a list')
            for order in tool['order_by']:
                fields(order, ['field', 'direction'])
                require(order['field'] in cols and order['direction'] in {'asc', 'desc'}, 'invalid order')
            require(type(tool['page_size']) is int and 1 <= tool['page_size'] <= 128, 'page size limit')
        elif tool.get('kind') == 'transaction':
            fields(tool, common + ['operations'], optional)
            require(isinstance(tool['operations'], list) and 1 <= len(tool['operations']) <= 64, 'transaction operation limit')
            for operation in tool['operations']:
                fields(operation, ['table', 'key', 'set'])
                require(operation['table'] in tables, 'unknown write table')
                require(isinstance(operation['set'], dict) and operation['set'], 'empty write')
                require(set(operation['set']) <= set(tables[operation['table']]['schema']['properties'])
                        and tables[operation['table']]['primary_key'] not in operation['set'], 'invalid write columns')
                validate_expr(operation['key'], roots={'args', 'data', 'user', 'time'})
                for expression in operation['set'].values():
                    validate_expr(expression, roots={'args', 'row', 'data', 'user', 'time'})
        else:
            raise ContractError('unsupported tool kind; add a reviewed platform capability')
        validate_schema(tool['parameters'])
        require(tool['parameters']['type'] == 'object', 'tool parameters must be an object')
        require(isinstance(tool['rule_ids'], list) and set(tool['rule_ids']) <= set(rules), 'unknown public rule')
        for binding in tool.get('result_bindings', []):
            fields(binding, ['argument', 'tool', 'path'])
            providers = binding['tool'] if isinstance(binding['tool'], list) else [binding['tool']]
            require(providers and all(isinstance(provider, str) and provider in tools for provider in providers), 'invalid dependency')
            require(binding['argument'] in tool['parameters']['required'], 'dependency argument must be required')
        for argument, fact in tool.get('user_bindings', {}).items():
            available = set(source['user'].get('facts', {})) | {
                stage.get('as_fact', stage.get('fact')) for stage in source['user'].get('stages', [])}
            require(argument in tool['parameters']['required'] and isinstance(fact, str) and fact in available, 'invalid private fact binding')
        for rule_id in tool['rule_ids']:
            for reference in refs(rules[rule_id]['predicate']):
                if reference.startswith('args.'):
                    require(reference.split('.')[1] in tool['parameters']['properties'], 'rule uses undeclared parameter')
        expressions = [rules[r]['predicate'] for r in tool['rule_ids']]
        for operation in tool.get('operations', []):
            expressions.extend([operation['key'], *operation['set'].values()])
        consumed = {reference.split('.')[1] for expression in expressions for reference in refs(expression)
                    if reference.startswith('args.')}
        consumed.update(tool.get('user_bindings', {}))
        consumed.update(b['argument'] for b in tool.get('result_bindings', []))
        require(set(tool['parameters']['properties']) <= consumed, 'tool parameter not consumed by a rule or real dependency')
    fields(source['user'], ['facts', 'stages'])
    require(isinstance(source['user']['facts'], dict), 'private facts must be an object')
    stages = named(source['user']['stages'], 'user stage')
    seen = set()
    for stage in stages.values():
        fields(stage, ['id', 'kind', 'depends_on'], ['fact', 'as_fact', 'source_tool', 'source_path', 'value_field', 'fallback', 'tool'])
        require(isinstance(stage['depends_on'], list) and set(stage['depends_on']) <= seen, 'user dependencies must be ordered and acyclic')
        if stage['kind'] in {'fact', 'choice', 'correction'}:
            require(stage.get('fact') in source['user']['facts'], 'unknown private fact')
            require(isinstance(stage.get('as_fact', stage['fact']), str) and stage.get('as_fact', stage['fact']), 'invalid disclosed fact name')
        if stage['kind'] == 'choice':
            require(stage.get('source_tool') in tools and stage.get('fallback') in {'refuse', 'first_available'}, 'invalid dynamic choice')
            require(isinstance(stage.get('source_path'), str) and isinstance(stage.get('value_field'), str), 'choice source required')
        elif stage['kind'] == 'authorization':
            require(stage.get('tool') in tools and tools[stage['tool']]['kind'] == 'transaction', 'authorization must protect a write')
            require(type(source['user']['facts'].get(f'approve_{stage["id"]}')) is bool, 'explicit authorization preference required')
        else:
            require(stage['kind'] in {'fact', 'correction'}, 'unsupported user stage')
        seen.add(stage['id'])
    for tool in tools.values():
        if tool['kind'] == 'transaction':
            require(tool.get('authorization') in stages
                    and stages[tool['authorization']].get('tool') == tool['id'], 'write authorization required')
    answers = named(source['answers'], 'answer')
    require(bool(answers), 'answer contract required')
    for field in answers.values():
        fields(field, ['id', 'meaning', 'expected', 'weight', 'comparison', 'schema'])
        require(field['comparison'] in {'exact', 'unordered'}, 'unsupported answer comparison')
        require(isinstance(field['meaning'], str) and field['meaning'].strip(), 'answer meaning required')
        require(type(field['weight']) in {int, float} and 0 < field['weight'] <= 1, 'invalid answer weight')
        validate_schema(field['schema'])
        validate_expr(field['expected'], roots={'data', 'initial', 'evidence', 'user', 'time'})
        validate_rule_links(field['expected'], rules, shared_filters=True)
    explanations = named(source.get('explanations', []), 'explanation')
    require(not (set(answers) & set(explanations)), 'duplicate answer/explanation id')
    for explanation in explanations.values():
        fields(explanation, ['id', 'meaning', 'criteria', 'context_fields', 'weight'])
        require(type(explanation['weight']) in {int, float} and 0 < explanation['weight'] <= 1, 'invalid explanation weight')
        require(isinstance(explanation['context_fields'], list)
                and set(explanation['context_fields']) <= set(answers), 'unknown explanation evidence')
        require(bool(explanation['criteria']), 'explanation criteria required')
        require(isinstance(explanation['meaning'], str) and explanation['meaning'].strip(), 'explanation meaning required')
        for criterion in named(explanation['criteria'], 'criterion').values():
            fields(criterion, ['id', 'rule_id', 'obligation'])
            require(criterion['rule_id'] in rules and isinstance(criterion['obligation'], str)
                    and criterion['obligation'].strip(), 'invalid reasoning obligation')
    require(abs(sum(f['weight'] for f in [*answers.values(), *explanations.values()]) - 1) < 1e-9, 'reward weights must sum to one')
    for invariant in source.get('invariants', []):
        fields(invariant, ['rule_id', 'predicate'])
        require(invariant['rule_id'] in rules, 'invariant must reference a public rule')
        validate_expr(invariant['predicate'], roots={'data', 'initial', 'user', 'time'})
        validate_rule_links(invariant['predicate'], rules, shared_filters=True)
    for goal in named(source.get('goals', []), 'state goal').values():
        fields(goal, ['id', 'rule_ids', 'predicate'])
        require(bool(goal['rule_ids']) and set(goal['rule_ids']) <= set(rules), 'state goal rule provenance required')
        validate_expr(goal['predicate'], roots={'data', 'initial', 'evidence', 'user', 'time'})
        validate_rule_links(goal['predicate'], rules, shared_filters=True)
    require(not any(t['kind'] == 'transaction' for t in tools.values()) or source.get('goals'), 'writes require actual-state goals')
    fields(source['limits'], ['max_actions'])
    require(type(source['limits']['max_actions']) is int and 1 <= source['limits']['max_actions'] <= 4096, 'invalid action budget')
    fields(source['time'], ['base', 'seconds_per_action'])
    require(isinstance(source['time']['base'], str) and datetime.fromisoformat(source['time']['base']).tzinfo is not None, 'timezone-aware base time required')
    require(type(source['time']['seconds_per_action']) in {int, float}
            and 0 <= source['time']['seconds_per_action'] <= 86400, 'invalid virtual time increment')
    for event in source.get('events', []):
        fields(event, ['at_seconds', 'table', 'key', 'set'])
        require(type(event['at_seconds']) in {int, float} and event['at_seconds'] >= 0 and event['table'] in tables, 'invalid timed event')
        require(set(event['set']) <= set(tables[event['table']]['schema']['properties']), 'unknown event field')
        require(tables[event['table']]['primary_key'] not in event['set'], 'events cannot change primary keys')
    scenarios = named(source['scenarios'], 'scenario')
    require(any(s.get('expect_success') is True and not s.get('mutations') for s in scenarios.values())
            and any(s.get('expect_success') is False for s in scenarios.values()), 'positive and negative scenarios required')
    for scenario in scenarios.values():
        fields(scenario, ['id', 'steps', 'answer', 'expect_success'], ['mutations', 'expect_error'])
        require(type(scenario['expect_success']) is bool and isinstance(scenario['steps'], list), 'invalid scenario')
        require(len(scenario['steps']) <= source['limits']['max_actions'], 'scenario action budget exceeded')
        require(isinstance(scenario['answer'], dict)
                and set(scenario['answer']) == set(answers) | set(explanations), 'scenario answer fields mismatch')
        for expression in scenario['answer'].values():
            validate_expr(expression, roots={'data', 'initial', 'evidence', 'user', 'time'})
        call_ids = set()
        for step in scenario['steps']:
            require(isinstance(step, dict) and 'kind' in step, 'scenario action required')
            if step['kind'] in {'call', 'traverse'}:
                fields(step, ['kind', 'tool', 'id', 'arguments'], ['cursor'] if step['kind'] == 'call' else [])
                require(step['id'] not in call_ids and '.' not in step['id'] and step['id'], 'invalid scenario call id')
                call_ids.add(step['id'])
                require(isinstance(step['arguments'], dict), 'arguments must be an object')
            elif step['kind'] == 'user':
                fields(step, ['kind', 'stage', 'question'], ['proposal', 'source_id'])
                require(step['stage'] in stages, 'unknown scenario user stage')
            elif step['kind'] == 'wait':
                fields(step, ['kind', 'seconds'])
            elif step['kind'] == 'revoke':
                fields(step, ['kind', 'stage'])
            else:
                raise ContractError('unsupported scenario action')
    require(isinstance(source['probes'], list), 'probes must be a list')
    for probe in source['probes']:
        fields(probe, ['id', 'scenario', 'rule_ids', 'mutations', 'expect_success'], ['answer', 'expect_error'])
        require(probe['scenario'] in scenarios and set(probe['rule_ids']) <= set(rules)
                and probe['rule_ids'], 'probe must identify actual rules and a scenario')
        require(type(probe['expect_success']) is bool and isinstance(probe['mutations'], list), 'invalid probe')
    category = source['category']
    require((category != 'direct_response' or not tools and not stages and not tables and not source['user']['facts']), 'direct response must be stateless and have no private facts')
    require((category != 'simple_agentic' or len(tools) == 1), 'simple agentic requires one business tool')
    require((category != 'multi_step_agentic' or len(tools) >= 2
             and any(t.get('result_bindings') for t in tools.values())), 'multistep requires actual result dependencies')
    success = source.setdefault('success', [{'id': 'complete', 'public': 'Complete all declared business tools and user stages.',
        'when': {'literal': True}, 'required_tools': list(tools), 'required_stages': list(stages)}])
    require(bool(success), 'success branches required')
    for branch in named(success, 'success branch').values():
        fields(branch, ['id', 'public', 'when', 'required_tools', 'required_stages'])
        require(isinstance(branch['public'], str) and branch['public'].strip(), 'branch meaning required')
        require(set(branch['required_tools']) <= set(tools) and set(branch['required_stages']) <= set(stages), 'unknown success dependency')
        validate_expr(branch['when'], roots={'data', 'initial', 'evidence', 'user', 'time'})
    digest = hashlib.sha256(canonical(source).encode()).hexdigest()
    from .user_simulator import statechart_spec
    public = {'task': source['task'], 'category': source['category'],
              'business_call_limit': 0 if category == 'direct_response' else 1 if category == 'simple_agentic' else None,
              'rules': [{'id': r['id'], 'text': r['public']} for r in rules.values()],
              'answer_fields': [{k: v for k, v in a.items() if k not in {'expected', 'weight'}} for a in answers.values()],
              'explanations': [{k: v for k, v in e.items() if k != 'weight'} for e in explanations.values()],
              'tools': [public_tool(tool, rules) for tool in tools.values()],
              'user_stages': [{k: v for k, v in s.items() if k != 'fallback'} for s in stages.values()],
              'user_statechart': statechart_spec(),
              'data_schemas': [{'id': t['id'], 'schema': t['schema']} for t in tables.values()],
              'success': success, 'safety': 'An unauthorized protected write attempt prevents full success.',
              'runtime_api': {'collect': 'Assemble contiguous, complete actual query pages by their observation IDs; never fetches or invents records.',
                              'time': 'Read virtual datetime/tick/elapsed_seconds without advancing time.',
                              'user-state': 'Read stage states and transition history only; no private facts or proposals; does not advance time.'},
              'limits': source['limits'], 'time': source['time']}
    return {'version': '2.0', 'identity': digest, 'source': source, 'public': public}


def public_tool(tool, rules):
    published = {key: copy.deepcopy(value) for key, value in tool.items() if key != 'operations'}
    published['business_parameters'] = copy.deepcopy(tool['parameters'])
    wire = copy.deepcopy(tool['parameters'])
    result_ref = {'type': 'object', 'properties': {'$result': {'type': 'string'}, 'path': {'type': 'string'}},
                  'required': ['$result', 'path'], 'additionalProperties': False}
    user_ref = {'type': 'object', 'properties': {'$user': {'type': 'string'}, 'path': {'type': 'string'}},
                'required': ['$user', 'path'], 'additionalProperties': False}
    for argument in tool.get('user_bindings', {}):
        wire['properties'][argument] = {'anyOf': [wire['properties'][argument], user_ref]}
    for binding in tool.get('result_bindings', []):
        wire['properties'][binding['argument']] = copy.deepcopy(result_ref)
    published['parameters'] = wire
    texts = [rules[rule]['public'] for rule in tool['rule_ids']]
    published['description'] = (f"{tool['kind']} operation. Public business rules: " + '; '.join(texts)
        + '. Parameter/result dependencies and authorization below are mandatory. '
        + ('Returns items, total and query-bound pagination cursors. Nulls sort last; ties retain fixture order.'
           if tool['kind'] == 'query' else 'Commits all declared updates atomically only after exact user approval.'))
    return published
