"""Compose reusable business environments and task-owned goals without overrides."""
from __future__ import annotations

import copy
import hashlib

from .contract import compile_contract, fields, named, require
from .expressions import canonical, refs


def assemble_contract(environment, task):
    environment, task = copy.deepcopy(environment), copy.deepcopy(task)
    fields(environment, ['version', 'id', 'tables', 'rules', 'tools', 'time'], ['invariants', 'events', 'communications'])
    require(environment['version'] == '1.0' and isinstance(environment['id'], str)
            and environment['id'], 'invalid business environment identity')
    # Task definitions cannot replace rules, tables, operations or event logic.
    fields(task, ['version', 'id', 'category', 'request', 'user', 'scenarios', 'probes', 'limits'],
           ['answers', 'reports', 'messages', 'explanations', 'success', 'goals', 'demand'])
    require(task['version'] == '1.0', 'unsupported task definition version')
    tables = named(environment['tables'], 'table')
    tools = named(environment['tools'], 'tool')
    rules = named(environment['rules'], 'rule')
    answers = copy.deepcopy(task.get('answers', []))
    for report in named(task.get('reports', []), 'derived report').values():
        fields(report, ['id', 'meaning', 'query', 'arguments', 'columns', 'weight'])
        require(report['query'] in tools, 'unknown report query')
        tool = tools[report['query']]
        require(tool['kind'] == 'query', 'reports require a read-only query')
        columns = report['columns']
        require(isinstance(columns, list) and columns and all(isinstance(c, str) for c in columns)
                and len(columns) == len(set(columns))
                and set(columns) <= set(tool['columns']), 'report columns must be published query columns')
        require(isinstance(report['arguments'], dict), 'report arguments must be expressions')
        parameters = tool['parameters']
        require(set(parameters.get('required', [])) <= set(report['arguments'])
                <= set(parameters['properties']), 'report arguments differ from query parameters')
        # Invoke the query's named predicates, rather than asking the author to
        # implement eligibility a second time in a reward expression.
        predicates = []
        for rule_id in tool['rule_ids']:
            require(rule_id in rules, 'unknown query rule')
            arguments = {path.split('.')[1] for path in refs(rules[rule_id]['predicate'])
                         if path.startswith('args.')}
            require(arguments <= set(report['arguments']), 'missing report rule argument')
            predicates.append({'rule': {'id': rule_id, 'arguments': {
                key: report['arguments'][key] for key in sorted(arguments)}}})
        require(tool['table'] in tables, 'unknown report table')
        selected = {'ref': 'data.' + tool['table']}
        if predicates:
            selected = {'filter': {'items': selected, 'value': {'and': predicates}}}
        expected = {'map': {'items': selected, 'value': {
            'object': {column: {'ref': 'row.' + column} for column in columns}}}}
        properties = tables[tool['table']]['schema']['properties']
        require(set(columns) <= set(properties), 'unknown report column')
        answers.append({'id': report['id'], 'meaning': report['meaning'], 'weight': report['weight'],
            'comparison': 'unordered', 'schema': {'type': 'array', 'items': {
                'type': 'object', 'properties': {c: copy.deepcopy(properties[c]) for c in columns},
                'required': columns, 'additionalProperties': False}}, 'expected': expected})
    communications = named(environment.get('communications', []), 'communication')
    for obligation in communications.values():
        fields(obligation, ['id', 'rule_id', 'obligation'], ['when'])
        require(obligation['rule_id'] in rules, 'unknown communication rule')
    explanations = copy.deepcopy(task.get('explanations', []))
    for message in named(task.get('messages', []), 'message').values():
        fields(message, ['id', 'meaning', 'obligations', 'context_fields', 'weight'])
        ids = message['obligations']
        require(isinstance(ids, list) and ids and all(isinstance(i, str) for i in ids)
                and len(ids) == len(set(ids)) and set(ids) <= set(communications),
                'message must reference distinct environment communication obligations')
        explanations.append({k: copy.deepcopy(v) for k, v in message.items() if k != 'obligations'})
        explanations[-1]['criteria'] = [copy.deepcopy(communications[i]) for i in ids]
    source = {k: copy.deepcopy(v) for k, v in environment.items() if k not in {'id', 'version', 'communications'}}
    source.update({k: copy.deepcopy(v) for k, v in task.items() if k not in {'version', 'request', 'reports', 'messages', 'answers'}})
    source.update(version='2.0', task=task['request'], answers=answers)
    if explanations:
        source['explanations'] = explanations
    return source


def compile_definition(environment, task):
    compiled = compile_contract(assemble_contract(environment, task))
    return compiled, {'version': '1.0',
        'environment_identity': hashlib.sha256(canonical(environment).encode()).hexdigest(),
        'task_identity': hashlib.sha256(canonical(task).encode()).hexdigest(),
        'contract_identity': compiled['identity'],
        'derived_reports': [report['id'] for report in task.get('reports', [])],
        'derived_messages': [message['id'] for message in task.get('messages', [])],
        'scope': 'compiled_definition_not_independent_business_acceptance'}


def is_definition(source):
    return isinstance(source, dict) and 'environment' in source


def compile_source(source):
    """Compile authored input without discarding its separate source identities."""
    if not is_definition(source):
        return compile_contract(source), None
    fields(source, ['environment', 'task'])
    return compile_definition(source['environment'], source['task'])
