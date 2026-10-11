"""Compose reusable business environments and task-owned goals without overrides."""
from __future__ import annotations

import copy
import hashlib

from .contract import compile_contract, fields, named, require
from .expressions import canonical, refs


def assemble_contract(environment, task):
    environment, task = copy.deepcopy(environment), copy.deepcopy(task)
    fields(environment, ['version', 'id', 'tables', 'rules', 'tools', 'time'], ['invariants', 'events'])
    require(environment['version'] == '1.0' and isinstance(environment['id'], str)
            and environment['id'], 'invalid business environment identity')
    # Task definitions cannot replace rules, tables, operations or event logic.
    fields(task, ['version', 'id', 'category', 'request', 'user', 'scenarios', 'probes', 'limits'],
           ['answers', 'reports', 'explanations', 'success', 'goals', 'demand'])
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
    source = {k: copy.deepcopy(v) for k, v in environment.items() if k not in {'id', 'version'}}
    source.update({k: copy.deepcopy(v) for k, v in task.items() if k not in {'version', 'request', 'reports', 'answers'}})
    source.update(version='2.0', task=task['request'], answers=answers)
    return source


def compile_definition(environment, task):
    compiled = compile_contract(assemble_contract(environment, task))
    return compiled, {'version': '1.0',
        'environment_identity': hashlib.sha256(canonical(environment).encode()).hexdigest(),
        'task_identity': hashlib.sha256(canonical(task).encode()).hexdigest(),
        'contract_identity': compiled['identity'],
        'derived_reports': [report['id'] for report in task.get('reports', [])],
        'scope': 'compiled_definition_not_independent_business_acceptance'}
