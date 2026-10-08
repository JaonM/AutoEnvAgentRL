"""Compile finite business decisions into exhaustive, disjoint execution routes.

Only explicitly authored boolean decisions are enumerated. Natural-language
conditions are reviewed separately; we never guess branches from keywords.
"""
from __future__ import annotations

import copy
import itertools


def expand_decision_table(source):
    table = source.get('decision_table')
    if table is None:
        return source
    def reject(message):
        raise ValueError('DECISION_TABLE_INVALID: ' + message)
    if (not isinstance(table, dict) or set(table) != {'version', 'decisions', 'branches'}
            or table['version'] != '1.0'):
        reject('requires version 1.0, decisions and branches')
    if 'execution_routes' in source or 'route_probes' in source:
        reject('execution_routes and route_probes are generated; do not duplicate them')
    decisions, branches = table['decisions'], table['branches']
    if (not isinstance(decisions, dict) or not 1 <= len(decisions) <= 3
            or any(not isinstance(key, str) or not key for key in decisions)):
        reject('declare one to three boolean decisions')
    if not isinstance(branches, list) or not 1 <= len(branches) <= 8:
        reject('declare one to eight branches')
    from ..sandbox_runtime import ExecutionRoutes
    for expression in decisions.values():
        # The regular expression/query compiler resolves from_tool later.
        ExecutionRoutes._stable_condition(expression)
        if not isinstance(expression, dict):
            reject('decision must be an observed boolean expression')
    ids = set()
    for branch in branches:
        if (not isinstance(branch, dict)
                or not {'id', 'when', 'required_tools', 'required_stages', 'writes'} <= set(branch)
                or set(branch) - {'id', 'when', 'required_tools', 'required_stages', 'writes', 'probe', 'goal_values'}):
            reject('branch requires id/when/required_tools/required_stages/writes')
        if not isinstance(branch['id'], str) or not branch['id'] or branch['id'] in ids:
            reject('branch IDs must be unique')
        ids.add(branch['id'])
        condition = branch['when']
        if (not isinstance(condition, dict) or set(condition) - set(decisions)
                or any(type(value) is not bool for value in condition.values())):
            reject('when must select declared decisions with boolean values')
    for values in itertools.product((False, True), repeat=len(decisions)):
        assignment = dict(zip(decisions, values))
        matching = [b['id'] for b in branches if all(assignment[k] == v for k, v in b['when'].items())]
        if len(matching) != 1:
            raise ValueError('DECISION_TABLE_COVERAGE: ' + str({'assignment': assignment,
                'matching_branches': matching, 'reason': 'missing' if not matching else 'overlap'}))
    result = copy.deepcopy(source)
    result.pop('decision_table')
    routes, probes, predicates = [], [], {}
    for branch in branches:
        clauses = [{'op': 'eq', 'args': [copy.deepcopy(decisions[key]), {'literal': value}]}
                   for key, value in branch['when'].items()]
        predicate = ({'op': 'and', 'args': clauses} if len(clauses) > 1 else clauses[0] if clauses else {'literal': True})
        predicates[branch['id']] = predicate
        routes.append({key: copy.deepcopy(branch[key]) for key in
                       ('id', 'required_tools', 'required_stages', 'writes')} | {'when': predicate})
        if 'probe' in branch:
            probe = branch['probe']
            if not isinstance(probe, dict) or set(probe) != {'initial_changes', 'reference'}:
                reject('probe requires initial_changes/reference')
            probes.append({'route_id': branch['id'], **copy.deepcopy(probe)})
    # Optional per-branch targets are lowered into one shared goal/reward contract.
    patches = [branch.get('goal_values', {}) for branch in branches]
    if any('goal_values' in branch for branch in branches):
        contract = result.get('business_contract', {})
        targets = contract.get('state_targets', {})
        shapes = []
        for patch in patches:
            if not isinstance(patch, dict) or any(not isinstance(v, dict) for v in patch.values()):
                reject('goal_values must map shared target IDs to field expressions')
            shapes.append({(target, field) for target, fields in patch.items() for field in fields})
        if any(shape != shapes[0] for shape in shapes) or not shapes[0]:
            reject('every branch must define the same conditional goal fields')
        for target, field in shapes[0]:
            if target not in targets or field not in (set(targets[target].get('values', {})) |
                                                     set(targets[target].get('value_expressions', {}))):
                reject('conditional goal must refer to an existing shared target field')
            expression = copy.deepcopy(patches[-1][target][field])
            for branch, patch in reversed(list(zip(branches[:-1], patches[:-1]))):
                expression = {'if': {'condition': copy.deepcopy(predicates[branch['id']]),
                                     'then': copy.deepcopy(patch[target][field]), 'else': expression}}
            targets[target].setdefault('values', {}).pop(field, None)
            targets[target].setdefault('value_expressions', {})[field] = expression
    result['execution_routes'] = routes
    result['route_probes'] = probes
    return result
