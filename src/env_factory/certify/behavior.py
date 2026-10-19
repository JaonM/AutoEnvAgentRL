"""Caller-owned executable behavior contracts, independent of tool/reference outputs.

The small JSON expression language has no eval, imports or arbitrary callbacks.
Fixtures and expected properties belong to the specification, not the repairer.
"""
import copy
import hashlib
import json

from env_factory.runtime import Episode, BusinessError, canonical


def digest(contract):
    return hashlib.sha256(canonical(contract).encode()).hexdigest()


def at(value, path):
    parts = path.split('.') if isinstance(path, str) else path
    if not parts:
        return value
    key, *tail = parts
    if key == '*':
        return [at(v, tail) for v in (value.values() if isinstance(value, dict) else value)]
    return at(value[int(key)] if isinstance(value, list) else value[key], tail)


def expression(value, context):
    if isinstance(value, list):
        return [expression(v, context) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'path'}:
        return at(context, value['path'])
    if 'op' in value:
        args = [expression(v, context) for v in value.get('args', [])]
        op = value['op']
        if op == 'sum': return sum(args[0])
        if op == 'len': return len(args[0])
        if op == 'add': return sum(args)
        if op == 'sub': return args[0] - args[1]
        if op == 'eq': return args[0] == args[1]
        if op == 'ne': return args[0] != args[1]
        if op == 'ge': return args[0] >= args[1]
        if op == 'le': return args[0] <= args[1]
        if op == 'unique': return len(args[0]) == len(set(map(canonical, args[0])))
        if op == 'contains': return args[1] in args[0]
        if op == 'split': return args[0].split(args[1])
        if op == 'get': return args[0][args[1]]
        raise ValueError('unknown behavior expression: ' + str(op))
    return {k: expression(v, context) for k, v in value.items()}


def run(env, contract):
    if contract.get('version') != 1 or not contract.get('cases'):
        return {'passed': False, 'problems': ['behavior contract needs version 1 and nonempty cases'], 'cases': []}
    try:
        matrix = coverage(contract)
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        return {'passed': False, 'cases': [], 'problems': ['malformed rule coverage: ' + str(error)]}
    if not matrix['passed']:
        return {'passed': False, 'coverage': matrix, 'cases': [], 'problems': matrix['problems']}
    rows = []
    for case in contract['cases']:
        errors = [];outputs = [];step_index = None
        try:
            if not case.get('steps') or len(case['steps']) > 1000:
                raise ValueError('case needs 1..1000 steps')
            api = Episode(env.make_tools(), case['data'], {}, start=case.get('now', '2024-05-15T15:00:00'), approve=lambda *_: True)
            for step_index, step in enumerate(case['steps']):
                before = copy.deepcopy(api.db)
                if 'now' in step:
                    api.now = step['now']
                context = {'initial': case['data'], 'before': before, 'after': api.db, 'outputs': outputs}
                args = expression(step.get('args', {}), context)
                try:
                    result = api.call(step['tool'], **args)
                except BusinessError as exc:
                    if not step.get('error') or (step['error'] != '*' and step['error'] not in str(exc)):
                        raise
                    if canonical(api.db) != canonical(before):
                        raise ValueError('failed operation changed state')
                    result = {'error': str(exc)}
                else:
                    if step.get('error'):
                        raise ValueError('expected rejection, operation succeeded')
                outputs.append(result)
                context.update(result=result, after=api.db)
                checks = step.get('checks', []) + contract.get('invariants', []) + case.get('invariants', [])
                if not checks and not step.get('error'):
                    raise ValueError('successful step has no independently declared checks')
                for check in checks:
                    if expression(check['assert'], context) is not True:
                        errors.append({'step': step_index, 'rule': check['name'], 'assert': check['assert']})
                if errors:
                    break
        except Exception as exc:
            errors.append({'step': step_index, 'error': f'{type(exc).__name__}: {exc}'})
        rows.append({'id': case['id'], 'passed': not errors, 'failures': errors})
    return {'passed': all(r['passed'] for r in rows), 'cases': rows, 'coverage': matrix, 'contract_sha256': digest(contract)}


def qualify(env, request):
    contract = request.get('behavior_contract')
    if contract is None:
        return {'passed': True, 'applicable': False}
    if request.get('behavior_contract_sha256') != digest(contract):
        return {'passed': False, 'problems': ['behavior contract hash mismatch']}
    from env_factory.core import spec
    for rule in contract.get('rules', []):
        if not isinstance(rule, dict):
            return {'passed': False, 'problems': ['malformed rule declaration']}
        if not spec.quoted(request, rule.get('brief_quote')):
            return {'passed': False, 'problems': [f"{rule.get('id')}: rule quote not found in business authority"]}
    return run(env, contract)


CASE_KINDS = frozenset({'positive', 'negative', 'boundary', 'interleaving'})


def coverage(contract):
    """Rule-to-case matrix from caller-owned assertions, including rejected operations."""
    rules = contract.get('rules', [])
    if not rules:
        return {'passed': True, 'applicable': False, 'rules': {}, 'problems': []}
    problems, matrix = [], {}
    for rule in rules:
        identifier = rule.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in matrix:
            problems.append('rule ids must be unique nonempty strings')
            continue
        required = rule.get('required_cases', ['positive', 'negative', 'boundary'])
        if not required or not set(required) <= CASE_KINDS:
            problems.append(f'{identifier}: invalid required_cases')
        if len(set(rule.get('workflows', []))) > 1 and 'interleaving' not in required:
            required = [*required, 'interleaving']
        matrix[identifier] = {'required': required, 'cases': {kind: [] for kind in CASE_KINDS}}
    seen = set()
    for case in contract.get('cases', []):
        identifier, kind = case.get('id'), case.get('kind')
        if not identifier or identifier in seen:
            problems.append('case ids must be unique nonempty strings')
        seen.add(identifier)
        if kind not in CASE_KINDS:
            problems.append(f'{identifier}: missing or invalid case kind')
            continue
        steps = case.get('steps', [])
        evidence = {check.get('rule_id') for step in steps for check in step.get('checks', [])}
        evidence.update(check.get('rule_id') for check in contract.get('invariants', []) + case.get('invariants', []))
        evidence.update(step.get('rule_id') for step in steps if step.get('error'))
        for rule_id in case.get('rules', []):
            if rule_id not in matrix or rule_id not in evidence:
                problems.append(f'{identifier}: {rule_id} has no independently declared assertion/rejection')
                continue
            if kind == 'negative' and not any(s.get('error') for s in steps):
                problems.append(f'{identifier}: negative case needs an expected rejection')
                continue
            if kind == 'interleaving' and (len(set(case.get('workflows', []))) < 2
                                           or len({step.get('tool') for step in steps}) < 2):
                problems.append(f'{identifier}: interleaving case needs multiple workflows and tools')
                continue
            matrix[rule_id]['cases'][kind].append(identifier)
    for identifier, row in matrix.items():
        row['missing'] = [kind for kind in row['required'] if not row['cases'].get(kind)]
        if row['missing']:
            problems.append(f"{identifier}: missing {', '.join(row['missing'])} cases")
    return {'passed': not problems, 'applicable': True, 'rules': matrix, 'problems': problems}
