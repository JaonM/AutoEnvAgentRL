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
    return {'passed': all(r['passed'] for r in rows), 'cases': rows, 'contract_sha256': digest(contract)}


def qualify(env, request):
    contract = request.get('behavior_contract')
    if contract is None:
        return {'passed': True, 'applicable': False}
    if request.get('behavior_contract_sha256') != digest(contract):
        return {'passed': False, 'problems': ['behavior contract hash mismatch']}
    return run(env, contract)
