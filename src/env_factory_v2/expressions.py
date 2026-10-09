"""Small, total expression language: data only, never Python evaluation."""
from __future__ import annotations

import copy
import json
import math


class ContractError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def typed_equal(a, b):
    if type(a) in {int, float} and type(b) in {int, float}:
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(typed_equal(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def path(value, pointer):
    if not isinstance(pointer, str):
        raise ContractError('path must be a string')
    for key in pointer.split('.') if pointer else []:
        try:
            value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ContractError(f'unresolved path: {pointer}') from exc
    return copy.deepcopy(value)


ARITY = {'eq': 2, 'ne': 2, 'lt': 2, 'le': 2, 'gt': 2, 'ge': 2,
         'in': 2, 'add': 2, 'sub': 2, 'mul': 2, 'not': 1,
         'count': 1, 'sum': 1, 'unique': 1, 'contains': 2, 'at': 2}


def refs(expression):
    if isinstance(expression, dict):
        if set(expression) == {'literal'}:
            return []
        if set(expression) == {'ref'}:
            return [expression['ref']]
        return [ref for value in expression.values() for ref in refs(value)]
    if isinstance(expression, list):
        return [ref for item in expression for ref in refs(item)]
    return []


def validate_expr(expression, *, roots, depth=0):
    if depth > 32:
        raise ContractError('expression exceeds depth limit')
    if not isinstance(expression, dict) or len(expression) != 1:
        raise ContractError('expression must have exactly one operator')
    op, value = next(iter(expression.items()))
    if op == 'literal':
        canonical(value)
        return
    if op == 'ref':
        if not isinstance(value, str) or value.split('.')[0] not in roots:
            raise ContractError(f'forbidden expression reference: {value}')
        return
    if op == 'rule':
        if not isinstance(value, dict) or set(value) != {'id', 'arguments'} or not isinstance(value['id'], str) or not isinstance(value['arguments'], dict):
            raise ContractError('invalid public rule invocation')
        for expression in value['arguments'].values():
            validate_expr(expression, roots=roots, depth=depth + 1)
        return
    if op in {'and', 'or', 'list', 'object'} or op in ARITY:
        values = list(value.values()) if op == 'object' and isinstance(value, dict) else value
        if not isinstance(values, list) or (op in ARITY and len(values) != ARITY[op]):
            raise ContractError(f'invalid operands: {op}')
        if op in {'and', 'or'} and not values:
            raise ContractError('empty boolean expression')
        for item in values:
            validate_expr(item, roots=roots, depth=depth + 1)
        return
    if op in {'filter', 'map'}:
        if not isinstance(value, dict) or set(value) != {'items', 'value'}:
            raise ContractError(f'invalid {op} expression')
        validate_expr(value['items'], roots=roots, depth=depth + 1)
        validate_expr(value['value'], roots=roots | {'row'}, depth=depth + 1)
        return
    if op == 'if':
        if not isinstance(value, dict) or set(value) != {'condition', 'then', 'else'}:
            raise ContractError('invalid conditional')
        for item in value.values():
            validate_expr(item, roots=roots, depth=depth + 1)
        return
    raise ContractError(f'unsupported expression operator: {op}')


def evaluate(expression, context):
    op, value = next(iter(expression.items()))
    if op == 'literal':
        return copy.deepcopy(value)
    if op == 'ref':
        return path(context, value)
    if op == 'rule':
        if value['id'] not in context.get('_rules', {}):
            raise ContractError('unknown executable public rule')
        arguments = {key: evaluate(expression, context) for key, expression in value['arguments'].items()}
        return evaluate(context['_rules'][value['id']]['predicate'], {**context, 'args': arguments})
    if op == 'if':
        condition = evaluate(value['condition'], context)
        if type(condition) is not bool:
            raise ContractError('condition must be boolean')
        return evaluate(value['then' if condition else 'else'], context)
    if op in {'filter', 'map'}:
        items = evaluate(value['items'], context)
        if not isinstance(items, list):
            raise ContractError('collection expression requires a list')
        result = []
        for item in items:
            computed = evaluate(value['value'], {**context, 'row': item})
            if op == 'map':
                result.append(computed)
            elif type(computed) is not bool:
                raise ContractError('filter condition must be boolean')
            elif computed:
                result.append(copy.deepcopy(item))
        return result
    if op == 'object':
        return {key: evaluate(item, context) for key, item in value.items()}
    if op in {'and', 'or'}:
        for item in value:
            computed = evaluate(item, context)
            if type(computed) is not bool:
                raise ContractError('boolean operands required')
            if op == 'and' and not computed:
                return False
            if op == 'or' and computed:
                return True
        return op == 'and'
    args = [evaluate(item, context) for item in value]
    if op == 'list':
        return args
    if op in {'and', 'or', 'not'}:
        if any(type(item) is not bool for item in args):
            raise ContractError('boolean operands required')
        return all(args) if op == 'and' else any(args) if op == 'or' else not args[0]
    if op in {'eq', 'ne'}:
        equal = typed_equal(args[0], args[1])
        return equal if op == 'eq' else not equal
    if op in {'add', 'sub', 'mul', 'sum'}:
        numbers = args[0] if op == 'sum' else args
        if not isinstance(numbers, list) or any(type(n) not in {int, float} or not math.isfinite(n) for n in numbers):
            raise ContractError('finite numeric operands required')
        result = sum(numbers) if op in {'add', 'sum'} else numbers[0] - numbers[1] if op == 'sub' else numbers[0] * numbers[1]
        if not math.isfinite(result):
            raise ContractError('numeric overflow')
        return result
    if op == 'count':
        return len(args[0])
    if op == 'at':
        if not isinstance(args[0], list) or type(args[1]) is not int or not 0 <= args[1] < len(args[0]):
            raise ContractError('unresolved collection index')
        return copy.deepcopy(args[0][args[1]])
    if op == 'unique':
        return list({canonical(item): item for item in args[0]}.values())
    if op in {'in', 'contains'}:
        item, collection = args if op == 'in' else args[::-1]
        return any(typed_equal(item, other) for other in collection)
    if op in {'lt', 'le', 'gt', 'ge'}:
        a, b = args
        if type(a) is not type(b) and not (type(a) in {int, float} and type(b) in {int, float}):
            raise ContractError('incompatible comparison types')
        return {'lt': lambda: a < b, 'le': lambda: a <= b,
                'gt': lambda: a > b, 'ge': lambda: a >= b}[op]()
    raise ContractError(f'unsupported operator: {op}')
