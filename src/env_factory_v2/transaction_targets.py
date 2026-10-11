"""Project declared transaction intentions without executing an agent trajectory."""
from __future__ import annotations

import copy

from .expressions import canonical, evaluate, refs, typed_equal, validate_expr


def validate_plan(plan, source, tools, rules):
    from .contract import fields, require, validate_rule_links

    require(isinstance(plan, list) and 1 <= len(plan) <= 64, 'transaction plan requires 1..64 intentions')
    require(not source.get('events'), 'event-driven worlds require explicit state goals')
    for intent in plan:
        fields(intent, ['tool', 'arguments'], ['when'])
        require(intent['tool'] in tools and tools[intent['tool']]['kind'] == 'transaction',
                'transaction plan requires an existing transaction')
        tool = tools[intent['tool']]
        require(isinstance(intent['arguments'], dict)
                and set(tool['parameters'].get('required', [])) <= set(intent['arguments'])
                <= set(tool['parameters']['properties']), 'transaction plan argument mismatch')
        for expression in [*intent['arguments'].values(), intent.get('when', {'literal': True})]:
            validate_expr(expression, roots={'initial', 'user', 'user_state'})
            validate_rule_links(expression, rules)
        # Named rules may conceal time/state reads behind a rule invocation.
        def stable(expression, allowed, stack=()):
            require(all(path.split('.')[0] in allowed for path in refs(expression)),
                    'transaction plan dependencies require explicit state goals')
            if isinstance(expression, dict):
                if set(expression) == {'literal'}:
                    return
                if set(expression) == {'rule'}:
                    identity = expression['rule']['id']
                    require(identity in rules and identity not in stack, 'invalid transaction target rule')
                    stable(rules[identity]['predicate'], allowed | {'args', 'row'}, (*stack, identity))
                for value in expression.values():
                    stable(value, allowed, stack)
            elif isinstance(expression, list):
                for value in expression:
                    stable(value, allowed, stack)
        for expression in [*intent['arguments'].values(), intent.get('when', {'literal': True})]:
            stable(expression, {'initial', 'user', 'user_state', 'row', 'vars'})
        for operation in tool['operations']:
            for expression in [operation['key'], *operation['set'].values()]:
                stable(expression, {'initial', 'data', 'args', 'row', 'user', 'vars'})


def project_plan(plan, source, context):
    from .contract import require, validate_value

    tables = {table['id']: table for table in source['tables']}
    tools = {tool['id']: tool for tool in source['tools']}
    projected = copy.deepcopy(context['initial'])
    for intent in plan:
        enabled = evaluate(intent.get('when', {'literal': True}), context)
        require(type(enabled) is bool, 'transaction intention condition must be boolean')
        if not enabled:
            continue
        args = {name: evaluate(expr, context) for name, expr in intent['arguments'].items()}
        tool = tools[intent['tool']]
        validate_value(args, tool['parameters'], 'transaction intention arguments')
        for operation in tool['operations']:
            projection_context = {**context, 'data': projected, 'args': args}
            key = evaluate(operation['key'], projection_context)
            table = tables[operation['table']]
            rows = projected[table['id']]
            matches = [row for row in rows if canonical(row[table['primary_key']]) == canonical(key)]
            if operation.get('create', False):
                require(not matches, 'transaction intention creates duplicate key')
                row = {table['primary_key']: copy.deepcopy(key)}
                updates = {name: evaluate(expr, projection_context) for name, expr in operation['set'].items()}
                row.update(updates)
                rows.append(row)
            else:
                require(len(matches) == 1, 'transaction intention target must identify exactly one row')
                row = matches[0]
                updates = {name: evaluate(expr, {**projection_context, 'row': row})
                           for name, expr in operation['set'].items()}
                row.update(updates)
    return projected


def matches_plan(plan, source, context):
    projected = project_plan(plan, source, context)
    # Compare every table and column, independent of storage order. This also
    # rejects collateral writes and preserves optional fields untouched by ops.
    return all(typed_equal(sorted(context['data'][table], key=canonical), sorted(rows, key=canonical))
               for table, rows in projected.items())
