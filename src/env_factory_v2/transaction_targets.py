"""Project declared transaction intentions without executing an agent trajectory."""
from __future__ import annotations

import copy

from .expressions import canonical, evaluate, refs, typed_equal, validate_expr


def validate_plan(plan, source, tools, rules, *, temporal=False):
    from .contract import fields, require, validate_rule_links

    require(isinstance(plan, list) and 1 <= len(plan) <= 64, 'transaction plan requires 1..64 intentions')
    require(temporal or not source.get('events'), 'event-driven worlds require timeline or explicit state goals')
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
                stable(expression, {'initial', 'data', 'args', 'row', 'user', 'vars'}
                       | ({'time'} if temporal else set()))


def project_plan(plan, source, context, *, trace=None):
    from .contract import require, validate_value

    tables = {table['id']: table for table in source['tables']}
    tools = {tool['id']: tool for tool in source['tools']}
    projected = copy.deepcopy(context['initial'])
    commits = None if trace is None else [turn for turn in trace
        if turn.get('kind') == 'tool' and turn.get('status') == 'ok'
        and turn.get('tool') in tools and tools[turn['tool']]['kind'] == 'transaction']
    commit_index, event_index = 0, 0
    events = sorted(enumerate(source.get('events', [])), key=lambda pair: (pair[1]['at_seconds'], pair[0]))
    def apply_events(seconds):
        nonlocal event_index
        while event_index < len(events) and events[event_index][1]['at_seconds'] <= seconds:
            event = events[event_index][1]
            table = tables[event['table']]
            matches = [row for row in projected[event['table']]
                       if canonical(row[table['primary_key']]) == canonical(event['key'])]
            require(len(matches) == 1, 'projected event target must identify exactly one row')
            matches[0].update(copy.deepcopy(event['set']))
            event_index += 1
    previous_seconds, previous_tick = 0, 0
    for intent in plan:
        enabled = evaluate(intent.get('when', {'literal': True}), context)
        require(type(enabled) is bool, 'transaction intention condition must be boolean')
        if not enabled:
            continue
        args = {name: evaluate(expr, context) for name, expr in intent['arguments'].items()}
        tool = tools[intent['tool']]
        validate_value(args, tool['parameters'], 'transaction intention arguments')
        operation_time = context['time']
        if commits is not None:
            require(commit_index < len(commits), 'intended transaction has no successful execution')
            commit = commits[commit_index]
            require(commit['tool'] == intent['tool'] and typed_equal(commit['arguments'], args),
                    'executed transaction differs from intended plan')
            operation_time = commit['time']
            require(previous_seconds <= operation_time['elapsed_seconds'] <= context['time']['elapsed_seconds']
                    and previous_tick < operation_time['tick'] <= context['time']['tick'],
                    'transaction timeline is not monotonic')
            previous_seconds, previous_tick = operation_time['elapsed_seconds'], operation_time['tick']
            apply_events(previous_seconds)
            commit_index += 1
        for operation in tool['operations']:
            projection_context = {**context, 'data': projected, 'args': args, 'time': operation_time}
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
    if commits is not None:
        require(commit_index == len(commits), 'unplanned successful transaction')
        apply_events(context['time']['elapsed_seconds'])
    return projected


def matches_plan(plan, source, context, *, trace=None):
    projected = project_plan(plan, source, context, trace=trace)
    # Compare every table and column, independent of storage order. This also
    # rejects collateral writes and preserves optional fields untouched by ops.
    return all(typed_equal(sorted(context['data'][table], key=canonical), sorted(rows, key=canonical))
               for table, rows in projected.items())
