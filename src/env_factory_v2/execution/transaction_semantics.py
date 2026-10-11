"""One pure operation/precondition interpreter for execution and target projection."""
import copy

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import ContractError, canonical, evaluate


class TransactionUnavailable(ContractError):
    """A concrete target/boolean precondition prevents the declared operation."""


def evaluate_preconditions(value, context):
    """Task-facing precondition query over an explicit world/clock basis."""
    from env_factory_v2.contracts.contract import validate_value
    source = context.get('_source')
    require(isinstance(source, dict), 'transaction preconditions require runtime source context')
    tool = next((t for t in source['tools'] if t['id'] == value['tool']), None)
    require(tool is not None and tool['kind'] == 'transaction', 'unknown transaction precondition target')
    arguments = {key: evaluate(expr, context) for key, expr in value['arguments'].items()}
    validate_value(arguments, tool['parameters'], 'transaction precondition arguments')
    basis = value['basis']
    require(basis in {'initial', 'current'}, 'unknown transaction precondition basis')
    selected = context if basis == 'current' else context.get('_eligibility_initial')
    require(isinstance(selected, dict), 'initial transaction precondition snapshot unavailable')
    # Arguments/facts retain the declared task inputs. Initial freezes world,
    # clock and lifecycle after reset events; current uses the caller's world.
    preview = {**context, **{key: selected[key] for key in ('data', 'time', 'user_state')},
               'args': arguments}
    try:
        prepare_transaction(tool, source, selected['data'], preview)
    except TransactionUnavailable:
        return False
    return True


def prepare_transaction(tool, source, before, context, *, check_preconditions=True):
    """Return candidate data and changes; never authorize, commit or advance time.

    Keys and updates see sequential tentative data. All business predicates see
    the original pre-transaction world and original target row. Final schemas,
    invariants and consent consumption remain commit-layer responsibilities.
    """
    tables = {table['id']: table for table in source['tables']}
    candidate, changes = copy.deepcopy(before), []
    expanded = 0
    for operation in tool['operations']:
        # Membership is fixed from the pre-transaction world, so earlier writes
        # cannot silently change which rows a collection operation will touch.
        items = evaluate(operation['for_each'], {**context, 'data': before}) if 'for_each' in operation else [None]
        require(isinstance(items, list), 'transaction collection must produce a list')
        expanded += len(items)
        require(expanded <= 4096, 'transaction expanded operation limit')
        targets = set()
        for item in items:
            item_context = {**context, 'item': item} if 'for_each' in operation else context
            # Applicability is a pre-state decision, independent of earlier writes.
            # A skipped operation must not even resolve an irrelevant/missing target.
            enabled = evaluate(operation.get('when', {'literal': True}), {**item_context, 'data': before})
            require(type(enabled) is bool, 'transaction operation condition must produce boolean')
            if not enabled:
                continue
            staged = {**item_context, 'data': candidate}
            key = evaluate(operation['key'], staged)
            marker = canonical(key)
            if 'for_each' in operation:
                require(marker not in targets, 'duplicate target in collection operation')
                targets.add(marker)
            table = tables[operation['table']]
            rows = candidate[table['id']]
            matches = [row for row in rows if canonical(row[table['primary_key']]) == canonical(key)]
            create = operation.get('create', False)
            if create and matches:
                raise TransactionUnavailable('create target already exists')
            if not create and len(matches) != 1:
                raise TransactionUnavailable('write/mutation target must identify exactly one row')
            if create:
                updates = {name: evaluate(expr, staged) for name, expr in operation['set'].items()}
                row = {table['primary_key']: copy.deepcopy(key), **updates}
                original_row = copy.deepcopy(row)
            else:
                row = matches[0]
                original = [r for r in before[table['id']] if canonical(r[table['primary_key']]) == canonical(key)]
                original_row = original[0] if original else copy.deepcopy(row)
            if check_preconditions:
                checks = [evaluate(context['_rules'][identity]['predicate'],
                                   {**context, 'data': before, 'row': original_row})
                          for identity in tool['rule_ids']]
                require(all(type(value) is bool for value in checks), 'transaction precondition must produce boolean')
                failed = [identity for identity, value in zip(tool['rule_ids'], checks) if not value]
                if failed:
                    raise TransactionUnavailable('transaction business precondition failed: ' + ', '.join(failed))
            if create:
                rows.append(row)
            else:
                updates = {name: evaluate(expr, {**staged, 'row': row}) for name, expr in operation['set'].items()}
                row.update(updates)
            changes.append({'table': table['id'], 'key': key, 'set': copy.deepcopy(updates),
                            **({'create': True} if create else {})})
    if not changes:
        raise TransactionUnavailable('transaction has no applicable operations')
    return candidate, changes
