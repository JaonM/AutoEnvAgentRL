"""Exact bounded integer covering, independent of any business family."""
from env_factory_v2.contracts.expressions import ContractError


def bounded_cover(groups, target, *, max_states, max_transitions, max_plans):
    for value, ceiling in [(max_states, 10000), (max_transitions, 1000000), (max_plans, 10000)]:
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ContractError('cover budget outside allowed bounds')
    if type(target) is not int or target < 0 or not isinstance(groups, list):
        raise ContractError('cover requires nonnegative integer target and group list')
    if len(groups) > 10000:
        raise ContractError('cover exceeds 10000 groups')
    keys = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {'key', 'units', 'cost', 'minimum', 'maximum'}:
            raise ContractError('cover group fields mismatch')
        if not isinstance(group['key'], str) or not group['key'] or group['key'] in keys:
            raise ContractError('cover group keys must be nonempty and unique')
        keys.add(group['key'])
        if (any(type(group[k]) is not int for k in ['units', 'cost', 'minimum', 'maximum'])
                or group['units'] <= 0 or group['cost'] < 0
                or group['minimum'] < 1 or group['maximum'] < 0):
            raise ContractError('cover coefficients must be integers with positive units/minimum and nonnegative cost/maximum')
    # An optimal nonnegative-cost, minimum-overage plan cannot contain a
    # removable pack/batch while still covering the target. This bounds quantity
    # without truncating a valid optimum, including large minimum order batches.
    largest_batch = max((g['units'] * g['minimum'] for g in groups), default=1)
    cap = target + largest_batch - 1 if target else 0
    states = {0: (0, [()])}
    transitions = 0
    for group in sorted(groups, key=lambda g: g['key']):
        following = {}
        for quantity, (cost, plans) in states.items():
            upper = min(group['maximum'], (cap - quantity) // group['units'])
            counts = range(group['minimum'], upper + 1)
            # Zero always means omit this group, regardless of its minimum.
            for count in _with_zero(counts):
                transitions += 1
                if transitions > max_transitions:
                    raise ContractError('cover exceeds max_transitions')
                q, c = quantity + count * group['units'], cost + count * group['cost']
                previous = following.get(q)
                if previous is not None and previous[0] < c:
                    continue
                expanded = [p + ((group['key'], count),) if count else p for p in plans]
                if previous is not None and previous[0] == c:
                    expanded = previous[1] + expanded
                if len(expanded) > max_plans:
                    raise ContractError('cover exceeds max_plans')
                following[q] = (c, expanded)
                if len(following) > max_states:
                    raise ContractError('cover exceeds max_states')
        if sum(len(v[1]) for v in following.values()) > max_plans:
            raise ContractError('cover exceeds max_plans')
        states = following
    eligible = [(cost, quantity) for quantity, (cost, _) in states.items() if quantity >= target]
    if not eligible:
        return {'feasible': False, 'cost': None, 'quantity': None, 'plans': []}
    cost, quantity = min(eligible)
    plans = [[{'key': key, 'count': count} for key, count in p] for p in states[quantity][1]]
    return {'feasible': True, 'cost': cost, 'quantity': quantity, 'plans': plans}


def _with_zero(counts):
    yield 0
    yield from counts
