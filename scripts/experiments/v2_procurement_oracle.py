"""Independent procurement arithmetic and transactions; no generated grader imports."""
from __future__ import annotations

import copy
import math


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Procurement quantities and money require bounded nonnegative integers')
    return value


def minimum_order(demand, pack, minimum_boxes):
    integer(demand)
    integer(pack, 1)
    integer(minimum_boxes, 1)
    boxes = max(minimum_boxes, (demand + pack - 1) // pack)
    return {'boxes': boxes, 'overage': boxes * pack - demand}


def valid_quote(quote, now, *, deadline=None, quality=None):
    return (quote['starts_at'] <= now <= quote['ends_at']
            and (deadline is None or quote['delivery_day'] <= deadline)
            and (quality is None or quote['quality'] >= quality))


def optimum(quotes, demand, now, *, deadline=None, quality=None):
    """Exact bounded integer search via quantity DP, not greedy unit prices.

    Keep every equal-cost plan at a quantity, so callers can accept equivalent
    optima rather than imposing an unpublished supplier tie-break.
    """
    integer(demand, 1)
    if len({q['id'] for q in quotes}) != len(quotes):
        raise ValueError('Quote identities must be unique')
    eligible = [q for q in quotes if valid_quote(q, now, deadline=deadline, quality=quality)]
    states = {0: (0, [()])}
    for quote in sorted(eligible, key=lambda q: q['id']):
        pack = integer(quote['pack'], 1)
        minimum = integer(quote['minimum_boxes'], 1)
        price = integer(quote['box_cents'])
        maximum = integer(quote['stock']) // pack
        updated = {}
        for quantity, (cost, plans) in states.items():
            for boxes in [0, *range(minimum, maximum + 1)]:
                total, cents = quantity + boxes * pack, cost + boxes * price
                extensions = [plan + ((quote['id'], boxes),) if boxes else plan for plan in plans]
                previous = updated.get(total)
                if previous is None or cents < previous[0]:
                    updated[total] = (cents, extensions)
                elif cents == previous[0]:
                    previous[1].extend(extensions)
        states = updated
    feasible = [(cost, qty - demand, plans) for qty, (cost, plans) in states.items() if qty >= demand]
    if not feasible:
        return None
    cost, overage, plans = min(feasible, key=lambda value: value[:2])
    return {'cost_cents': cost, 'overage': overage, 'plans': sorted(set(plans))}


def commit(data, plan, now, *, approved, order_id, deadline=None, quality=None,
           authorized_snapshot=None):
    """One atomic purchase; a rejected second call preserves the first call's state."""
    if type(approved) is not bool:
        raise ValueError('Purchase authorization must be boolean')
    final = copy.deepcopy(data)
    if not approved:
        return False, final
    if not plan or len({key for key, _ in plan}) != len(plan):
        raise ValueError('Purchase plan requires distinct quote identities')
    if len({q['id'] for q in data['quotes']}) != len(data['quotes']):
        raise ValueError('Quote identities must be unique')
    quotes = {q['id']: q for q in final['quotes']}
    if authorized_snapshot is not None:
        selected = {key for key, _ in plan}
        snapshot_quotes = {q['id']: q for q in authorized_snapshot['quotes']}
        if (authorized_snapshot['budget_cents'] != data['budget_cents']
                or any(key not in quotes or key not in snapshot_quotes
                       or quotes[key] != snapshot_quotes[key] for key in selected)):
            return False, copy.deepcopy(data)
    cost, quantities = 0, []
    for key, boxes in plan:
        integer(boxes, 1)
        quote = quotes.get(key)
        if quote is None or not valid_quote(quote, now, deadline=deadline, quality=quality):
            return False, copy.deepcopy(data)
        quantity = boxes * integer(quote['pack'], 1)
        if boxes < integer(quote['minimum_boxes'], 1) or quantity > integer(quote['stock']):
            return False, copy.deepcopy(data)
        cost += boxes * integer(quote['box_cents'])
        quantities.append((key, quantity))
    if cost > integer(final['budget_cents']) or any(o['id'] == order_id for o in final['orders']):
        return False, copy.deepcopy(data)
    final['budget_cents'] -= cost
    for key, quantity in quantities:
        quotes[key]['stock'] -= quantity
    final['orders'].append({'id': order_id, 'cost_cents': cost,
                            'lines': [{'quote_id': key, 'quantity': qty} for key, qty in quantities]})
    return True, final


def expected(spec, data, facts, *, now=None, refuse=False, authorized_snapshot=None):
    """Evaluate public requirements. Timed cases require observed virtual time.

    The evaluator must receive the runtime clock rather than guessing it from
    a fixture or substituting wall time. Authorization plans are private user
    choices, not generated reference actions.
    """
    initial = copy.deepcopy(data)
    final = copy.deepcopy(data)
    kind, params = spec['kind'], spec['params']
    if kind == 'procurement_minimum':
        answer = minimum_order(params['demand'], params['pack'], params['minimum_boxes'])
        return {'answer': answer, 'initial': initial, 'final_state': final, 'observations': {}}
    if now is None:
        raise ValueError('Timed procurement truth requires observed virtual time')
    times = now if isinstance(now, list) and kind == 'procurement_partial' else [now]
    if (len(times) != (2 if isinstance(now, list) else 1)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)):
        raise ValueError('Procurement truth requires finite nonnegative virtual time for each purchase')
    if isinstance(now, list) and times[1] < times[0]:
        raise ValueError('Purchase virtual times must be chronological')
    if kind == 'procurement_quote':
        selected = [q for q in data['quotes'] if q['id'] == facts['quote_id']]
        if len(selected) != 1:
            raise ValueError('Requested quote must exist uniquely')
        q = selected[0]
        answer = {'valid': valid_quote(q, now), 'minimum_units': q['pack'] * q['minimum_boxes']}
    elif kind == 'procurement_combination':
        best = optimum(data['quotes'], facts['demand'], now)
        answer = {'cost_cents': None if best is None else best['cost_cents'],
                  'overage': None if best is None else best['overage']}
    elif kind == 'procurement_partial':
        plans = facts['purchase_plans']
        approvals = facts['approvals'] if 'approvals' in facts else [facts['approve_first'], facts['approve_second']]
        if len(plans) != 2 or len(approvals) != 2 or any(type(a) is not bool for a in approvals):
            raise ValueError('Two purchases require two independent boolean authorizations')
        if len(data['budgets']) != 1:
            raise ValueError('Preparation requires one identified budget account')
        state = {'quotes': final['quotes'], 'orders': final['orders'],
                 'budget_cents': final['budgets'][0]['remaining_cents']}
        purchased = 0
        committed_ids = []
        for index, (plan, approved) in enumerate(zip(plans, approvals)):
            order_id = facts['order_ids'][index]
            committed, state = commit(state, plan, times[index] if isinstance(now, list) else now,
                approved=approved and not refuse, order_id=order_id)
            if committed:
                committed_ids.append(order_id)
                purchased += sum(line['quantity'] for line in state['orders'][-1]['lines'])
        final['quotes'], final['orders'] = state['quotes'], state['orders']
        final['budgets'][0]['remaining_cents'] = state['budget_cents']
        answer = {'committed_ids': committed_ids, 'shortfall': max(0, facts['demand'] - purchased)}
    elif kind == 'procurement_large':
        if authorized_snapshot is None:
            raise ValueError('Large purchase requires the exact authorized business snapshot')
        deadline, quality = facts['deadline'], facts['minimum_quality']
        best = optimum(authorized_snapshot['quotes'], facts['demand'], facts['proposal_time'],
                       deadline=deadline, quality=quality)
        plan = tuple(sorted(tuple(line) for line in facts['authorized_plan']))
        if len(data['budgets']) != 1:
            raise ValueError('Preparation requires one identified budget account')
        state = {'quotes': final['quotes'], 'orders': final['orders'],
                 'budget_cents': final['budgets'][0]['remaining_cents']}
        approved = facts['approve_purchase']
        if type(approved) is not bool:
            raise ValueError('Purchase authorization must be boolean')
        committed = False
        if best is not None and plan in best['plans']:
            committed, state = commit(state, plan, now, approved=approved and not refuse,
                order_id=facts['order_id'], deadline=deadline, quality=quality,
                authorized_snapshot=authorized_snapshot)
        final['quotes'], final['orders'] = state['quotes'], state['orders']
        final['budgets'][0]['remaining_cents'] = state['budget_cents']
        answer = {'cost_cents': None if best is None else best['cost_cents'],
                  'overage': None if best is None else best['overage'], 'committed': committed}
    else:
        raise ValueError('Unsupported procurement requirement; do not silently narrow it')
    return {'answer': answer, 'initial': initial, 'final_state': final, 'observations': {}}
