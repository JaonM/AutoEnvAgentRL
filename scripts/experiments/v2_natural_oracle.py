"""Independent business oracle: no contract expressions or generated targets.

Only evaluator code imports this module. Frozen business records and user facts
determine the expected result, including variants. Field matching is a separate
native semantic review and may not change these values or verdicts.
"""
from __future__ import annotations

import copy
import math


def expand(business):
    data = {}
    for table in business['tables']:
        if 'rows' in table:
            data[table['id']] = copy.deepcopy(table['rows'])
            continue
        plan = table['generator']
        rows = []
        for index in range(plan['count']):
            row = {}
            for column, spec in plan['columns'].items():
                if spec['kind'] == 'sequence':
                    value = spec['start'] + index * spec['step']
                    if 'prefix' in spec:
                        value = spec['prefix'] + str(value).zfill(spec.get('width', 0))
                elif spec['kind'] == 'cycle':
                    value = spec['values'][(index + plan['seed'] + spec.get('offset', 0)) % len(spec['values'])]
                elif spec['kind'] == 'constant':
                    value = copy.deepcopy(spec['value'])
                else:
                    raise ValueError('Oracle fixture generator unsupported')
                row[column] = value
            rows.append(row)
        data[table['id']] = rows
    return data


def change(data, mutation):
    candidates = [r for r in data[mutation['table']] if r['id'] == mutation['key']]
    if len(candidates) != 1:
        raise ValueError('Oracle mutation must identify exactly one record')
    candidates[0][mutation['column']] = copy.deepcopy(mutation['value'])


def expected(spec, business, user, *, mutation=None, refuse=False):
    data = expand(business)
    if mutation:
        change(data, mutation)
    if spec['kind'].startswith('library_'):
        from v2_library_oracle import expected as library_expected
        truth = library_expected(spec, data, user['facts'], refuse=refuse)
        truth['observations'] = copy.deepcopy(spec['observations'])
        return truth
    final = copy.deepcopy(data)
    facts, kind, p = user['facts'], spec['kind'], spec['params']
    observation = copy.deepcopy(spec['observations'])
    def get(table, key):
        rows = [row for row in final[table] if row['id'] == key]
        if len(rows) != 1:
            raise ValueError('Oracle identity missing or nonunique')
        return rows[0]
    def fields(**values):
        return values
    if kind == 'energy':
        energy = sum(w * h / 1000 for w, h in zip(p['watts'], p['hours']))
        answer = fields(energy_kwh=energy, cost=energy * p['price'])
    elif kind == 'discount':
        original = sum(p['prices'])
        payable = max(0, original * (p['rate'] if original >= p['threshold'] else 1) - p['coupon'])
        answer = fields(payable=payable, savings=original - payable)
    elif kind == 'duration':
        hours = (p['end_minutes'] - p['start_minutes']) / 60
        answer = fields(hours=hours, cost=hours * p['hourly_rate'])
    elif kind == 'ranking':
        chosen = sorted(p['candidates'], key=lambda r: (-r['rating'], r['price'], r['id']))[:3]
        answer = fields(names=[r['name'] for r in chosen], total_price=sum(r['price'] for r in chosen))
    elif kind == 'ticket':
        row = get('tickets', facts['ticket_id'])
        answer = fields(status=row['status'], eta=row['eta'])
    elif kind == 'member':
        row = get('members', facts['member_id'])
        answer = fields(points=row['points'], enabled=row['enabled'])
    elif kind == 'shoes':
        rows = [r for r in final['shoes'] if r['size'] == facts['size'] and r['active'] is True and r['stock'] > 0]
        answer = fields(ids=[r['id'] for r in rows])
        observation = {'shoes': answer['ids']}
    elif kind == 'receipts':
        rows = [r for r in final['receipts'] if r['employee_id'] == facts['employee_id'] and not r['paid'] and not r['cancelled']]
        answer = fields(ids=[r['id'] for r in rows], total_amount=sum(r['amount'] for r in rows))
        observation = {'receipts': answer['ids']}
    elif kind == 'shipment':
        order = get('orders', facts['order_id'])
        row = get('shipments', order['shipment_id'])
        answer = fields(status=row['status'], eta=row['eta'])
    elif kind == 'warranty':
        device = get('devices', facts['device_id'])
        model = get('models', device['model_id'])
        covered = device['age_days'] <= model['warranty_days']
        answer = fields(covered=covered, fee=0 if covered else model['inspection_fee'])
    elif kind == 'employees':
        department = next(r for r in final['departments'] if r['name'] == facts['department_name'])
        rows = [r for r in final['employees'] if r['department_id'] == department['id'] and r['active']]
        answer = fields(ids=[r['id'] for r in rows], count=len(rows))
        observation = {'departments': [department['id']], 'employees': answer['ids']}
    elif kind == 'cancel':
        booking = get('reservations', facts['reservation_id'])
        slot = get('slots', booking['slot_id'])
        if not refuse:
            booking['active'], slot['reserved'] = False, False
        answer = fields(submitted=not refuse, active=booking['active'], reserved=slot['reserved'])
    elif kind == 'chain':
        application = get('applications', facts['application_id'])
        path, identifier, duration = [], application['entry_id'], 0
        while identifier is not None:
            if identifier in path:
                raise ValueError('Oracle chain cycle')
            row = get('nodes', identifier)
            path.append(identifier)
            duration += row['duration']
            identifier = row['next_id']
        owner = get('people', row['owner_id'])
        answer = fields(path=path, total_duration=duration, owner_name=owner['name'])
        observation = {'applications': [application['id']], 'nodes': path, 'people': [owner['id']]}
    elif kind == 'expenses':
        project = next(r for r in final['projects'] if r['name'] == facts['project_name'])
        rows = [r for r in final['expenses'] if r['project_id'] == project['id'] and r['status'] == 'approved' and r['amount'] > 4900]
        answer = fields(count=len(rows), total_amount=sum(r['amount'] for r in rows))
        observation = {'projects': [project['id']], 'expenses': [r['id'] for r in rows]}
    elif kind == 'inventory':
        warehouse = next(r for r in final['warehouses'] if r['name'] == facts['warehouse_name'])
        rows = [r for r in final['inventory'] if r['warehouse_id'] == warehouse['id'] and r['stock'] >= 6]
        answer = fields(count=len(rows), total_stock=sum(r['stock'] for r in rows))
        observation = {'warehouses': [warehouse['id']], 'inventory': [r['id'] for r in rows]}
    elif kind == 'replace':
        assignment = get('assignments', facts['assignment_id'])
        old, new = get('equipment', assignment['device_id']), get('equipment', facts['device_choice'])
        old_maintenance = next(r for r in final['maintenance'] if r['device_id'] == old['id'])
        new_maintenance = next(r for r in final['maintenance'] if r['device_id'] == new['id'])
        if not refuse:
            old['capacity'] += 1
            new['capacity'] -= 1
            assignment['device_id'] = new['id']
            old_maintenance['scheduled'], new_maintenance['scheduled'] = False, True
        answer = fields(submitted=not refuse, device_id=assignment['device_id'])
    elif kind == 'transfer':
        document = get('transfers', facts['transfer_id'])
        source, target = get('stocks', document['source_id']), get('stocks', document['target_id'])
        if not refuse:
            source['stock'] -= 12
            target['stock'] += 12
            document['completed'] = True
        answer = fields(submitted=not refuse, source_stock=source['stock'], target_stock=target['stock'], completed=document['completed'])
    elif kind == 'enroll':
        activity = get('activities', facts['activity_id'])
        registration = get('registrations', activity['registration_id'])
        if not refuse:
            activity['remaining'] -= 1
            registration['enrolled'] = True
        answer = fields(submitted=not refuse, enrolled=registration['enrolled'], remaining=activity['remaining'])
    elif kind == 'correction':
        # The public obligation is the complete available menu for the final
        # preference. An early correction can make the old region obsolete
        # before any query; observing its records is not a business requirement.
        observation['venues'] = [row['id'] for row in data['venues']
                                 if row['region'] == facts['region_new'] and row['available']]
        venue = get('venues', facts['venue_choice'])
        booking = get('bookings', facts['booking_id'])
        if not refuse:
            venue['available'] = False
            booking['venue_id'] = venue['id']
        answer = fields(submitted=not refuse, venue_id=booking['venue_id'])
    elif kind == 'sequential':
        account = get('accounts', facts['account_id'])
        committed = []
        for identifier in ['F1', 'F2']:
            feature = get('features', identifier)
            if not refuse and account['quota'] >= 1:
                account['quota'] -= 1
                feature['active'] = True
                get('logs', feature['log_id'])['completed'] = True
                committed.append(identifier)
        answer = fields(first_active=get('features', 'F1')['active'], second_active=get('features', 'F2')['active'], quota=account['quota'])
    else:
        raise ValueError('Unknown independently frozen business kind')
    return {'answer': answer, 'initial': data, 'final_state': final, 'observations': observation}


UNORDERED = {'ids'}


def equivalent(actual, expected_value, *, unordered=False):
    if isinstance(expected_value, bool) or isinstance(actual, bool):
        return type(actual) is type(expected_value) and actual == expected_value
    if isinstance(expected_value, (int, float)):
        return type(actual) in {int, float} and math.isclose(actual, expected_value, rel_tol=1e-9, abs_tol=1e-7)
    if isinstance(expected_value, list):
        if not isinstance(actual, list) or len(actual) != len(expected_value):
            return False
        if unordered:
            import json
            return sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in actual) == sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in expected_value)
        return all(equivalent(a, e) for a, e in zip(actual, expected_value))
    return type(actual) is type(expected_value) and actual == expected_value


def wrong_value(value):
    if type(value) is bool:
        return not value
    if type(value) in {int, float}:
        return value + 1
    if isinstance(value, str):
        return value + '__wrong__'
    if isinstance(value, list):
        return value[:-1] if value else ['__wrong__']
    raise ValueError('No declared schema-safe negative for this answer type')
