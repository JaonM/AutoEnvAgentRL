"""Original retail/airline data recipes; no benchmark database or task inputs.

Domain semantics stay here. The production helpers supply seeded budgets,
identity rewriting, integrity checks, profiling and bounded author context.
"""
from collections import Counter
import copy
from datetime import datetime, timedelta
import itertools
import random

from env_factory.production.world_data import allocate, remap


def new_customer(template, domain, ordinal, rng):
    old = template['user_id']
    uid = f"{domain}_cedar_{ordinal:05d}"
    mapping = {old: uid}
    for j, key in enumerate(template['payment_methods']):
        mapping[key] = f"{template['payment_methods'][key]['source']}_sx_{domain}_{ordinal:05d}_{j}"
    user = remap(template, mapping)
    surname = ['Alderford', 'Birchwater', 'Cedarwick', 'Elmbridge', 'Foxmere', 'Hazelbrook',
               'Ivyford', 'Juniperhill', 'Larchfield', 'Maplewood', 'Oakridge', 'Pinehaven',
               'Reedbank', 'Willowmere', 'Ashgrove', 'Brookvale', 'Ferncliff', 'Hawthorn'][ordinal % 18]
    user['name']['last_name'] = surname
    user['email'] = f'{uid}@synthetic-world.test'
    address = copy.deepcopy(user['address'])
    address.update(address1=f'{100 + ordinal} {surname} Lane', address2='' if ordinal % 3 else f'Unit {ordinal % 17 + 1}',
                   zip=f'{70000 + ordinal:05d}')
    user['address'] = address
    for method in user['payment_methods'].values():
        if 'last_four' in method:
            method['last_four'] = f'{(ordinal * 37) % 10000:04d}'
        if method['source'] in ('gift_card', 'certificate'):
            field = 'balance' if domain == 'retail' else 'amount'
            method[field] = rng.choice([0, 15, 50, 100, 250, 700, 1400])
    if domain == 'airline':
        user['membership'] = rng.choice(['regular', 'regular', 'silver', 'gold'])
        for passenger in user['saved_passengers']:
            passenger['last_name'] = surname
        user['reservations'] = []
    else:
        user['orders'] = []
    return user


def retail(tables, targets, seed):
    tables = copy.deepcopy(tables)
    rng = random.Random(seed)
    original_users = list(tables['users'])
    counts = allocate(targets['variants'], len(tables['products']), minimum=6, maximum=20, rng=rng)
    serial = 0
    colors = ['sand', 'plum', 'olive', 'slate', 'rose', 'teal', 'cream', 'bronze', 'charcoal', 'amber', 'pine', 'pearl']
    sizes = ['mini', 'compact', 'standard', 'wide', 'tall', 'extended']
    for product, count in zip(tables['products'], counts):
        options = list(itertools.product(colors, sizes));rng.shuffle(options)
        base = min(v['price'] for v in product['variants'].values())
        for j in range(count - len(product['variants'])):
            item = str(9500000000 + serial);serial += 1
            color, size = options[j]
            product['variants'][item] = {'item_id': item, 'options': {'color': color, 'size': size},
                'available': rng.random() > .2, 'price': round(base + rng.randrange(1, 160) * .25, 2)}
    needed = targets['users'] - len(original_users)
    new_counts = allocate(targets['orders'] - len(tables['orders']), needed, maximum=9, rng=rng,
                          weights=[12, 16, 15, 9, 6, 4, 2, 1, 1, 1])
    prototypes = {r['status']: r for r in tables['orders']}
    index = 0
    for ordinal, count in enumerate(new_counts, 1):
        user = new_customer(original_users[(ordinal - 1) % len(original_users)], 'retail', ordinal, rng)
        tables['users'].append(user)
        for _ in range(count):
            index += 1;oid = f'#SX{7100000 + index}'
            status = rng.choices(['pending', 'delivered', 'processed', 'cancelled', 'pending (item modified)'],
                                 weights=[40, 35, 10, 10, 5])[0]
            order = copy.deepcopy(prototypes.get(status, prototypes['pending']))
            order.update(id=oid, order_id=oid, user_id=user['user_id'], address=copy.deepcopy(user['address']),
                         status=status, items=[], fulfillments=[], cancel_reason=None,
                         exchange_items=None, exchange_new_items=None, exchange_payment_method_id=None,
                         exchange_price_difference=None, return_items=None, return_payment_method_id=None)
            for _ in range(rng.choices([1, 2, 3, 4, 5], weights=[15, 25, 30, 20, 10])[0]):
                product = rng.choice(tables['products']);variant = rng.choice(list(product['variants'].values()))
                order['items'].append({'name': product['name'], 'product_id': product['product_id'],
                                       'item_id': variant['item_id'], 'price': variant['price'],
                                       'options': copy.deepcopy(variant['options'])})
            amount = round(sum(i['price'] for i in order['items']), 2)
            card = next(k for k, v in user['payment_methods'].items() if v['source'] == 'credit_card')
            order['payment_history'] = [{'transaction_type': 'payment', 'amount': amount, 'payment_method_id': card}]
            if status == 'cancelled':
                order['cancel_reason'] = rng.choice(['no longer needed', 'ordered by mistake'])
                order['payment_history'].append({'transaction_type': 'refund', 'amount': amount, 'payment_method_id': card})
            elif index % 7 == 0:
                other = next(k for k in user['payment_methods'] if k != card)
                order['payment_history'] += [{'transaction_type': 'refund', 'amount': amount, 'payment_method_id': card},
                                            {'transaction_type': 'payment', 'amount': amount, 'payment_method_id': other}]
            if status in ('delivered', 'processed'):
                order['fulfillments'] = [{'tracking_id': [f'TR-SX-{index:06d}'],
                                          'item_ids': [i['item_id'] for i in order['items']]}]
            user['orders'].append(oid);tables['orders'].append(order)
    return tables


def airline(tables, targets, seed):
    tables = copy.deepcopy(tables);rng = random.Random(seed)
    airports = 'SFO JFK LAX ORD DFW DEN SEA ATL MIA BOS PHX IAH LAS MCO EWR CLT MSP DTW PHL LGA'.split()
    routes = {(f['origin'], f['destination']) for f in tables['flights']}
    # Bidirectional ring guarantees every airport has inbound/outbound connectivity.
    for a, b in zip(airports, airports[1:] + airports[:1]):
        routes.update([(a, b), (b, a)])
    candidates = [(a, b) for a in airports for b in airports if a != b and (a, b) not in routes]
    rng.shuffle(candidates)
    routes.update(candidates[:targets['routes'] - len(routes)])
    routes = sorted(routes)
    existing = {(f['origin'], f['destination']) for f in tables['flights']}
    new_routes = [r for r in routes if r not in existing]
    route_cycle = new_routes + routes
    exemplar = tables['flights'][0]
    for i in range(targets['flights'] - len(tables['flights'])):
        origin, destination = route_cycle[i % len(route_cycle)]
        hour = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19])
        duration = rng.choice([1, 2, 3, 4])
        flight = copy.deepcopy(exemplar)
        flight.update(id=f'SX{8100 + i}', flight_number=f'SX{8100 + i}', origin=origin, destination=destination,
                      scheduled_departure_time_est=f'{hour:02d}:00:00',
                      scheduled_arrival_time_est=f'{hour + duration:02d}:00:00', dates={})
        tables['flights'].append(flight)
    dates = [f'2024-05-{day:02d}' for day in range(1, targets['dates'] + 1)]
    for flight in tables['flights']:
        for date in dates:
            if date in flight['dates']:
                continue
            base = rng.randrange(65, 430)
            if date < '2024-05-15':
                entry = {'status': rng.choice(['landed', 'landed', 'cancelled']),
                         'actual_departure_time_est': date + 'T' + flight['scheduled_departure_time_est'],
                         'actual_arrival_time_est': date + 'T' + flight['scheduled_arrival_time_est']}
            else:
                entry = {'status': 'cancelled' if rng.random() < .06 else 'available',
                         'prices': {'basic_economy': base, 'economy': base + rng.randrange(30, 100),
                                    'business': base + rng.randrange(150, 500)},
                         'available_seats': {c: rng.choice([0, 1, 2, 3, 5, 8, 15, 30])
                                             for c in ('basic_economy', 'economy', 'business')}}
            flight['dates'][date] = entry
    original_users = list(tables['users'])
    count = targets['users'] - len(original_users)
    counts = allocate(targets['reservations'] - len(tables['reservations']), count, maximum=13, rng=rng,
                      weights=[4, 5, 8, 12, 15, 12, 8, 5, 3, 2, 1, 1, 1, 1])
    connections = [(a, b) for a in tables['flights'] for b in tables['flights']
                   if a['destination'] == b['origin'] and a['origin'] != b['destination']
                   and '+1' not in a['scheduled_arrival_time_est']
                   and b['scheduled_departure_time_est'] > a['scheduled_arrival_time_est']]
    paths = [(f,) for f in tables['flights']] + connections
    indexed = {}
    for path in paths:
        indexed.setdefault((path[0]['origin'], path[-1]['destination']), []).append(path)
    index = 0
    for ordinal, number in enumerate(counts, 1):
        user = new_customer(original_users[(ordinal - 1) % len(original_users)], 'airline', ordinal, rng)
        tables['users'].append(user)
        for _ in range(number):
            index += 1;rid = f'SR{51000 + index}'
            day = rng.randrange(18, 25);date = f'2024-05-{day:02d}'
            desired = 1 if index % 3 == 0 else 2
            eligible = [p for p in paths if len(p) == desired and
                        all(f['dates'][date]['status'] == 'available' for f in p)]
            chosen = rng.choice(eligible);legs = [(f, date) for f in chosen]
            origin, destination = chosen[0]['origin'], chosen[-1]['destination']
            return_date = f'2024-05-{day + rng.randrange(2, 6):02d}'
            back = [p for p in indexed.get((destination, origin), [])
                    if all(f['dates'][return_date]['status'] == 'available' for f in p)]
            roundtrip = index % 2 == 0 and bool(back)
            if roundtrip:
                legs += [(f, return_date) for f in rng.choice(back)]
            cabin = rng.choice(['basic_economy', 'economy', 'economy', 'business'])
            n = rng.randrange(1, 6);passengers = copy.deepcopy(user['saved_passengers'][:n])
            n = len(passengers)
            free = n * ({'regular': 0, 'silver': 1, 'gold': 2}[user['membership']] +
                        {'basic_economy': 0, 'economy': 1, 'business': 2}[cabin])
            bags = rng.randrange(0, free + 4);paid = max(0, bags - free);insured = rng.choice(['yes', 'no'])
            fs = [{'flight_number': f['flight_number'], 'origin': f['origin'], 'destination': f['destination'],
                   'date': d, 'price': f['dates'][d]['prices'][cabin]} for f, d in legs]
            amount = sum(f['price'] for f in fs) * n + 50 * paid + 30 * n * (insured == 'yes')
            card = next(k for k, v in user['payment_methods'].items() if v['source'] == 'credit_card')
            payments = [{'payment_id': card, 'amount': amount}]
            if index % 5 == 0:
                gift = next(k for k, v in user['payment_methods'].items() if v['source'] == 'gift_card')
                split = min(amount, rng.choice([10, 25, 50]))
                payments = [{'payment_id': gift, 'amount': split}, {'payment_id': card, 'amount': amount - split}]
            created = datetime.fromisoformat('2024-05-15T15:00:00') - timedelta(hours=rng.choice([1, 6, 23, 24, 25, 48, 168]))
            reservation = {'id': rid, 'reservation_id': rid, 'user_id': user['user_id'], 'origin': origin,
                'destination': destination, 'flight_type': 'round_trip' if roundtrip else 'one_way',
                'cabin': cabin, 'flights': fs, 'passengers': passengers, 'payment_history': payments,
                'created_at': created.isoformat(), 'total_baggages': bags, 'nonfree_baggages': paid,
                'insurance': insured, 'status': None}
            user['reservations'].append(rid);tables['reservations'].append(reservation)
    return tables


def contract(domain, targets):
    child, child_key = ('orders', 'order_id') if domain == 'retail' else ('reservations', 'reservation_id')
    catalog, catalog_key = ('products', 'product_id') if domain == 'retail' else ('flights', 'flight_number')
    refs = [{'table': child, 'path': 'user_id', 'target_table': 'users', 'target_path': 'user_id'},
            {'table': 'users', 'path': child + '.*', 'target_table': child, 'target_path': child_key},
            {'table': child, 'path': 'items.*.product_id' if domain == 'retail' else 'flights.*.flight_number',
             'target_table': catalog, 'target_path': catalog_key}]
    if domain == 'retail':
        refs.append({'table': child, 'path': 'items.*.item_id', 'target_table': catalog, 'target_path': 'variants.*.item_id'})
    return {'counts': {t: targets[t] for t in ('users', child, catalog)},
            'unique': {'users': ['id', 'user_id', 'email', 'payment_methods.*.id'], child: ['id', child_key],
                       catalog: ['id', catalog_key] + (['variants.*.item_id'] if domain == 'retail' else [])},
            'references': refs, 'inverse': [{'owner_table': 'users', 'owner_key': 'user_id', 'owner_list': child,
                'child_table': child, 'child_owner': 'user_id', 'child_key': child_key}]}


def semantics(domain, tables):
    errors = [];users = {u['user_id']: u for u in tables['users']}
    if domain == 'retail':
        products = {p['product_id']: p for p in tables['products']}
        for order in tables['orders']:
            for item in order['items']:
                if item['item_id'] not in products[item['product_id']]['variants']:
                    errors.append('item belongs to another product')
            if any(p['payment_method_id'] not in users[order['user_id']]['payment_methods'] for p in order['payment_history']):
                errors.append('payment belongs to another customer')
            if order['id'].startswith('#SX'):
                total = sum(i['price'] for i in order['items'])
                net = sum(p['amount'] * (1 if p['transaction_type'] == 'payment' else -1) for p in order['payment_history'])
                if abs(net - (0 if order['status'] == 'cancelled' else total)) > .001:
                    errors.append('order balance mismatch')
    else:
        flights = {f['flight_number']: f for f in tables['flights']}
        for r in tables['reservations']:
            for leg in r['flights']:
                f = flights[leg['flight_number']]
                if leg['date'] not in f['dates'] or (leg['origin'], leg['destination']) != (f['origin'], f['destination']):
                    errors.append('reservation leg mismatch')
            # Redeemed certificates are consumed and disappear from saved methods;
            # their historical payment entries must remain intact.
            if any(p['payment_id'] not in users[r['user_id']]['payment_methods']
                   and not p['payment_id'].startswith('certificate_') for p in r['payment_history']):
                errors.append('payment belongs to another customer')
            if r['id'].startswith('SR'):
                n = len(r['passengers']);u = users[r['user_id']]
                free = n * ({'regular': 0, 'silver': 1, 'gold': 2}[u['membership']] +
                            {'basic_economy': 0, 'economy': 1, 'business': 2}[r['cabin']])
                amount = n * sum(f['price'] for f in r['flights']) + 50 * r['nonfree_baggages'] + 30 * n * (r['insurance'] == 'yes')
                if r['nonfree_baggages'] != max(0, r['total_baggages'] - free) or sum(p['amount'] for p in r['payment_history']) != amount:
                    errors.append('reservation baggage/payment mismatch')
                legs = r['flights']
                if legs[0]['origin'] != r['origin'] or legs[-1]['destination'] != (r['origin'] if r['flight_type'] == 'round_trip' else r['destination']):
                    errors.append('reservation endpoints mismatch')
                for a, b in zip(legs, legs[1:]):
                    if a['destination'] != b['origin'] or a['date'] > b['date']:
                        errors.append('disconnected itinerary')
                    if a['date'] == b['date'] and flights[a['flight_number']]['scheduled_arrival_time_est'] >= flights[b['flight_number']]['scheduled_departure_time_est']:
                        errors.append('unreachable connection')
    return errors


def metrics(domain):
    child = 'orders' if domain == 'retail' else 'reservations'
    result = [{'name': 'records_per_customer', 'table': 'users', 'path': child, 'measure': 'length'},
              {'name': 'payment_history_length', 'table': child, 'path': 'payment_history', 'measure': 'length'}]
    if domain == 'retail':
        result += [{'name': 'variants_per_product', 'table': 'products', 'path': 'variants', 'measure': 'length'},
                   {'name': 'items_per_order', 'table': 'orders', 'path': 'items', 'measure': 'length'},
                   {'name': 'order_status', 'table': 'orders', 'path': 'status'},
                   {'name': 'variant_available', 'table': 'products', 'path': 'variants.*.available'}]
    else:
        result += [{'name': n, 'table': t, 'path': p, **({'measure': 'length'} if length else {})} for n, t, p, length in [
            ('membership', 'users', 'membership', False), ('legs', 'reservations', 'flights', True),
            ('passengers', 'reservations', 'passengers', True), ('cabin', 'reservations', 'cabin', False),
            ('insurance', 'reservations', 'insurance', False), ('flight_type', 'reservations', 'flight_type', False),
            ('dates_per_flight', 'flights', 'dates', True), ('seats', 'flights', 'dates.*.available_seats.*', False)]]
    return result
