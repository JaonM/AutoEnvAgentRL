from copy import deepcopy
from collections import Counter
import json


def expected(data, facts, now):
    state = deepcopy(data)
    users = {u['user_id']: u for u in state['users']}
    orders = {o['order_id']: o for o in state['orders']}
    products = {p['product_id']: p for p in state['products']}
    user = next((u for u in users.values() if isinstance(facts.get('email'), str) and u['email'].lower() == facts['email'].lower()), None)
    if user is None and all(k in facts for k in ('first_name', 'last_name', 'zip')):
        user = next((u for u in users.values() if u['name']['first_name'].lower() == facts['first_name'].lower() and u['name']['last_name'].lower() == facts['last_name'].lower() and u['address']['zip'] == facts['zip']), None)
    answer = {}
    address_keys = ('address1', 'address2', 'city', 'state', 'country', 'zip')
    class Refusal(Exception):
        pass
    def require(ok, reason):
        if not ok:
            raise Refusal(reason)
    def consent(process, target):
        key = 'approve_' + process + '_' + target
        require(key in facts, 'approval_missing')
        require(facts[key] is True, 'approval_declined')
    def transaction(order, kind, amount, method):
        order['payment_history'].append(dict(transaction_type=kind, amount=amount, payment_method_id=method))
    def gift(method, amount):
        if method['source'] == 'gift_card':
            method['balance'] = round(method['balance'] + amount, 2)
    for i, goal in enumerate(facts.get('goals', []), 1):
        proc, p = goal['process'], goal.get('params', {})
        out = {'status': 'refused', 'reason': None}
        if proc in ('pending_items_change', 'delivered_exchange_request'):
            out['price_difference'] = None
        if proc == 'delivered_return_request':
            out['instructions_pending'] = False
        if proc == 'account_lookup':
            out['facts'] = []
        if proc == 'handover':
            out['message'] = ''
        try:
            require(user is not None, 'identity_not_verified')
            uid = user['user_id']
            if 'user_id' in p:
                require(p['user_id'] == uid, 'other_customer')
            o = orders.get(p.get('order_id'))
            if 'order_id' in p:
                require(p['order_id'] in user['orders'] and o is not None, 'record_not_found')
            if proc == 'account_lookup':
                typ, rid = p.get('record_type'), p.get('record_id', '')
                record = None
                if typ == 'profile':
                    require(not rid or rid == uid, 'other_customer')
                    record = user
                elif typ == 'order':
                    record = orders.get(rid)
                    require(rid in user['orders'], 'record_not_found')
                elif typ == 'product':
                    record = products.get(rid)
                elif typ == 'item':
                    record = next((v for product in products.values() for k, v in product['variants'].items() if k == rid), None)
                elif typ == 'product_types':
                    record = {x['name']: x['product_id'] for x in sorted(products.values(), key=lambda x: x['name'])}
                require(record is not None, 'record_not_found')
                selector = p.get('selector', '')
                if selector and typ == 'product_types':
                    require(selector in record, 'record_not_found')
                    record = {'name':selector, 'product_id':record[selector]}
                    typ = 'product'
                elif selector:
                    for component in selector.split('.'):
                        if isinstance(record, list):
                            require(component.isdigit() and int(component) < len(record), 'record_not_found')
                            record = record[int(component)]
                        else:
                            require(isinstance(record, dict) and component in record, 'record_not_found')
                            record = record[component]
                selected = p.get('fields') or {'profile':['name.first_name','name.last_name','email'],'order':['status'],'product':['name'],'item':['available','price'],'product_types':['product_name']}[typ]
                atoms = []
                if typ == 'product_types':
                    entries = [('product_name', name) for name in sorted(record)]
                else:
                    entries = []
                    for field in selected:
                        value = record
                        for component in field.split('.'):
                            require(isinstance(value, dict) and component in value, 'record_not_found')
                            value = value[component]
                        entries.extend((field, item) for item in (value if isinstance(value, list) else [value]))
                for field, value in entries:
                    require(value is None or isinstance(value, (str, bool, int, float)), 'tool_error')
                    atom = dict(field=field, string_value=None, number_value=None, boolean_value=None)
                    if isinstance(value, bool): atom['boolean_value'] = value
                    elif isinstance(value, (int,float)): atom['number_value'] = value
                    elif isinstance(value, str): atom['string_value'] = value
                    atoms.append(atom)
                out.update(status='answered', facts=atoms)
            elif proc == 'handover':
                require(not p.get('solvable', False), 'contract_ambiguity')
                out.update(status='transferred', message='YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.')
            elif proc == 'default_address_change':
                consent(proc, p['user_id'])
                user['address'] = {k:p[k] for k in address_keys}
                out['status'] = 'completed'
            else:
                require(o is not None, 'record_not_found')
                delivered = proc in ('delivered_return_request', 'delivered_exchange_request')
                if delivered:
                    require(o['status'] == 'delivered', 'not_delivered')
                else:
                    require(o['status'] != 'pending (item modified)', 'item_edit_lock')
                    require(o['status'] == 'pending', 'not_pending')
                methods = user['payment_methods']
                mid = p.get('payment_method_id')
                method = methods.get(mid)
                old = p.get('item_ids', [])
                new = p.get('new_item_ids', [])
                delta = 0
                variants = []
                if proc == 'cancel':
                    require(p.get('reason') in ('no longer needed', 'ordered by mistake'), 'invalid_cancel_reason')
                if proc == 'delivered_return_request':
                    require(method is not None, 'payment_not_owned')
                    require(method['source'] == 'gift_card' or (o['payment_history'] and mid == o['payment_history'][0]['payment_method_id']), 'invalid_refund_destination')
                if proc in ('pending_items_change', 'delivered_exchange_request', 'delivered_return_request'):
                    available = Counter(x['item_id'] for x in o['items'])
                    require(bool(old) and not (Counter(old) - available), 'items_not_found')
                if proc in ('pending_items_change', 'delivered_exchange_request'):
                    require(len(old) == len(new), 'list_length_mismatch')
                    available_items=list(range(len(o['items'])));selected_items=[]
                    for a,b in zip(old,new):
                        require(a != b, 'same_item')
                        selected_index=next(i for i in available_items if o['items'][i]['item_id']==a)
                        available_items.remove(selected_index);selected_items.append(selected_index)
                        item=o['items'][selected_index]
                        product = products.get(item['product_id'])
                        variant = product['variants'].get(b) if product else None
                        if variant is None:
                            exists = any(b in x['variants'] for x in products.values())
                            require(False, 'wrong_product' if exists else 'record_not_found')
                        require(variant['available'], 'unavailable')
                        variants.append(variant)
                        delta += variant['price'] - item['price']
                    if delivered:
                        delta = round(delta, 2)
                    require(method is not None, 'payment_not_owned')
                    require(method['source'] != 'gift_card' or method['balance'] >= delta, 'insufficient_gift_balance')
                if proc == 'payment_method_change':
                    require(method is not None, 'payment_not_owned')
                    h = o['payment_history']
                    require(len(h) == 1 and h[0]['transaction_type'] == 'payment', 'payment_history_not_single_payment')
                    require(mid != h[0]['payment_method_id'], 'same_payment_method')
                    require(method['source'] != 'gift_card' or method['balance'] >= h[0]['amount'], 'insufficient_gift_balance')
                consent(proc, p['order_id'])
                out['status'] = 'completed'
                if proc == 'shipping_address_change':
                    o['address'] = {k:p[k] for k in address_keys}
                elif proc == 'cancel':
                    history = deepcopy(o['payment_history'])
                    destinations=list(dict.fromkeys(entry['payment_method_id'] for entry in history))
                    balances={d:round(sum(e['amount']*(1 if e['transaction_type']=='payment' else -1) for e in history if e['payment_method_id']==d),2) for d in destinations}
                    require(round(sum(balances.values()),2)>=0,'invalid_payment_history')
                    offset=-sum(v for v in balances.values() if v<0)
                    for destination in destinations:
                        paid=sum(entry['amount'] for entry in history if entry['payment_method_id']==destination and entry['transaction_type']=='payment')
                        refunded=sum(entry['amount'] for entry in history if entry['payment_method_id']==destination and entry['transaction_type']=='refund')
                        amount=round(paid-refunded,2)
                        available=max(0,amount); consumed=min(available,offset)
                        amount=round(available-consumed,2); offset=round(offset-consumed,2)
                        if amount:
                            transaction(o, 'refund', amount, destination)
                            gift(methods[destination], amount)
                    o.update(status='cancelled', cancel_reason=p['reason'])
                elif proc == 'payment_method_change':
                    original = deepcopy(o['payment_history'][0])
                    transaction(o, 'payment', original['amount'], mid)
                    transaction(o, 'refund', original['amount'], original['payment_method_id'])
                    gift(method, -original['amount'])
                    gift(methods[original['payment_method_id']], original['amount'])
                elif proc == 'pending_items_change':
                    transaction(o, 'payment' if delta > 0 else 'refund', abs(delta), mid)
                    gift(method, -delta)
                    for index,b,variant in zip(selected_items,new,variants):
                        o['items'][index].update(item_id=b,price=variant['price'],options=deepcopy(variant['options']))
                    o['status'] = 'pending (item modified)'
                    out['price_difference'] = delta
                elif proc == 'delivered_return_request':
                    o.update(status='return requested', return_items=sorted(old), return_payment_method_id=mid)
                    out.update(status='requested', instructions_pending=True)
                elif proc == 'delivered_exchange_request':
                    o.update(status='exchange requested', exchange_items=sorted(old), exchange_new_items=sorted(new), exchange_payment_method_id=mid, exchange_price_difference=delta)
                    out.update(status='requested', price_difference=delta)
        except Refusal as e:
            out['reason'] = str(e)
        if out.get('status') == 'failed' and 'price_difference' in out:
            out['price_difference'] = None
        answer['goal_' + str(i)] = out
    return {'answer': answer, 'final_state': state}
