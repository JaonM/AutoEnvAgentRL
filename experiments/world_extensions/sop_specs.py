"""Domain-owned SOP declarations; engine and composition checks are industry neutral.

This registry is specification, never synthesized from candidate tool behavior.
Public API input types are copied from its caller-owned contract; effects and
scope boundaries below are explicit business design decisions.
"""
from copy import deepcopy


RETAIL_WRITES = {
    'cancel_pending_order': ['orders.*.status', 'orders.*.cancel_reason', 'orders.*.payment_history', 'users.*.payment_methods'],
    'modify_pending_order_address': ['orders.*.address'],
    'modify_user_address': ['users.*.address'],
    'modify_pending_order_items': ['orders.*.items', 'orders.*.status', 'orders.*.payment_history', 'users.*.payment_methods'],
    'modify_pending_order_payment': ['orders.*.payment_history', 'users.*.payment_methods'],
    'return_delivered_order_items': ['orders.*.status', 'orders.*.return_items', 'orders.*.return_payment_method_id'],
    'exchange_delivered_order_items': ['orders.*.status', 'orders.*.exchange_items', 'orders.*.exchange_new_items',
                                      'orders.*.exchange_payment_method_id', 'orders.*.exchange_price_difference'],
}
AIRLINE_WRITES = {
    'book_reservation': ['reservations', 'users.*.reservations', 'users.*.payment_methods', 'flights.*.dates'],
    'cancel_reservation': ['reservations.*.status', 'reservations.*.payment_history', 'users.*.payment_methods', 'flights.*.dates'],
    'update_reservation_flights': ['reservations.*.flights', 'reservations.*.cabin', 'reservations.*.payment_history',
                                  'users.*.payment_methods', 'flights.*.dates'],
    'update_reservation_passengers': ['reservations.*.passengers'],
    'update_reservation_baggages': ['reservations.*.total_baggages', 'reservations.*.nonfree_baggages',
                                   'reservations.*.payment_history', 'users.*.payment_methods'],
    'send_certificate': ['users.*.payment_methods'],
}

# Sampling targets narrow a method's customer scenario without changing the
# sandbox's public rules or treating every policy refusal as the same branch.
SAMPLING_PREDICATES = {
    'unavailable_exchange_then_return': [
        {'path':'process','equals':'delivered_exchange_request'},
        {'path':'observed.reason','equals':'unavailable'}],
}

EFFECTS = {
    'cancel_pending_order': 'Mark the selected pending order cancelled; refund outstanding signed funding once, update gift balance and append refund ledger entries.',
    'modify_pending_order_address': 'Replace only the selected pending order address; leave account default address unchanged.',
    'modify_user_address': 'Replace the account default address; leave existing order addresses unchanged.',
    'modify_pending_order_items': 'Resolve each selected occurrence, replace with its own variant price/options, settle signed price difference and set the once-only edit lock.',
    'modify_pending_order_payment': 'Charge the new owned method and refund the previous single payment; retain both ledger entries.',
    'return_delivered_order_items': 'Persist a queryable return request and destination. No shipment, receipt inspection or final refund settlement.',
    'exchange_delivered_order_items': 'Persist the selected exchange request and independently priced difference. No fulfillment or settlement is performed.',
    'book_reservation': 'Allocate a unique reservation ID, consume dated cabin seats, charge saved methods, link ownership and record creation using runtime time.',
    'cancel_reservation': 'Cancel once; release current dated occupancy and refund net outstanding payments, including paid certificate portion only.',
    'update_reservation_flights': 'Atomically release old and acquire new dated cabin seats, settle signed fare difference and replace current itinerary.',
    'update_reservation_passengers': 'Update passenger details without changing passenger count, ownership, occupancy or payment history.',
    'update_reservation_baggages': 'Update baggage counts and charge only newly purchased nonfree allowance using the current cabin/tier.',
    'send_certificate': 'Create a fresh saved certificate; ID uniqueness includes historical consumed certificates. Issuance eligibility is an agent obligation.',
}

PRECONDITIONS = {
    'cancel_pending_order': ['order exists and status is pending', 'reason is allowed', 'all positive net funding destinations belong to owner'],
    'modify_pending_order_address': ['order exists and is pending'],
    'modify_user_address': ['user exists'],
    'modify_pending_order_items': ['pending order without prior item edit', 'equal nonempty selections consume existing occurrences',
                                   'each replacement belongs to same product and is available', 'owned funding covers positive difference'],
    'modify_pending_order_payment': ['pending order', 'exactly one original payment', 'new method is distinct and owned', 'sufficient gift balance'],
    'return_delivered_order_items': ['order is delivered', 'selected items exist', 'refund destination is original method or owned gift card'],
    'exchange_delivered_order_items': ['order is delivered', 'matched selections exist', 'available same-product variants', 'owned funding for declared difference'],
    'book_reservation': ['valid distinct dated segments', 'enough seats in cabin', 'valid saved funding and nonnegative amounts', 'total payment equals quote'],
    'cancel_reservation': ['reservation exists', 'outstanding signed funding is nonnegative'],
    'update_reservation_flights': ['active reservation', 'valid distinct segments', 'enough net dated seats', 'valid funding for signed difference'],
    'update_reservation_passengers': ['active reservation', 'same passenger count'],
    'update_reservation_baggages': ['active reservation', 'valid saved funding'],
    'send_certificate': ['existing user', 'valid certificate amount'],
}


def method(intent, processes, *, kind='resource', causal=False, same=(), statuses=None, bindings=()):
    return dict(intent=intent, processes=processes,
                edges=[{'from': i, 'to': i+1, 'kind': kind, 'causal': causal} for i in range(len(processes)-1)],
                same=list(same), bindings=list(bindings), statuses=statuses or {},
                compensation='none; completed earlier operations persist if a later operation is refused')


def methods(domain):
    same_order = [[(0, 'order_id'), (1, 'order_id')]]
    same_reservation = [[(0, 'reservation_id'), (1, 'reservation_id')]]
    if domain == 'retail':
        return {
            'replace_funding_then_cancel': method('Customer changes funding, then later changes their mind and cancels that order.',
                ['payment_method_change', 'cancel'], kind='state', causal=True, same=same_order, statuses={'0':['completed'], '1':['completed']}),
            'correct_address_then_funding': method('Correct delivery details and funding for the same pending purchase; separate consents. Shared order requires serialization, not a causal-goal claim.',
                ['shipping_address_change', 'payment_method_change'], same=same_order, statuses={'0':['completed'], '1':['completed']}),
            'unavailable_exchange_then_return': method('Customer prefers an unavailable replacement; after that refusal chooses an eligible return and confirms the new request.',
                ['delivered_exchange_request', 'delivered_return_request'], kind='state', same=same_order,
                statuses={'0':['refused'], '1':['requested']}),
            'correct_address_then_verify': method('Correct the pending order address, then retrieve the updated address field on that order.',
                ['shipping_address_change', 'account_lookup'], kind='state', causal=True,
                same=[[(0,'order_id'),(1,'record_id')]], statuses={'0':['completed'], '1':['answered']}),
        }
    if domain == 'airline':
        return {
            'change_then_cancel': method('Replan the trip, then a later customer decision cancels its current itinerary and net payment.',
                ['itinerary_change','eligible_cancellation'], kind='state', causal=True, same=same_reservation, statuses={'0':['completed'],'1':['completed']}),
            'cabin_then_baggage': method('Change cabin, then calculate and buy extra baggage using the new allowance.',
                ['cabin_change','adding_baggage'], kind='state', causal=True, same=same_reservation, statuses={'0':['completed'],'1':['completed']}),
            'change_then_delay_compensation': method('Resolve the delayed trip by an itinerary change; customer then explicitly requests eligible delay compensation.',
                ['itinerary_change','requested_compensation'], kind='state', causal=True, same=same_reservation, statuses={'0':['completed'],'1':['completed']}),
            'cancel_then_delay_compensation': method('Cancel an eligible delayed trip before a separate explicit request for delay compensation.',
                ['eligible_cancellation','requested_compensation'], kind='state', causal=True, same=same_reservation, statuses={'0':['completed'],'1':['completed']}),
            'correct_passengers_then_baggage': method('Correct passenger spelling and buy extra baggage for the same booking; shared reservation, separate consent, no invented causal dependency.',
                ['passenger_list_update','adding_baggage'], same=same_reservation, statuses={'0':['completed'],'1':['completed']}),
            'compensate_then_book': method('Issue eligible requested compensation, then use that actual certificate to fund a new affordable trip. Never predict a certificate ID; saved credit funds may be needed if the method cannot cover the fare.',
                ['requested_compensation','constrained_booking'], kind='data', same=[[(0,'user_id'),(1,'user_id')]],
                statuses={'0':['completed'],'1':['completed']},
                bindings=[{'from':0,'source':'certificate_id','to':1,'target':'payment_id'}]),
            'book_then_inspect': method('Book a trip, then verify baggage on the reservation ID returned by the booking.',
                ['booking_creation','reservation_inspection'], kind='data', statuses={'0':['completed'],'1':['completed']},
                bindings=[{'from':0,'source':'reservation_id','to':1,'target':'reservation_id'}]),
        }
    raise ValueError('no caller-owned SOP specification for ' + domain)


def blueprint(domain, request):
    writes = RETAIL_WRITES if domain == 'retail' else AIRLINE_WRITES
    objects = ({'users': {'identity':'user_id', 'lifecycle':['active'], 'role':'owner and saved funding'},
                'orders': {'identity':'order_id', 'lifecycle':['pending','pending (item modified)','cancelled','processed','delivered','return requested','exchange requested']},
                'products': {'identity':'product_id', 'lifecycle':['catalogued'], 'role':'variants with boolean availability'}}
               if domain == 'retail' else
               {'users': {'identity':'user_id', 'lifecycle':['active'], 'role':'owner, saved funding and certificates'},
                'reservations': {'identity':'reservation_id', 'lifecycle':['active','cancelled'], 'role':'current itinerary plus signed funding history'},
                'flights': {'identity':'flight_number', 'lifecycle':['available','delayed','cancelled','flying','landed'], 'role':'date/cabin inventory; segment identity includes date'}})
    atoms = {}
    for spec in request['public_contract']['tools']:
        fn = spec['function']; name = fn['name']; inputs = deepcopy(fn['parameters'])
        allowed = writes.get(name, [])
        scopes = []
        for table, key in [('orders','order_id'), ('reservations','reservation_id'), ('users','user_id')]:
            if key in inputs.get('properties', {}) and any(p.split('.')[0] == table for p in allowed):
                scopes.append(dict(table=table,key=key,argument=key))
        atoms[name] = dict(tool=name, intent=EFFECTS.get(name, 'Read current business records or perform the documented deterministic query: '+name),
            policy={'source':'public_contract.policy and public_contract.tools.'+name, 'version':'sop-v1'},
            inputs=inputs, output={'type':['object','array','string','number','boolean','null']},
            preconditions=PRECONDITIONS.get(name,['arguments identify available public records or supported query']),
            agent_obligations=['establish customer identity and object ownership under public policy'] +
                (['check policy eligibility, explain exact proposal and obtain explicit consent before each write'] if allowed else []),
            reads=list(objects) if name not in ('calculate','transfer_to_human_agents') else [], writes=allowed,
            effects=[EFFECTS.get(name,'No database mutation; transfer is evidenced by tool trace only.' if name=='transfer_to_human_agents' else 'Return current record/query result.')],
            failure='atomic_rollback', retry=('idempotent after cancellation' if name=='cancel_reservation' else
                'do not blindly retry a write; inspect current state and ledger first' if allowed else 'repeatable read'),
            scope='Single API transaction. Multi-atom workflows have no implicit global rollback or compensation.',
            money_unit='public decimal dollars; no cents conversion', row_scopes=scopes)
    return {'version':1, 'profile':'sop-v1', 'objects':objects, 'atoms':atoms, 'methods':methods(domain),
            'limits':['Finite acyclic plans only; loops require an explicitly bounded expansion.',
                      'Return/exchange stops at a durable request; human handover stops at a trace receipt.',
                      'Method catalogue is extensible and does not claim coverage of every possible business SOP.']}


def make_behavior(domain, tables, seeds=(7,19,41)):
    from world_extensions.semantic_cases import make, path, op, step, check
    contract=make(domain,tables,seeds=seeds)
    for seed in seeds:
        if domain=='retail':
            base=next(c for c in contract['cases'] if c['id']==f'retail-pair-prices-{seed}')
            data=deepcopy(base['data']);u=data['users'][0];o=data['orders'][0]
            address=dict(address1='42 Contract Avenue',address2='',city='Example',state='CA',country='USA',zip='94000')
            contract['cases'].append(dict(id=f'retail-address-scope-{seed}',data=data,steps=[
                step('modify_pending_order_address',dict(order_id=o['order_id'],**address),
                     check('order address updated',path('after.orders.0.address'),address),
                     check('profile remains unchanged',path('after.users.0.address'),u['address'])),
                step('modify_user_address',dict(user_id=u['user_id'],**{**address,'city':'Second'}),
                     check('profile address updated',path('after.users.0.address.city'),'Second'),
                     check('order address remains independent',path('after.orders.0.address'),address))]))
            exchange=deepcopy(base['steps'][0]);exchange['tool']='exchange_delivered_order_items'
            delta=sum(p['variants'][i]['price'] for p,i in zip(data['products'],exchange['args']['new_item_ids']))-sum(x['price'] for x in o['items'])
            exchange['checks']=[check('request stored',path('after.orders.0.status'),'exchange requested'),
                check('difference priced independently',path('after.orders.0.exchange_price_difference'),delta),
                check('no immediate item replacement',path('after.orders.0.items'),o['items']),
                check('no immediate settlement',path('after.orders.0.payment_history'),o['payment_history'])]
            delivered=deepcopy(data);delivered['orders'][0]['status']='delivered'
            contract['cases'].append(dict(id=f'retail-exchange-boundary-{seed}',data=delivered,steps=[exchange]))
            unavailable=deepcopy(delivered)
            for p,i in zip(unavailable['products'],exchange['args']['new_item_ids']):p['variants'][i]['available']=False
            contract['cases'].append(dict(id=f'retail-fallback-{seed}',data=unavailable,steps=[
                step('exchange_delivered_order_items',exchange['args'],error='unavailable'),
                step('return_delivered_order_items',{k:v for k,v in exchange['args'].items() if k!='new_item_ids'},
                     check('alternative request recorded',path('after.orders.0.status'),'return requested'))]))
        else:
            base=next(c for c in contract['cases'] if c['id']==f'airline-book-change-cancel-{seed}')
            data=deepcopy(base['data']);book=deepcopy(base['steps'][0]);args=book['args']
            uid=args['user_id'];paid=args['payment_methods'][0]['amount'];pid=args['payment_methods'][0]['payment_id']
            passenger=deepcopy(args['passengers']);passenger[0]['first_name']='Corrected'
            contract['cases'].append(dict(id=f'airline-passengers-baggage-{seed}',data=data,now=base['now'],steps=[book,
                step('update_reservation_passengers',dict(reservation_id=path('outputs.0.reservation_id'),passengers=passenger),
                     check('passenger correction',path('after.reservations.0.passengers'),passenger),
                     check('occupancy unchanged',path('after.flights'),path('before.flights'))),
                step('update_reservation_baggages',dict(reservation_id=path('outputs.0.reservation_id'),total_baggages=2,nonfree_baggages=1,payment_id=pid),
                     check('baggage updated',path('after.reservations.0.total_baggages'),2),
                     check('one paid bag charged',path(['after','users','0','payment_methods',pid,'amount']),10000-paid-50))]))
            certificate=op('get',op('split',path('outputs.0'),' '),1)
            certbook=deepcopy(book);certbook['args']['payment_methods']=[{'payment_id':certificate,'amount':paid}]
            contract['cases'].append(dict(id=f'airline-certificate-binding-{seed}',data=data,now=base['now'],steps=[
                step('send_certificate',dict(user_id=uid,amount=paid+20),
                     check('certificate added',op('len',path('after.users.0.payment_methods')),len(data['users'][0]['payment_methods'])+1)),
                certbook,
                step('cancel_reservation',dict(reservation_id=path('outputs.1.reservation_id')),
                     check('refund is paid portion',path('after.reservations.0.payment_history.1.amount'),-paid))]))
    return contract


def executable_contract(behavior):
    """Compile independently authored fixtures to DAGs, preserving every assertion.

    Ordering in a sequential fixture is a conservative resource dependency;
    explicit output references add data edges. It is never labelled as causal
    evidence merely because two literal argument values happen to coincide.
    """
    from env_factory.sop import references
    cases = []
    for case in behavior['cases']:
        def rewrite(value):
            if isinstance(value, dict):
                if set(value) == {'path'}:
                    parts=value['path'].split('.') if isinstance(value['path'],str) else list(value['path'])
                    if parts and parts[0]=='outputs': parts[1]='n'+str(parts[1])
                    return {'path':parts}
                return {k:rewrite(v) for k,v in value.items()}
            return [rewrite(v) for v in value] if isinstance(value,list) else value
        nodes=[];edges=[]
        for i,step in enumerate(case['steps']):
            node={'id':f'n{i}', 'atom':step['tool'], **rewrite({k:v for k,v in step.items() if k!='tool'})}
            node['checks']=node.get('checks',[])+rewrite(behavior.get('invariants',[]))
            if i:edges.append({'from':f'n{i-1}','to':f'n{i}','kind':'resource'})
            for predecessor in set(references([node.get('args',{}),node.get('when')])):
                edges.append({'from':predecessor,'to':f'n{i}','kind':'data'})
            nodes.append(node)
        cases.append({**{k:deepcopy(v) for k,v in case.items() if k not in ('steps','invariants')},
                      'invariants':rewrite(case.get('invariants',[])), 'graph':{'nodes':nodes,'edges':edges}})
        if case['id'].startswith('retail-fallback-'):
            nodes[1]['when']={'op':'eq','args':[{'path':'outputs.n0.error'},'unavailable']}
            edges.append({'from':'n0','to':'n1','kind':'data'})
        if case['id'].startswith('airline-book-change-cancel-'):
            book=deepcopy(nodes[0]);uid=book['args']['user_id']
            parallel={'nodes':[book,
                {'id':'user','atom':'get_user_details','args':{'user_id':uid},
                 'checks':[{'name':'linked owner','assert':{'op':'contains','args':[{'path':'result.reservations'},{'path':'outputs.n0.reservation_id'}]}}]},
                {'id':'reservation','atom':'get_reservation_details','args':{'reservation_id':{'path':'outputs.n0.reservation_id'}},
                 'checks':[{'name':'same booking','assert':{'op':'eq','args':[{'path':'result.user_id'},uid]}}]}],
                'edges':[{'from':'n0','to':'user','kind':'state'},{'from':'n0','to':'reservation','kind':'data'}]}
            cases.append(dict(id=case['id']+'-parallel-reads',data=deepcopy(case['data']),now=case['now'],graph=parallel))
    return {'version':1,'cases':cases,'oracle':'caller-owned arithmetic and fixtures; never candidate observations'}
