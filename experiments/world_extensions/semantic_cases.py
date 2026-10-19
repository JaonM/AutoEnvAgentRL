"""Independent property fixtures for coherent business semantics, not benchmark records.

Amounts and IDs vary by seed. Expected effects are specified arithmetically from
the contract, never by calling candidate tools or the candidate verifier.
"""
import copy
import random


def path(p): return {'path': p}
def op(name, *args): return {'op': name, 'args': list(args)}
def check(name, a, b): return {'name': name, 'assert': op('eq', a, b)}
def step(tool, args, *checks, **options): return dict(tool=tool, args=args, checks=list(checks), **options)


def retail(tables, seed):
    rng = random.Random(seed);data = copy.deepcopy(tables)
    user = data['users'][0];uid = user['user_id'];products = data['products'][:2]
    gift = next(k for k,v in user['payment_methods'].items() if v['source']=='gift_card')
    card = next(k for k,v in user['payment_methods'].items() if v['source']=='credit_card')
    user['payment_methods'][gift]['balance'] = 1000
    order = data['orders'][0];oid = f'#PROPERTY{seed}'
    order.update(id=oid,order_id=oid,user_id=uid,status='pending',items=[],address=copy.deepcopy(user['address']))
    old,new=[],[]
    prices=[]
    for n,p in enumerate(products):
        variants=list(p['variants'].values())[:2]
        a,b = variants;price=rng.randrange(10,30)+n*50;replacement=price+rng.randrange(3,10)
        a.update(price=price,available=True);b.update(price=replacement,available=True)
        p['variants']={a['item_id']:a,b['item_id']:b}
        order['items'].append(dict(name=p['name'],product_id=p['product_id'],item_id=a['item_id'],price=price,options=copy.deepcopy(a['options'])))
        old.append(a['item_id']);new.append(b['item_id']);prices.append(replacement)
    total=sum(x['price'] for x in order['items'])
    order['payment_history']=[dict(transaction_type='payment',amount=total,payment_method_id=card)]
    user['orders']=[oid]
    data={'users':[user],'orders':[order],'products':products}
    cancel=dict(order_id=oid,reason='no longer needed')
    cases=[]
    paid=copy.deepcopy(data);paid['orders'][0]['payment_history']=[
        dict(transaction_type='payment',amount=100,payment_method_id=gift),
        dict(transaction_type='refund',amount=20,payment_method_id=gift),
        dict(transaction_type='payment',amount=40,payment_method_id=card)]
    cases.append(dict(id=f'retail-net-refund-{seed}',data=paid,steps=[step('cancel_pending_order',cancel,
        check('refund only outstanding amount',op('sum',path('after.orders.0.payment_history.*.amount')),280),
        check('gift refund is net',path(['after','users','0','payment_methods',gift,'balance']),1080)),
        step('cancel_pending_order',cancel,error='not_pending')]))
    cross=copy.deepcopy(data);cross['orders'][0]['payment_history']=[
        dict(transaction_type='payment',amount=100,payment_method_id=card),
        dict(transaction_type='refund',amount=20,payment_method_id=gift)]
    cases.append(dict(id=f'retail-cross-method-refund-{seed}',data=cross,steps=[step('cancel_pending_order',cancel,
        check('refund remaining global funding',path('after.orders.0.payment_history.2.amount'),80),
        check('refund original positive funding',path('after.orders.0.payment_history.2.payment_method_id'),card),
        check('previous refund not repeated',path(['after','users','0','payment_methods',gift,'balance']),1000))]))
    delta=sum(prices)-total
    cases.append(dict(id=f'retail-pair-prices-{seed}',data=data,steps=[step('modify_pending_order_items',
        dict(order_id=oid,item_ids=old,new_item_ids=new,payment_method_id=gift),
        check('each row uses its own variant price',path('after.orders.0.items.*.price'),prices),
        check('each row uses its own variant options',path('after.orders.0.items.*.options'),[p['variants'][v]['options'] for p,v in zip(products,new)]),
        check('charge exact total difference',path(['after','users','0','payment_methods',gift,'balance']),1000-delta))]))
    duplicated=copy.deepcopy(data);item=duplicated['orders'][0]['items'][0]
    duplicated['orders'][0]['items']=[copy.deepcopy(item),{**copy.deepcopy(item),'price':item['price']+2}]
    duplicate_delta=2*prices[0]-(2*item['price']+2)
    cases.append(dict(id=f'retail-duplicate-occurrences-{seed}',data=duplicated,steps=[step('modify_pending_order_items',
        dict(order_id=oid,item_ids=[old[0],old[0]],new_item_ids=[new[0],new[0]],payment_method_id=gift),
        check('consume each original occurrence once',path(['after','users','0','payment_methods',gift,'balance']),1000-duplicate_delta))]))
    cases.append(dict(id=f'retail-change-funding-cancel-{seed}',data=data,steps=[
        step('modify_pending_order_payment',dict(order_id=oid,payment_method_id=gift),check('new funding charged',path(['after','users','0','payment_methods',gift,'balance']),1000-total)),
        step('cancel_pending_order',cancel,check('funding round trip',path(['after','users','0','payment_methods',gift,'balance']),1000),
             check('only final funding refunded',op('len',path('after.orders.0.payment_history')),4))]))
    bad=copy.deepcopy(data);bad['orders'][0]['payment_history']=[dict(transaction_type='payment',amount=10,payment_method_id=gift),dict(transaction_type='payment',amount=20,payment_method_id='missing')]
    cases.append(dict(id=f'retail-atomic-refund-{seed}',data=bad,steps=[step('cancel_pending_order',cancel,error='payment_not_owned')]))
    returned=copy.deepcopy(data);returned['orders'][0]['status']='delivered'
    cases.append(dict(id=f'retail-request-boundary-{seed}',data=returned,steps=[step('return_delivered_order_items',dict(order_id=oid,item_ids=old,payment_method_id=card),
        check('request is queryable',path('after.orders.0.status'),'return requested'),check('request does not claim settlement',path('after.orders.0.payment_history'),returned['orders'][0]['payment_history']))]))
    return cases


def airline(tables, seed):
    rng=random.Random(seed);u=copy.deepcopy(tables['users'][0]);u['reservations']=[]
    gift=next(k for k,v in u['payment_methods'].items() if v['source']=='gift_card')
    card=next(k for k,v in u['payment_methods'].items() if v['source']=='credit_card')
    u['payment_methods'][gift]['amount']=10000
    base=rng.randrange(50,150);date='2024-06-02';flights=[]
    for n in range(2):
        f=copy.deepcopy(tables['flights'][0]);fid=f'PROPERTY{seed}X{n}'
        f.update(id=fid,flight_number=fid,origin='SFO',destination='JFK',dates={date:{'status':'available',
            'prices':{'basic_economy':base-10+n*30,'economy':base+n*30,'business':base+100+n*30},
            'available_seats':{'basic_economy':50,'economy':50,'business':50}}})
        flights.append(f)
    data={'users':[u],'reservations':[],'flights':flights}
    args=dict(user_id=u['user_id'],origin='SFO',destination='JFK',flight_type='one_way',cabin='economy',
        flights=[dict(flight_number=flights[0]['flight_number'],date=date)],passengers=copy.deepcopy(u['saved_passengers'][:1]),
        payment_methods=[dict(payment_id=gift,amount=base)],total_baggages=0,nonfree_baggages=0,insurance='no')
    rid=path('outputs.0.reservation_id');seat0=path(['after','flights','0','dates',date,'available_seats','economy']);seat1=path(['after','flights','1','dates',date,'available_seats','economy'])
    balance=path(['after','users','0','payment_methods',gift,'amount'])
    now='2024-06-01T11:30:00'
    book=step('book_reservation',args,check('booking consumes a seat',seat0,49),check('clock is supplied by runtime',path('after.reservations.0.created_at'),now))
    cases=[dict(id=f'airline-book-change-cancel-{seed}',data=data,now=now,steps=[book,
        step('update_reservation_flights',dict(reservation_id=rid,cabin='economy',flights=[dict(flight_number=flights[1]['flight_number'],date=date)],payment_id=gift),
             check('old seat released',seat0,50),check('new seat consumed',seat1,49),check('difference charged',balance,10000-base-30)),
        step('cancel_reservation',dict(reservation_id=rid),check('new seat restored',seat1,50),check('gift refund settled',balance,10000)),
        step('cancel_reservation',dict(reservation_id=rid),check('repeat cancellation is a no-op',path('after'),path('before')))])]
    crossargs=copy.deepcopy(args);crossargs['payment_methods']=[dict(payment_id=card,amount=base)]
    cases.append(dict(id=f'airline-cross-method-refund-{seed}',data=data,now=now,steps=[
        step('book_reservation',crossargs,check('original card payment',path('after.reservations.0.payment_history.0.amount'),base)),
        step('update_reservation_flights',dict(reservation_id=rid,cabin='basic_economy',flights=args['flights'],payment_id=gift),
             check('difference refunded to gift',balance,10010)),
        step('cancel_reservation',dict(reservation_id=rid),check('only remaining funding refunded',path('after.reservations.0.payment_history.2.amount'),10-base),
             check('gift refund not duplicated',balance,10010)),
        step('update_reservation_passengers',dict(reservation_id=rid,passengers=args['passengers']),error='Reservation already cancelled')]))
    cases.append(dict(id=f'airline-id-scale-{seed}',data=data,now=now,steps=[step('book_reservation',args,
        check('create all requested reservations',op('len',path('after.reservations')),i+1),
        {'name':'unique identifiers','assert':op('unique',path('after.reservations.*.reservation_id'))}) for i in range(12)]))
    cases.append(dict(id=f'airline-certificate-scale-{seed}',data=data,steps=[step('send_certificate',dict(user_id=u['user_id'],amount=20),
        check('certificate not silently overwritten',op('len',path('after.users.0.payment_methods')),len(u['payment_methods'])+i+1)) for i in range(12)]))
    empty=copy.deepcopy(data);empty['flights'][1]['dates'][date]['available_seats']['economy']=0
    cases.append(dict(id=f'airline-failed-change-atomic-{seed}',data=empty,now=now,steps=[book,
        step('update_reservation_flights',dict(reservation_id=rid,cabin='economy',flights=[dict(flight_number=flights[1]['flight_number'],date=date)],payment_id=gift),error='Not enough seats')]))
    repeated=copy.deepcopy(args);repeated['flights']*=2;repeated['payment_methods'][0]['amount']*=2
    cases.append(dict(id=f'airline-duplicate-segments-{seed}',data=data,steps=[step('book_reservation',repeated,error='*')]))
    return cases


def make(domain,tables,seeds=(7,19,41)):
    cases=[c for seed in seeds for c in globals()[domain](tables,seed)]
    return {'version':1,'profile':'coherent-v1','cases':cases}
