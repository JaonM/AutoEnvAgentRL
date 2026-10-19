from copy import deepcopy
import re
from datetime import datetime, timedelta
from collections import Counter
from env_factory.runtime import write, BusinessError

# Public policy calls schedules local times, while public field names say EST.
# Preserve source strings and comparisons; use the stipulated fixed EST clock.


def _reject(reason, message):
    error=BusinessError(message)
    error.reason=reason
    error.is_public_tool_error=True
    raise error


def _row(db, table, key):
    col = {'users':'user_id','reservations':'reservation_id','flights':'flight_number'}[table]
    for r in db[table]:
        if r[col] == key:
            return r
    _reject({'users':'user_not_found','reservations':'reservation_not_found','flights':'not_available'}[table], {'users':'User','reservations':'Reservation','flights':'Flight'}[table] + f' {key} not found')


def _public(r):
    r = deepcopy(r)
    r.pop('id', None)
    if 'payment_methods' in r:
        for p in r['payment_methods'].values():
            if 'amount' in p:
                p['amount'] = float(p['amount'])
    return r


def _instance(db, f):
    row = _row(db, 'flights', f['flight_number'])
    if f['date'] not in row['dates']:
        _reject('not_available', f"Flight {f['flight_number']} not found on date {f['date']}")
    return row, row['dates'][f['date']]


def _priced(db, f, cabin, count):
    row, d = _instance(db, f)
    if d['status'] != 'available':
        _reject('not_available', f"Flight {f['flight_number']} not available on date {f['date']}")
    if d['available_seats'][cabin] < count:
        _reject('insufficient_seats', f"Not enough seats on flight {f['flight_number']}")
    return dict(flight_number=f['flight_number'], date=f['date'], origin=row['origin'], destination=row['destination'], price=d['prices'][cabin])


def _pay(u, payment_id, amount):
    p = u['payment_methods'].get(payment_id)
    if p is None:
        _reject('payment_not_saved', 'Payment method not found')
    if p['source'] == 'certificate':
        _reject('certificate_not_allowed', 'Certificate cannot be used to update reservation')
    if p['source'] == 'gift_card':
        if p['amount'] < amount:
            _reject('insufficient_gift_balance', 'Gift card balance is not enough')
        p['amount'] = float(p['amount'] - amount)
    return {'payment_id': payment_id, 'amount': amount} if amount else None


def _search(db, origin, destination, date, leave_after=None):
    result = []
    for f in db['flights']:
        d = f['dates'].get(date, {})
        if (origin is None or f['origin']==origin) and (destination is None or f['destination']==destination) and d.get('status')=='available' and (leave_after is None or f['scheduled_departure_time_est'] >= leave_after):
            result.append({**{k:deepcopy(f[k]) for k in ('flight_number','origin','destination','scheduled_departure_time_est','scheduled_arrival_time_est')}, **deepcopy(d), 'date':None})
    return result


def _arithmetic(expression):
    # Match only the public arithmetic alphabet; never execute Python source.
    # Whitespace separates tokens and cannot concatenate otherwise invalid numbers.
    tokens=[]
    cursor=0
    for match in re.finditer(r'\*\*|//|[0-9]+(?:\.[0-9]*)?|\.[0-9]+|[()+*/-]', expression):
        if any(c!=' ' for c in expression[cursor:match.start()]):
            raise BusinessError('Invalid expression')
        tokens.append(match.group())
        cursor=match.end()
    if any(c!=' ' for c in expression[cursor:]):
        raise BusinessError('Invalid expression')
    position=[0]

    def peek():
        return tokens[position[0]] if position[0]<len(tokens) else None

    def atom():
        token=peek()
        if token is None:
            raise BusinessError('Invalid expression')
        position[0]+=1
        if token=='(':
            result=expression_value()
            if peek()!=')':
                raise BusinessError('Invalid expression')
            position[0]+=1
            return result
        if not (token[0].isdigit() or token[0]=='.'):
            raise BusinessError('Invalid expression')
        if '.' in token:
            return float(token)
        if len(token)>1 and token.startswith('0') and any(c!='0' for c in token):
            raise BusinessError('Invalid expression')
        return int(token)

    def power():
        result=atom()
        if peek()=='**':
            position[0]+=1
            result=result**factor()
        return result

    def factor():
        if peek() in ('+','-'):
            sign=peek()
            position[0]+=1
            result=factor()
            return result if sign=='+' else -result
        return power()

    def term():
        result=factor()
        while peek() in ('*','/','//'):
            op=peek()
            position[0]+=1
            right=factor()
            result=result*right if op=='*' else result/right if op=='/' else result//right
        return result

    def expression_value():
        result=term()
        while peek() in ('+','-'):
            op=peek()
            position[0]+=1
            right=term()
            result=result+right if op=='+' else result-right
        return result

    result=expression_value()
    if position[0]!=len(tokens):
        raise BusinessError('Invalid expression')
    return result


def _identifier(db, certificate=False):
    if certificate:
        used={key for u in db['users'] for key in u['payment_methods']}
        used.update(p['payment_id'] for r in db['reservations'] for p in r['payment_history'])
        prefix='certificate_coherent_'
    else:
        used={r['reservation_id'] for r in db['reservations']};prefix='NR'
    index=1
    while prefix+str(index).zfill(8) in used: index+=1
    return prefix+str(index).zfill(8)


def _inventory(db, old, new, count, old_cabin, new_cabin):
    changes=Counter()
    for f in old: changes[(f['flight_number'],f['date'],old_cabin)]+=count
    for f in new: changes[(f['flight_number'],f['date'],new_cabin)]-=count
    for (number,date,cabin),delta in changes.items():
        inst=_instance(db,dict(flight_number=number,date=date))[1]
        if 'available_seats' not in inst:
            if delta<0: _reject('not_available','Flight inventory unavailable')
            continue
        if inst['available_seats'][cabin]+delta<0: _reject('insufficient_seats',f'Not enough seats on flight {number}')
    for (number,date,cabin),delta in changes.items():
        inst=_instance(db,dict(flight_number=number,date=date))[1]
        if 'available_seats' in inst: inst['available_seats'][cabin]+=delta


def _dispatch(name, db, now, a):
    if name == 'get_user_details': return _public(_row(db,'users',a['user_id']))
    if name == 'get_reservation_details': return _public(_row(db,'reservations',a['reservation_id']))
    if name == 'get_flight_status': return _instance(db,a)[1]['status']
    if name == 'search_direct_flight': return _search(db,**a)
    if name == 'search_onestop_flight':
        out=[]
        for f in _search(db,a['origin'],None,a['date']):
            f['date']=a['date']
            date2=(datetime.fromisoformat(a['date'])+timedelta(days=1)).date().isoformat() if '+1' in f['scheduled_arrival_time_est'] else a['date']
            for g in _search(db,f['destination'],a['destination'],date2,f['scheduled_arrival_time_est'].split('+')[0]):
                g['date']=date2
                out.append([f,g])
        return out
    if name == 'list_all_airports':
        codes='SFO JFK LAX ORD DFW DEN SEA ATL MIA BOS PHX IAH LAS MCO EWR CLT MSP DTW PHL LGA'.split()
        cities=['San Francisco','New York','Los Angeles','Chicago','Dallas','Denver','Seattle','Atlanta','Miami','Boston','Phoenix','Houston','Las Vegas','Orlando','Newark','Charlotte','Minneapolis','Detroit','Philadelphia','LaGuardia']
        return [dict(iata=x,city=y) for x,y in zip(codes,cities)]
    if name == 'transfer_to_human_agents': return 'Transfer successful'
    if name == 'calculate':
        e=a['expression']
        if not e or not all(c in '0123456789+-*/(). ' for c in e): raise BusinessError('Invalid characters in expression')
        try: return str(round(float(_arithmetic(e)),2))
        except Exception: raise BusinessError('Invalid expression')
    if name == 'send_certificate':
        u=_row(db,'users',a['user_id'])
        if a['amount']<=0: _reject('invalid_request','Certificate amount must be positive')
        p=_identifier(db,True)
        u['payment_methods'][p]=dict(id=p,source='certificate',amount=float(a['amount']))
        return f"Certificate {p} added to user {a['user_id']} with amount {a['amount']}."
    if name == 'book_reservation':
        u=_row(db,'users',a['user_id'])
        rid=_identifier(db)
        if not a['flights'] or len({(f['flight_number'],f['date']) for f in a['flights']})!=len(a['flights']): _reject('invalid_request','Duplicate or empty flight list')
        if not 1<=len(a['passengers'])<=5 or a['nonfree_baggages']<0 or a['total_baggages']<a['nonfree_baggages']: _reject('invalid_request','Invalid passenger or baggage quantities')
        fs=[_priced(db,f,a['cabin'],len(a['passengers'])) for f in a['flights']]
        total=sum(f['price'] for f in fs)*len(a['passengers'])+30*len(a['passengers'])*(a['insurance']=='yes')+50*a['nonfree_baggages']
        totals={}
        for p in a['payment_methods']:
            if p['amount']<0: _reject('invalid_request','Payment amount cannot be negative')
            totals[p['payment_id']]=totals.get(p['payment_id'],0)+p['amount']
        if len(totals)!=len(a['payment_methods']): _reject('invalid_request','Repeated payment method')
        for p in a['payment_methods']:
            pm=u['payment_methods'].get(p['payment_id'])
            if pm is None: _reject('payment_not_saved', f"Payment method {p['payment_id']} not found")
            if pm['source'] in ('gift_card','certificate') and pm['amount'] < p['amount']: _reject('insufficient_gift_balance', f"Not enough balance in payment method {p['payment_id']}")
        total_payment=sum(p['amount'] for p in a['payment_methods'])
        if total_payment != total: _reject('invalid_request', f'Payment amount does not add up, total price is {total}, but paid {total_payment}')
        for p in a['payment_methods']:
            pm=u['payment_methods'][p['payment_id']]
            if pm['source']=='gift_card': pm['amount']=float(pm['amount']-p['amount'])
            elif pm['source']=='certificate': del u['payment_methods'][p['payment_id']]
        _inventory(db,[],a['flights'],len(a['passengers']),a['cabin'],a['cabin'])
        r={k:deepcopy(a[k]) for k in ('user_id','origin','destination','flight_type','cabin','passengers','total_baggages','nonfree_baggages','insurance')}
        r.update(id=rid,reservation_id=rid,flights=fs,payment_history=deepcopy(a['payment_methods']),created_at=datetime.fromisoformat(now).isoformat(),status=None)
        db['reservations'].append(r);u['reservations'].append(rid)
        return _public(r)
    r=_row(db,'reservations',a['reservation_id'])
    if name=='cancel_reservation':
        if r['status']=='cancelled': return _public(r)
        u=_row(db,'users',r['user_id']);net={}
        for p in r['payment_history']: net[p['payment_id']]=net.get(p['payment_id'],0)+p['amount']
        debt=round(-sum(min(v,0) for v in net.values()),2)
        if round(sum(net.values()),2)<0: _reject('invalid_request','Refund exceeds payment')
        for pid,value in net.items():
            amount=round(max(0,value-debt),2)
            debt=round(max(0,debt-max(value,0)),2)
            if not amount: continue
            method=u['payment_methods'].get(pid)
            if method is None and pid.startswith('certificate_'):
                method=dict(id=pid,source='certificate',amount=0.0);u['payment_methods'][pid]=method
            if method is None: _reject('payment_not_saved','Payment method not found')
            if method['source'] in ('gift_card','certificate'): method['amount']=float(method['amount']+amount)
            r['payment_history'].append(dict(payment_id=pid,amount=-amount))
        _inventory(db,r['flights'],[],len(r['passengers']),r['cabin'],r['cabin'])
        r['status']='cancelled'
    elif name=='update_reservation_passengers':
        if r['status']=='cancelled': _reject('already_cancelled','Reservation already cancelled')
        if len(a['passengers']) != len(r['passengers']): _reject('passenger_count_change', 'Number of passengers does not match')
        r['passengers']=deepcopy(a['passengers'])
    else:
        if r['status']=='cancelled': _reject('already_cancelled','Reservation already cancelled')
        u=_row(db,'users',r['user_id'])
        if name=='update_reservation_baggages':
            amount=50*max(0,a['nonfree_baggages']-r['nonfree_baggages'])
        else:
            if not a['flights'] or len({(f['flight_number'],f['date']) for f in a['flights']})!=len(a['flights']): _reject('invalid_request','Duplicate or empty flight list')
            fs=[]
            for f in a['flights']:
                retained=next((old for old in r['flights'] if old['flight_number']==f['flight_number'] and old['date']==f['date'] and a['cabin']==r['cabin']),None)
                fs.append(deepcopy(retained) if retained else _priced(db,f,a['cabin'],len(r['passengers'])))
            amount=(sum(f['price'] for f in fs)-sum(f['price'] for f in r['flights']))*len(r['passengers'])
        p=_pay(u,a['payment_id'],amount)
        if p: r['payment_history'].append(p)
        if name=='update_reservation_baggages':
            r.update(total_baggages=a['total_baggages'],nonfree_baggages=a['nonfree_baggages'])
        else:
            _inventory(db,r['flights'],fs,len(r['passengers']),r['cabin'],a['cabin'])
            r.update(flights=fs,cabin=a['cabin'])
    return _public(r)

@write
def book_reservation(db, now, user_id: str, origin: str, destination: str, flight_type: str, cabin: str, flights: list[dict], passengers: list[dict], payment_methods: list[dict], total_baggages: int, nonfree_baggages: int, insurance: str):
    "Book a reservation.\n\nArgs:\n    user_id: The ID of the user to book the reservation such as 'sara_doe_496'`.\n    origin: The IATA code for the origin city such as 'SFO'.\n    destination: The IATA code for the destination city such as 'JFK'.\n    flight_type: The type of flight such as 'one_way' or 'round_trip'.\n    cabin: Allowed values: basic_economy, economy, business.\n    flights: Objects with flight_number (flight id such as AV999) and date (YYYY-MM-DD); complete ordered flight list.\n    passengers: Objects with first_name (str), last_name (str), dob (YYYY-MM-DD).\n    payment_methods: Objects with payment_id (saved method id such as credit_card_123456) and amount (integer US dollars).\n    total_baggages: The total number of baggage items to book the reservation.\n    nonfree_baggages: The number of non-free baggage items to book the reservation.\n    insurance: Allowed values: yes, no."
    return _dispatch('book_reservation', db, now, {'user_id': user_id, 'origin': origin, 'destination': destination, 'flight_type': flight_type, 'cabin': cabin, 'flights': flights, 'passengers': passengers, 'payment_methods': payment_methods, 'total_baggages': total_baggages, 'nonfree_baggages': nonfree_baggages, 'insurance': insurance})

def calculate(db, now, expression: str):
    "Calculate the result of a mathematical expression.\n\nArgs:\n    expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces."
    return _dispatch('calculate', db, now, {'expression': expression})

@write
def cancel_reservation(db, now, reservation_id: str):
    "Cancel the whole reservation.\n\nArgs:\n    reservation_id: The reservation ID, such as 'ZFA04Y'."
    return _dispatch('cancel_reservation', db, now, {'reservation_id': reservation_id})

def get_reservation_details(db, now, reservation_id: str):
    "Get the details of a reservation.\n\nArgs:\n    reservation_id: The reservation ID, such as '8JX2WO'."
    return _dispatch('get_reservation_details', db, now, {'reservation_id': reservation_id})

def get_user_details(db, now, user_id: str):
    "Get the details of a user, including their reservations.\n\nArgs:\n    user_id: The user ID, such as 'sara_doe_496'."
    return _dispatch('get_user_details', db, now, {'user_id': user_id})

def list_all_airports(db, now):
    'Returns a list of all available airports.\n\nArgs:'
    return _dispatch('list_all_airports', db, now, {})

def search_direct_flight(db, now, origin: str, destination: str, date: str):
    "Search for direct flights between two cities on a specific date.\n\nArgs:\n    origin: The origin city airport in three letters, such as 'JFK'.\n    destination: The destination city airport in three letters, such as 'LAX'.\n    date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'."
    return _dispatch('search_direct_flight', db, now, {'origin': origin, 'destination': destination, 'date': date})

def search_onestop_flight(db, now, origin: str, destination: str, date: str):
    "Search for one-stop flights between two cities on a specific date.\n\nArgs:\n    origin: The origin city airport in three letters, such as 'JFK'.\n    destination: The destination city airport in three letters, such as 'LAX'.\n    date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'."
    return _dispatch('search_onestop_flight', db, now, {'origin': origin, 'destination': destination, 'date': date})

@write
def send_certificate(db, now, user_id: str, amount: int):
    "Send a certificate to a user. Be careful!\n\nArgs:\n    user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.\n    amount: Integer US dollars."
    return _dispatch('send_certificate', db, now, {'user_id': user_id, 'amount': amount})

def transfer_to_human_agents(db, now, summary: str):
    "Transfer the user to a human agent, with a summary of the user's issue.\nOnly transfer if\n -  the user explicitly asks for a human agent\n -  given the policy and the available tools, you cannot solve the user's issue.\n\nArgs:\n    summary: A summary of the user's issue."
    return _dispatch('transfer_to_human_agents', db, now, {'summary': summary})

@write
def update_reservation_baggages(db, now, reservation_id: str, total_baggages: int, nonfree_baggages: int, payment_id: str):
    "Update the baggage information of a reservation.\n\nArgs:\n    reservation_id: The reservation ID, such as 'ZFA04Y'\n    total_baggages: The updated total number of baggage items included in the reservation.\n    nonfree_baggages: The updated number of non-free baggage items included in the reservation.\n    payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'."
    return _dispatch('update_reservation_baggages', db, now, {'reservation_id': reservation_id, 'total_baggages': total_baggages, 'nonfree_baggages': nonfree_baggages, 'payment_id': payment_id})

@write
def update_reservation_flights(db, now, reservation_id: str, cabin: str, flights: list[dict], payment_id: str):
    "Update the flight information of a reservation.\n\n\nArgs:\n    reservation_id: The reservation ID, such as 'ZFA04Y'.\n    cabin: Allowed values: basic_economy, economy, business.\n    flights: Objects with flight_number (flight id such as AV999) and date (YYYY-MM-DD); complete ordered flight list.\n    payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'."
    return _dispatch('update_reservation_flights', db, now, {'reservation_id': reservation_id, 'cabin': cabin, 'flights': flights, 'payment_id': payment_id})

@write
def update_reservation_passengers(db, now, reservation_id: str, passengers: list[dict]):
    "Update the passenger information of a reservation.\n\nArgs:\n    reservation_id: The reservation ID, such as 'ZFA04Y'.\n    passengers: Objects with first_name (str), last_name (str), dob (YYYY-MM-DD)."
    return _dispatch('update_reservation_passengers', db, now, {'reservation_id': reservation_id, 'passengers': passengers})

def get_flight_status(db, now, flight_number: str, date: str):
    'Get the status of a flight.\n\nArgs:\n    flight_number: Flight id such as AV999.\n    date: ISO date YYYY-MM-DD.'
    return _dispatch('get_flight_status', db, now, {'flight_number': flight_number, 'date': date})

def make_tools() -> dict:
    return {'book_reservation': book_reservation, 'calculate': calculate, 'cancel_reservation': cancel_reservation, 'get_reservation_details': get_reservation_details, 'get_user_details': get_user_details, 'list_all_airports': list_all_airports, 'search_direct_flight': search_direct_flight, 'search_onestop_flight': search_onestop_flight, 'send_certificate': send_certificate, 'transfer_to_human_agents': transfer_to_human_agents, 'update_reservation_baggages': update_reservation_baggages, 'update_reservation_flights': update_reservation_flights, 'update_reservation_passengers': update_reservation_passengers, 'get_flight_status': get_flight_status}


def approval_key(tool, args, facts) -> str:
    prefixes={'book_reservation':'book','update_reservation_baggages':'baggage','update_reservation_passengers':'passengers','cancel_reservation':'cancel','send_certificate':'compensation'}
    if tool=='update_reservation_flights':
        # Identity of the proposal, rather than call order, distinguishes a cabin change.
        goals=facts.get('goals',[])
        for g in goals:
            p=g['params']
            if p.get('reservation_id') != args['reservation_id']: continue
            if g['process']=='itinerary_change' and p.get('flights')==args['flights'] and p.get('cabin',args['cabin'])==args['cabin']:
                return 'approve_itinerary_'+args['reservation_id']
        return 'approve_cabin_'+args['reservation_id']
    if tool=='send_certificate':
        for g in facts.get('goals',[]):
            if g['process']=='requested_compensation' and g['params'].get('user_id')==args['user_id']:
                return 'approve_compensation_'+g['params']['reservation_id']
    return 'approve_'+prefixes[tool]+'_'+args.get('reservation_id',args.get('user_id',''))


def _allowance(membership, cabin, count):
    return ({'regular':0,'silver':1,'gold':2}[membership]+{'basic_economy':0,'economy':1,'business':2}[cabin])*count


def reference(api) -> dict:
    from env_factory.runtime import resolve_goal_params
    goals=api.ask('goals')
    results={}
    completed={}
    disruptions={}

    def call(name, **args):
        return api.call(name,**args)

    def perform(name, args, key):
        token=api.confirm(key,{'tool':name,'args':args})
        if token is None: raise BusinessError('confirmation_declined')
        return api.call(name,consent=token,**args)

    def statuses(r):
        return [call('get_flight_status',flight_number=f['flight_number'],date=f['date']) for f in r['flights']]

    def payment_check(u, pid, amount):
        pm=u['payment_methods'].get(pid)
        if pm is None: return 'payment_not_saved'
        if pm['source']=='certificate': return 'certificate_not_allowed'
        if pm['source']=='gift_card' and pm['amount']<amount: return 'insufficient_gift_balance'
        return None

    def cancellation_rule(r, s, p):
        if any(x in ('flying','landed') for x in s): return 'already_flown'
        if 0<=(datetime.fromisoformat(api.now)-datetime.fromisoformat(r['created_at'])).total_seconds()<=86400: return 'cancellation_24h'
        if 'cancelled' in s: return 'cancellation_airline'
        if r['cabin']=='business': return 'cancellation_business'
        if r['insurance']=='yes' and p.get('cancellation_reason') in ('health','weather'): return 'cancellation_insurance'
        return 'cancellation_ineligible'

    def comp_rule(u,r,s,p):
        if not p.get('requested'): return 'compensation_not_requested'
        if not (u['membership'] in ('silver','gold') or r['insurance']=='yes' or r['cabin']=='business'): return 'compensation_ineligible'
        complaint=p.get('complaint')
        if complaint not in ('cancelled','delayed'): return 'unsupported_reason'
        if complaint not in s and complaint not in disruptions.get(r['reservation_id'],[]): return 'unsupported_reason'
        if complaint=='cancelled': return 'compensation_cancelled'
        if r['reservation_id'] not in completed: return 'delay_action_required'
        return 'compensation_delay_after_action'

    def flight_proposal(r,p,cabin):
        fs=p.get('flights') or [dict(flight_number=f['flight_number'],date=f['date']) for f in r['flights']]
        new=[]
        for f in fs:
            old=next((x for x in r['flights'] if x['flight_number']==f['flight_number'] and x['date']==f['date'] and r['cabin']==cabin),None)
            if old: new.append(old); continue
            # Prefer observed segment endpoints, then discover other public routes.
            routes=[]
            for old in r['flights']:
                route=(old['origin'],old['destination'])
                if route not in routes: routes.append(route)
            found=None
            for origin,destination in routes:
                candidates=call('search_direct_flight',origin=origin,destination=destination,date=f['date'])
                found=next((x for x in candidates if x['flight_number']==f['flight_number']),None)
                if found: break
            if found is None:
                airports=call('list_all_airports')
                for airport in airports:
                    for airport2 in airports:
                        if airport['iata']==airport2['iata'] or (airport['iata'],airport2['iata']) in routes: continue
                        candidates=call('search_direct_flight',origin=airport['iata'],destination=airport2['iata'],date=f['date'])
                        found=next((x for x in candidates if x['flight_number']==f['flight_number']),None)
                        if found: break
                    if found: break
            if found is None: raise BusinessError('not_available')
            if found['available_seats'][cabin]<len(r['passengers']): raise BusinessError('insufficient_seats')
            new.append(dict(flight_number=f['flight_number'],date=f['date'],origin=found['origin'],destination=found['destination'],price=found['prices'][cabin]))
        return fs,new,(sum(f['price'] for f in new)-sum(f['price'] for f in r['flights']))*len(r['passengers'])

    for i,g in enumerate(goals,1):
        process=g['process']; p=resolve_goal_params(g,results)
        out={'status':'completed','reason':None}
        if process in ('booking_creation','itinerary_change','cabin_change','adding_baggage','eligible_cancellation','requested_compensation'): out['amount']=0
        if process in ('booking_creation','constrained_booking'): out.update(reservation_id=None,amount=0)
        if process=='requested_compensation': out['certificate_id']=None
        if process=='reservation_inspection': out['total_baggages']=None
        if process=='eligibility_explanation': out.update(eligible=False,policy_reason='invalid_request')
        try:
            if process=='record_query':
                out=extension_query(api,p)
            elif process=='flight_selection':
                out=extension_selection(api,p)
            elif process=='constrained_booking':
                out=extension_booking(api,p)
            elif process=='handover':
                out=extension_handover(api,p)
            elif process=='policy_inquiry':
                topic=p['topic']; count=p['passenger_count']
                if topic=='insurance_cost': q=30*count
                else:
                    q=_allowance(p['membership'],p['cabin'],count)
                    if topic=='baggage_fee': q=50*max(0,max(0,p['total_baggages']-q)-p.get('existing_nonfree_baggages',0))
                out.update(quantity=q,unit='bags' if topic=='baggage_allowance' else 'dollars')
            elif process=='reservation_inspection':
                rid=p.get('reservation_id')
                if not rid:
                    uid=p.get('user_id')
                    if not uid:
                        try: uid=api.ask('user_id')
                        except KeyError: raise BusinessError('identity_missing')
                    uidrec=call('get_user_details',user_id=uid)
                    rid=uidrec['reservations'][0] if uidrec['reservations'] else None
                r=call('get_reservation_details',reservation_id=rid)
                out['total_baggages']=r['total_baggages']
            else:
                uid=p.get('user_id')
                if not uid:
                    try: uid=api.ask('user_id')
                    except KeyError: raise BusinessError('identity_missing')
                if not uid: raise BusinessError('identity_missing')
                u=call('get_user_details',user_id=uid)
                if process=='booking_creation':
                    a=deepcopy(p['booking']);a['user_id']=uid
                    if not 1<=len(a['passengers'])<=5: raise BusinessError('invalid_request')
                    count=len(a['passengers'])
                    free=_allowance(u['membership'],a['cabin'],count)
                    if a['nonfree_baggages']!=max(0,a['total_baggages']-free): raise BusinessError('invalid_request')
                    sources=[]
                    for payment in a['payment_methods']:
                        pm=u['payment_methods'].get(payment['payment_id'])
                        if pm is None: raise BusinessError('payment_not_saved')
                        sources.append(pm['source'])
                    if any(sources.count(x)>n for x,n in [('credit_card',1),('certificate',1),('gift_card',3)]): raise BusinessError('invalid_request')
                    total=0
                    for f in a['flights']:
                        found=None
                        for airport in call('list_all_airports'):
                            for af in call('search_direct_flight',origin=airport['iata'],destination=a['destination'],date=f['date']):
                                if af['flight_number']==f['flight_number']: found=af
                            if found: break
                        if found is None:
                            # Supports connecting and return segments as well.
                            dummy=dict(flights=[],cabin=a['cabin'],passengers=a['passengers'])
                            _,priced,_=flight_proposal(dummy,{'flights':[f]},a['cabin'])
                            total+=priced[0]['price']*count
                        else:
                            if found['available_seats'][a['cabin']]<count: raise BusinessError('insufficient_seats')
                            total+=found['prices'][a['cabin']]*count
                    total+=50*a['nonfree_baggages']+30*count*(a['insurance']=='yes')
                    r=perform('book_reservation',a,'approve_book_'+uid)
                    out.update(reservation_id=r['reservation_id'],amount=total)
                else:
                    rid=p.get('reservation_id')
                    if not rid:
                        try: rid=api.ask('reservation_id')
                        except KeyError: rid=None
                        if not rid:
                            ids=u['reservations']
                            if not ids: raise BusinessError('reservation_not_found')
                            # The customer may provide route/date hints when the number is forgotten.
                            candidates=[call('get_reservation_details',reservation_id=x) for x in ids]
                            candidates=[r for r in candidates if all(k not in p or p[k]==r[k] for k in ('origin','destination','cabin','flight_type'))]
                            if not candidates: raise BusinessError('reservation_not_found')
                            rid=candidates[0]['reservation_id']
                    r=call('get_reservation_details',reservation_id=rid)
                    explain=process=='eligibility_explanation'
                    action=p.get('action') if explain else process
                    ap={**p,**p.get('details',{})} if explain else p
                    s=statuses(r)
                    disruptions.setdefault(rid,s)
                    reason=None; rule='allowed'; amount=0; name=None; a={}; prefix=None
                    if r['status']=='cancelled' and action!='requested_compensation': reason='already_cancelled'
                    elif action=='adding_baggage':
                        if ap['total_baggages']<r['total_baggages']: reason='baggage_removal'
                        else:
                            nf=max(0,ap['total_baggages']-_allowance(u['membership'],r['cabin'],len(r['passengers'])))
                            amount=50*max(0,nf-r['nonfree_baggages'])
                            a=dict(reservation_id=rid,total_baggages=ap['total_baggages'],nonfree_baggages=nf,payment_id=ap['payment_id'])
                            reason=payment_check(u,ap['payment_id'],amount)
                            name='update_reservation_baggages';prefix='baggage'
                    elif action=='passenger_list_update':
                        if len(ap['passengers'])!=len(r['passengers']): reason='passenger_count_change'
                        else: name='update_reservation_passengers';prefix='passengers';a=dict(reservation_id=rid,passengers=ap['passengers'])
                    elif action in ('itinerary_change','cabin_change'):
                        cabin=ap.get('cabin',r['cabin'])
                        if action=='itinerary_change' and r['cabin']=='basic_economy': reason='basic_economy_itinerary'
                        elif action=='cabin_change' and any(x in ('flying','landed') for x in s): reason='already_flown'
                        else:
                            fs,new,amount=flight_proposal(r,ap,cabin)
                            if action=='itinerary_change':
                                if cabin!=r['cabin'] or ap.get('origin',r['origin'])!=r['origin'] or ap.get('destination',r['destination'])!=r['destination'] or ap.get('flight_type',r['flight_type'])!=r['flight_type']: reason='route_or_trip_change'
                                elif not new or new[0]['origin']!=r['origin'] or (r['flight_type']=='one_way' and new[-1]['destination']!=r['destination']) or (r['flight_type']=='round_trip' and (new[-1]['destination']!=r['origin'] or not any(f['destination']==r['destination'] for f in new))): reason='route_or_trip_change'
                            reason=reason or payment_check(u,ap['payment_id'],amount)
                            name='update_reservation_flights';prefix='cabin' if action=='cabin_change' else 'itinerary'
                            a=dict(reservation_id=rid,cabin=cabin,flights=fs,payment_id=ap['payment_id'])
                    elif action=='eligible_cancellation':
                        rule=cancellation_rule(r,s,ap)
                        if not rule.startswith('cancellation_') or rule=='cancellation_ineligible': reason=rule
                        else:
                            amount=-sum(x['amount'] for x in r['payment_history'])
                            name='cancel_reservation';prefix='cancel';a=dict(reservation_id=rid)
                    elif action=='requested_compensation':
                        rule=comp_rule(u,r,s,ap)
                        if rule not in ('compensation_cancelled','compensation_delay_after_action'): reason=rule
                        else:
                            amount=(100 if ap['complaint']=='cancelled' else 50)*len(r['passengers'])
                            name='send_certificate';prefix='compensation';a=dict(user_id=uid,amount=amount)
                    else: reason='unsupported_request'
                    if explain:
                        out.update(eligible=reason is None,policy_reason=reason or rule)
                    elif reason:
                        if reason=='already_flown' and action=='eligible_cancellation':
                            call('transfer_to_human_agents',summary='YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.')
                            out.update(status='transferred',reason=reason)
                        else: raise BusinessError(reason)
                    else:
                        result=perform(name,a,'approve_'+prefix+'_'+rid)
                        if 'amount' in out: out['amount']=amount
                        if action=='requested_compensation': out['certificate_id']=result.split()[1]
                        if action in ('itinerary_change','cabin_change','eligible_cancellation'): completed[rid]=action
        except BusinessError as e:
            reason=getattr(e, 'reason', str(e))
            known={'identity_missing','user_not_found','reservation_not_found','confirmation_declined','payment_not_saved','insufficient_gift_balance','certificate_not_allowed','not_available','insufficient_seats','basic_economy_itinerary','route_or_trip_change','already_flown','baggage_removal','passenger_count_change','cancellation_ineligible','compensation_ineligible','compensation_not_requested','delay_action_required','unsupported_reason','unsupported_request','invalid_request','already_cancelled'}
            if reason not in known: reason='invalid_request'
            status='failed' if getattr(e, 'is_public_tool_error', False) or reason in ('user_not_found','reservation_not_found','payment_not_saved','insufficient_gift_balance','certificate_not_allowed','not_available','insufficient_seats') else 'refused'
            out.update(status=status,reason=reason)
            if process=='eligibility_explanation': out.update(eligible=False,policy_reason=reason)
        if out['status']=='transferred' and 'amount' in out: out['amount']=None
        results[f'goal_{i}']=out
    return results

"""Caller-owned additions to an airline world; executed only through public tools."""


def extension_query(api, p):
    out = dict(status='failed', reason='record_or_field_unavailable', value=None, number=None, boolean=None)
    try:
        kind = p['record_type']
        if kind == 'user': value = api.call('get_user_details', user_id=p['record_id'])
        elif kind == 'reservation': value = api.call('get_reservation_details', reservation_id=p['record_id'])
        elif kind == 'flight_status': value = api.call('get_flight_status', flight_number=p['record_id'], date=p['date'])
        elif kind == 'airports': value = api.call('list_all_airports')
        else: return out
        for key in p.get('selector', '').split('.') if p.get('selector') else []:
            value = value[int(key)] if isinstance(value, list) else value[key]
        if value is None: pass
        elif isinstance(value, bool): out['boolean'] = value
        elif isinstance(value, (int, float)): out['number'] = value
        elif isinstance(value, str): out['value'] = value
        else: return out
        out['status'] = 'completed'
        out['reason'] = None
    except (BusinessError, KeyError, IndexError, ValueError, TypeError):
        pass
    return out


def extension_selection(api, p):
    if set(p.get('fallback') or {})-{'max_fare','max_stops','depart_after','arrive_before','dates'}:
        return dict(status='failed',reason='invalid_request',flights=[],dates=[],fare=None)
    def clock(value):
        parts = value.replace('+1', '').strip().split(':')
        return int(parts[0])*60 + int(parts[1]) + (1440 if '+1' in value else 0)
    def choose(c):
        options = []
        for date in c['dates']:
            direct = api.call('search_direct_flight', origin=c['origin'], destination=c['destination'], date=date)
            routes = [[dict(f, date=date)] for f in direct]
            if c.get('max_stops', 0) == 1:
                routes += api.call('search_onestop_flight', origin=c['origin'], destination=c['destination'], date=date)
            for legs in routes:
                if not legs or any(f['available_seats'][c['cabin']] < c['passenger_count'] for f in legs): continue
                fare = sum(f['prices'][c['cabin']] for f in legs)*c['passenger_count']
                if fare > c.get('max_fare', 1000000000): continue
                if clock(legs[0]['scheduled_departure_time_est']) < clock(c.get('depart_after','00:00')): continue
                day_delta = (datetime.fromisoformat(legs[-1]['date'])-datetime.fromisoformat(date)).days
                arrival = day_delta*1440 + clock(legs[-1]['scheduled_arrival_time_est'])
                if arrival > clock(c.get('arrive_before','23:59+1')): continue
                # The task states this total ordering; equal primary scores do not imply arbitrary hidden choices.
                pref = c.get('preference', 'cheapest')
                rank = ((len(legs)-1, fare) if pref=='direct_first' else
                        (date, arrival, fare) if pref=='earliest_arrival' else (fare, len(legs)-1))
                tie = (date, clock(legs[0]['scheduled_departure_time_est']), tuple(f['flight_number'] for f in legs))
                options.append((rank+tie, legs, fare))
        return min(options, key=lambda x: x[0]) if options else None
    selected = choose(p)
    if selected is None and p.get('fallback'):
        selected = choose({**p, **p['fallback'], 'fallback': None})
    if selected is None: return dict(status='no_match', reason='not_available', flights=[], dates=[], fare=None)
    _, legs, fare = selected
    return dict(status='completed', reason=None, flights=[f['flight_number'] for f in legs],
                dates=[f['date'] for f in legs], fare=fare)


def extension_booking(api, p):
    choice = extension_selection(api, p['selection'])
    if choice['status'] != 'completed':
        return dict(status='refused', reason=choice['reason'], reservation_id=None, amount=0)
    if p.get('return_selection'):
        outbound=p['selection']; inbound=p['return_selection']
        if (inbound['origin']!=outbound['destination'] or inbound['destination']!=outbound['origin'] or
            inbound['cabin']!=outbound['cabin'] or inbound['passenger_count']!=outbound['passenger_count'] or
            min(inbound['dates'])<=max(choice['dates'])):
            return dict(status='refused',reason='invalid_request',reservation_id=None,amount=0)
        returning=extension_selection(api,inbound)
        if returning['status']!='completed':
            return dict(status='refused',reason=returning['reason'],reservation_id=None,amount=0)
        choice=dict(choice,flights=choice['flights']+returning['flights'],dates=choice['dates']+returning['dates'],
                    fare=choice['fare']+returning['fare'])
    uid = p['user_id']
    user = api.call('get_user_details', user_id=uid)
    passengers = p['passengers']; n = len(passengers); cabin = p['selection']['cabin']
    if n != p['selection']['passenger_count'] or not 1 <= n <= 5:
        return dict(status='refused', reason='invalid_request', reservation_id=None, amount=0)
    nonfree = max(0, p['total_baggages']-_allowance(user['membership'], cabin, n))
    total = choice['fare'] + 50*nonfree + 30*n*(p['insurance']=='yes')
    if p['payment_id'] not in user['payment_methods']: raise BusinessError('payment_not_saved')
    proposal = dict(user_id=uid, origin=p['selection']['origin'], destination=p['selection']['destination'],
                    flight_type='round_trip' if p.get('return_selection') else 'one_way', cabin=cabin,
                    flights=[dict(flight_number=f,date=d) for f,d in zip(choice['flights'],choice['dates'])],
                    passengers=passengers, payment_methods=[dict(payment_id=p['payment_id'],amount=total)],
                    total_baggages=p['total_baggages'], nonfree_baggages=nonfree, insurance=p['insurance'])
    token = api.confirm('approve_book_'+uid, {'tool':'book_reservation','args':proposal})
    if token is None: return dict(status='refused',reason='confirmation_declined',reservation_id=None,amount=0)
    result = api.call('book_reservation',consent=token,**proposal)
    return dict(status='completed',reason=None,reservation_id=result['reservation_id'],amount=total)


def extension_handover(api, p):
    if p['request_kind'] in ('change_passenger_count','add_insurance_after_booking'):
        return dict(status='refused',reason=p['request_kind'])
    if p['request_kind'] != 'unsupported_service': raise BusinessError('invalid_request')
    api.call('transfer_to_human_agents',summary=p['summary'])
    return dict(status='transferred',reason=None)
