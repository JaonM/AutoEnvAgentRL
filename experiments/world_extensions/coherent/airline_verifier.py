from copy import deepcopy
from datetime import datetime, timedelta


class Rejection(Exception):
    def __init__(self, reason, status='refused'):
        self.reason, self.status = reason, status


def expected(data, facts, now):
    state = deepcopy(data)
    users = {r['user_id']: r for r in state['users']}
    reservations = {r['reservation_id']: r for r in state['reservations']}
    flights = {r['flight_number']: r for r in state['flights']}
    changes = {}
    original_status = {}
    def statuses(r):
        return [flights.get(f['flight_number'], {}).get('dates', {}).get(f['date'], {}).get('status') for f in r['flights']]
    for rid, r in reservations.items():
        original_status[rid] = statuses(r)
    def reject(reason, status='refused'):
        raise Rejection(reason, status)
    def allowance(membership, cabin, n):
        return n * ({'regular': 0, 'silver': 1, 'gold': 2}[membership] + {'basic_economy': 0, 'economy': 1, 'business': 2}[cabin])
    def payment(user, pid, amount, mutate=False):
        pm = user['payment_methods'].get(pid)
        if pm is None: reject('payment_not_saved', 'failed')
        if pm['source'] == 'certificate': reject('certificate_not_allowed', 'failed')
        if pm['source'] == 'gift_card':
            if pm['amount'] < amount: reject('insufficient_gift_balance', 'failed')
            if mutate: pm['amount'] = float(pm['amount']) - amount
    def priced(infos, cabin, n, old=None):
        result = []
        for info in infos:
            match = next((f for f in old['flights'] if f['flight_number'] == info['flight_number'] and f['date'] == info['date'] and cabin == old['cabin']), None) if old else None
            if match:
                result.append(deepcopy(match)); continue
            f = flights.get(info['flight_number'])
            d = f.get('dates', {}).get(info['date']) if f else None
            if not d or d['status'] != 'available': reject('not_available', 'failed')
            if d['available_seats'][cabin] < n: reject('insufficient_seats', 'failed')
            result.append(dict(flight_number=info['flight_number'], date=info['date'], origin=f['origin'], destination=f['destination'], price=d['prices'][cabin]))
        return result
    prefixes = {'booking_creation': 'book', 'itinerary_change': 'itinerary', 'cabin_change': 'cabin', 'adding_baggage': 'baggage', 'passenger_list_update': 'passengers', 'eligible_cancellation': 'cancel', 'requested_compensation': 'compensation'}
    def consent(proc, uid, rid):
        if facts.get('approve_' + prefixes[proc] + '_' + (uid if proc == 'booking_creation' else rid)) is not True:
            reject('confirmation_declined')
    def next_identifier(certificate=False):
        occupied=set(reservations)
        prefix='NR'
        if certificate:
            prefix='certificate_coherent_'
            occupied={k for u in users.values() for k in u['payment_methods']}
            occupied.update(p['payment_id'] for r in reservations.values() for p in r['payment_history'])
        for number in range(1,len(occupied)+2):
            candidate=prefix+str(number).zfill(8)
            if candidate not in occupied:return candidate
        reject('invalid_request','failed')
    def adjust_inventory(old,new,n,old_cabin,new_cabin):
        totals={}
        for legs,cabin,sign in ((old,old_cabin,1),(new,new_cabin,-1)):
            for f in legs:
                key=(f['flight_number'],f['date'],cabin)
                totals[key]=totals.get(key,0)+sign*n
        for (number,date,cabin),quantity in totals.items():
            inst=flights[number]['dates'][date]
            if 'available_seats' in inst:
                if inst['available_seats'][cabin]+quantity<0: reject('insufficient_seats','failed')
                inst['available_seats'][cabin]+=quantity
    def execute(proc, p, dry=False):
        out = dict(status='completed', reason=None)
        if proc in ('booking_creation','itinerary_change','cabin_change','adding_baggage','eligible_cancellation','requested_compensation'): out['amount'] = 0
        if proc == 'booking_creation': out['reservation_id'] = None
        if proc == 'requested_compensation': out['certificate_id'] = None
        if proc == 'reservation_inspection': out['total_baggages'] = None
        if proc == 'policy_inquiry': out.update(quantity=0, unit='bags' if p.get('topic') == 'baggage_allowance' else 'dollars')
        policy = 'allowed'
        try:
            if proc == 'record_query': return extension_expected_query(state,p), 'allowed'
            if proc == 'flight_selection': return extension_expected_selection(state,p), 'allowed'
            if proc == 'handover':
                return dict(status='transferred' if p['request_kind']=='unsupported_service' else 'refused',reason=None if p['request_kind']=='unsupported_service' else p['request_kind']), 'allowed'
            if proc == 'constrained_booking':
                chosen=extension_expected_selection(state,p['selection'])
                if chosen['status']!='completed': return dict(status='refused',reason=chosen['reason'],reservation_id=None,amount=0), chosen['reason']
                if p.get('return_selection'):
                    back=p['return_selection']; front=p['selection']
                    valid=(back['origin']==front['destination'] and back['destination']==front['origin'] and back['cabin']==front['cabin'] and back['passenger_count']==front['passenger_count'] and all(d>max(chosen['dates']) for d in back['dates']))
                    if not valid:return dict(status='refused',reason='invalid_request',reservation_id=None,amount=0), 'invalid_request'
                    returning=extension_expected_selection(state,back)
                    if returning['status']!='completed':return dict(status='refused',reason=returning['reason'],reservation_id=None,amount=0),returning['reason']
                    chosen=dict(chosen,flights=chosen['flights']+returning['flights'],dates=chosen['dates']+returning['dates'],fare=chosen['fare']+returning['fare'])
                u=users[p['user_id']]; n=len(p['passengers']); cabin=p['selection']['cabin']
                if n!=p['selection']['passenger_count'] or not 1<=n<=5: return dict(status='refused',reason='invalid_request',reservation_id=None,amount=0), 'invalid_request'
                paid=max(0,p['total_baggages']-allowance(u['membership'],cabin,n))
                total=chosen['fare']+50*paid+30*n*(p['insurance']=='yes')
                booking=dict(user_id=p['user_id'],origin=p['selection']['origin'],destination=p['selection']['destination'],
                    flight_type='round_trip' if p.get('return_selection') else 'one_way',cabin=cabin,passengers=deepcopy(p['passengers']),
                    flights=[dict(flight_number=f,date=d) for f,d in zip(chosen['flights'],chosen['dates'])],
                    payment_methods=[dict(payment_id=p['payment_id'],amount=total)],total_baggages=p['total_baggages'],nonfree_baggages=paid,insurance=p['insurance'])
                return execute('booking_creation',dict(user_id=p['user_id'],booking=booking),dry)
            if proc == 'policy_inquiry':
                topic = p.get('topic'); n = p.get('passenger_count', 0)
                if topic == 'insurance_cost': out['quantity'] = 30*n
                elif topic in ('baggage_fee','baggage_allowance'):
                    free = allowance(p['membership'], p['cabin'], n)
                    out['quantity'] = free if topic == 'baggage_allowance' else 50*max(0, max(0,p['total_baggages']-free)-p.get('existing_nonfree_baggages',0))
                else: reject('unsupported_request')
                return out, policy
            uid, rid = p.get('user_id'), p.get('reservation_id')
            if proc == 'reservation_inspection':
                if rid not in reservations: reject('reservation_not_found', 'failed')
                out['total_baggages'] = reservations[rid]['total_baggages']
                return out, policy
            if not uid: reject('identity_missing')
            if uid not in users: reject('user_not_found','failed')
            user = users[uid]
            if proc == 'booking_creation':
                b = deepcopy(p['booking']); n = len(b['passengers']); cabin = b['cabin']
                if not 1 <= n <= 5: reject('invalid_request')
                if not b['flights'] or len({(f['flight_number'],f['date']) for f in b['flights']})!=len(b['flights']): reject('invalid_request','failed')
                free = allowance(user['membership'], cabin, n)
                if b['nonfree_baggages'] != max(0,b['total_baggages']-free): reject('invalid_request')
                fs = priced(b['flights'], cabin, n)
                amount = n*sum(f['price'] for f in fs) + 30*n*(b['insurance']=='yes') + 50*b['nonfree_baggages']
                counts = {}
                if len({item['payment_id'] for item in b['payment_methods']})!=len(b['payment_methods']) or any(item['amount']<0 for item in b['payment_methods']):reject('invalid_request','failed')
                for item in b['payment_methods']:
                    pm = user['payment_methods'].get(item['payment_id'])
                    if pm is None: reject('payment_not_saved','failed')
                    source = pm['source']; counts[source] = counts.get(source,0)+1
                    if counts[source] > (3 if source=='gift_card' else 1): reject('invalid_request')
                    if source != 'credit_card' and pm['amount'] < item['amount']: reject('insufficient_gift_balance','failed')
                if sum(x['amount'] for x in b['payment_methods']) != amount: reject('invalid_request','failed')
                new_id = next_identifier()
                if not dry:
                    consent(proc, uid, rid)
                    for item in b['payment_methods']:
                        pm = user['payment_methods'][item['payment_id']]
                        if pm['source']=='certificate': del user['payment_methods'][item['payment_id']]
                        elif pm['source']=='gift_card': pm['amount'] = float(pm['amount'])-item['amount']
                    adjust_inventory([],fs,n,cabin,cabin)
                    r = {k:deepcopy(b[k]) for k in ('origin','destination','flight_type','cabin','passengers','total_baggages','nonfree_baggages','insurance')}
                    r.update(id=new_id,reservation_id=new_id,user_id=uid,flights=fs,payment_history=deepcopy(b['payment_methods']),created_at=datetime.fromisoformat(now).isoformat(),status=None)
                    state['reservations'].append(r); reservations[new_id]=r; user['reservations'].append(new_id)
                out.update(amount=amount,reservation_id=new_id)
                return out, policy
            if not rid:
                candidates = user['reservations']
                if len(candidates)==1: rid=candidates[0]
            if rid not in reservations: reject('reservation_not_found','failed')
            r=reservations[rid]; owner=users[r['user_id']]; n=len(r['passengers']); ss=statuses(r)
            if r.get('status')=='cancelled' and proc!='requested_compensation': reject('already_cancelled')
            amount=0
            if proc == 'adding_baggage':
                total=p['total_baggages']
                if total < r['total_baggages']: reject('baggage_removal')
                nonfree=max(0,total-allowance(owner['membership'],r['cabin'],n))
                amount=50*max(0,nonfree-r['nonfree_baggages'])
                payment(owner,p.get('payment_id'),amount)
            elif proc in ('itinerary_change','cabin_change'):
                cabin=p.get('cabin',r['cabin'])
                if proc=='cabin_change':
                    if any(s in ('flying','landed') for s in ss): reject('already_flown')
                    infos=r['flights']
                else:
                    if r['cabin']=='basic_economy': reject('basic_economy_itinerary')
                    infos=p['flights']
                    if p.get('origin',r['origin'])!=r['origin'] or p.get('destination',r['destination'])!=r['destination'] or p.get('flight_type',r['flight_type'])!=r['flight_type']: reject('route_or_trip_change')
                    endpoints=[flights.get(f['flight_number']) for f in infos]
                    if all(endpoints):
                        if endpoints[0]['origin']!=r['origin'] or endpoints[-1]['destination']!=(r['origin'] if r['flight_type']=='round_trip' else r['destination']): reject('route_or_trip_change')
                        if r['flight_type']=='round_trip' and not any(f['destination']==r['destination'] for f in endpoints): reject('route_or_trip_change')
                fs=priced(infos,cabin,n,r)
                if not infos or len({(f['flight_number'],f['date']) for f in infos})!=len(infos):reject('invalid_request','failed')
                amount=n*(sum(f['price'] for f in fs)-sum(f['price'] for f in r['flights']))
                payment(owner,p.get('payment_id'),amount)
            elif proc=='passenger_list_update':
                if len(p['passengers'])!=n: reject('passenger_count_change')
            elif proc=='eligible_cancellation':
                if any(s in ('flying','landed') for s in ss): reject('already_flown','transferred')
                if not p.get('cancellation_reason'): reject('invalid_request')
                elapsed=(datetime.fromisoformat(now)-datetime.fromisoformat(r['created_at'])).total_seconds()
                if 0<=elapsed<=86400: policy='cancellation_24h'
                elif 'cancelled' in ss: policy='cancellation_airline'
                elif r['cabin']=='business': policy='cancellation_business'
                elif r['insurance']=='yes' and p['cancellation_reason'] in ('health','weather'): policy='cancellation_insurance'
                else: reject('cancellation_ineligible')
                amount=-sum(x['amount'] for x in r['payment_history'])
            elif proc=='requested_compensation':
                if not p.get('requested'): reject('compensation_not_requested')
                if not (user['membership'] in ('silver','gold') or r['insurance']=='yes' or r['cabin']=='business'): reject('compensation_ineligible')
                complaint=p.get('complaint')
                if complaint not in ('cancelled','delayed'): reject('unsupported_reason')
                observed=original_status.get(rid,ss) if rid in changes else ss
                if complaint not in observed: reject('unsupported_reason')
                if complaint=='delayed' and rid not in changes: reject('delay_action_required')
                policy='compensation_cancelled' if complaint=='cancelled' else 'compensation_delay_after_action'
                amount=(100 if complaint=='cancelled' else 50)*n
                cid=next_identifier(True)
            else: reject('unsupported_request','transferred')
            if not dry:
                consent(proc,uid,rid)
                if proc in ('adding_baggage','itinerary_change','cabin_change'):
                    payment(owner,p.get('payment_id'),amount,True)
                    if amount: r['payment_history'].append(dict(payment_id=p['payment_id'],amount=amount))
                if proc=='adding_baggage': r.update(total_baggages=total,nonfree_baggages=nonfree)
                elif proc in ('itinerary_change','cabin_change'):
                    adjust_inventory(r['flights'],fs,n,r['cabin'],cabin)
                    r.update(flights=fs,cabin=cabin); changes[rid]='changed'
                elif proc=='passenger_list_update': r['passengers']=deepcopy(p['passengers'])
                elif proc=='eligible_cancellation':
                    history=list(r['payment_history'])
                    balances={pid:round(sum(x['amount'] for x in history if x['payment_id']==pid),2) for pid in dict.fromkeys(x['payment_id'] for x in history)}
                    if round(sum(balances.values()),2)<0:reject('invalid_request','failed')
                    offset=-sum(v for v in balances.values() if v<0)
                    for pid in balances:
                        outstanding=sum(x['amount'] for x in history if x['payment_id']==pid)
                        available=max(0,outstanding); consumed=min(available,offset)
                        outstanding=round(available-consumed,2); offset=round(offset-consumed,2)
                        if not outstanding:continue
                        method=owner['payment_methods'].get(pid)
                        if method is None and pid.startswith('certificate_'):
                            method=dict(id=pid,source='certificate',amount=0.0);owner['payment_methods'][pid]=method
                        if method is None:reject('payment_not_saved','failed')
                        if method['source'] in ('gift_card','certificate'):method['amount']=float(method['amount']+outstanding)
                        r['payment_history'].append(dict(payment_id=pid,amount=-outstanding))
                    adjust_inventory(r['flights'],[],n,r['cabin'],r['cabin'])
                    r['status']='cancelled';changes[rid]='cancelled'
                elif proc=='requested_compensation': user['payment_methods'][cid]=dict(id=cid,source='certificate',amount=float(amount))
            if 'amount' in out: out['amount']=amount
            if proc=='requested_compensation': out['certificate_id']=cid
        except Rejection as e:
            out.update(status=e.status,reason=e.reason); policy=e.reason
        if out['status']=='transferred' and 'amount' in out: out['amount']=None
        return out,policy
    answer={}
    for i,g in enumerate(facts.get('goals',[]),1):
        proc=g['process']; p=g.get('params',{})
        if proc=='eligibility_explanation':
            detail=dict(p.get('details') or {})
            detail.update(user_id=p.get('user_id'),reservation_id=p.get('reservation_id'))
            result,rule=execute(p['action'],detail,True)
            result=dict(status='completed' if result['reason'] is None or result['reason'] not in ('identity_missing','user_not_found','reservation_not_found') else result['status'],reason=result['reason'] if result['reason'] in ('identity_missing','user_not_found','reservation_not_found') else None,eligible=result['status']=='completed',policy_reason=rule)
        else: result,_=execute(proc,p)
        answer['goal_'+str(i)]=result
    return dict(answer=answer,final_state=state)

"""Independent expected outcomes computed from state, never from reference/tool code."""


def extension_expected_query(state, p):
    answer = {'status':'failed','reason':'record_or_field_unavailable','value':None,'number':None,'boolean':None}
    try:
        record_type = p['record_type']
        if record_type in ('user','reservation'):
            table, key = ('users','user_id') if record_type=='user' else ('reservations','reservation_id')
            result = next(r for r in state[table] if r[key]==p['record_id'])
            result = deepcopy(result); result.pop('id',None)
        elif record_type=='flight_status':
            result = next(r for r in state['flights'] if r['flight_number']==p['record_id'])['dates'][p['date']]['status']
        elif record_type=='airports':
            codes = ['SFO','JFK','LAX','ORD','DFW','DEN','SEA','ATL','MIA','BOS','PHX','IAH','LAS','MCO','EWR','CLT','MSP','DTW','PHL','LGA']
            names = ['San Francisco','New York','Los Angeles','Chicago','Dallas','Denver','Seattle','Atlanta','Miami','Boston','Phoenix','Houston','Las Vegas','Orlando','Newark','Charlotte','Minneapolis','Detroit','Philadelphia','LaGuardia']
            result = [dict(iata=c,city=n) for c,n in zip(codes,names)]
        else: return answer
        for component in p.get('selector','').split('.') if p.get('selector') else []:
            if isinstance(result,list): result=result[int(component)]
            else: result=result[component]
        value_type = type(result)
        if value_type is str: answer['value']=result
        elif value_type is bool: answer['boolean']=result
        elif value_type in (int,float): answer['number']=result
        elif result is not None: return answer
        answer['status']='completed'
        answer['reason']=None
    except (KeyError,IndexError,ValueError,TypeError,StopIteration): pass
    return answer


def extension_expected_selection(state, p):
    if any(key not in ('max_fare','max_stops','depart_after','arrive_before','dates') for key in p.get('fallback',{})):
        return {'status':'failed','reason':'invalid_request','flights':[],'dates':[],'fare':None}
    def minutes(time):
        h,m = time.split(':')[:2]
        return int(h)*60+int(m.split('+')[0])+(1440 if '+1' in time else 0)
    def collect(constraints):
        ranked=[]
        for date in constraints['dates']:
            available=[r for r in state['flights'] if r['dates'].get(date,{}).get('status')=='available']
            candidates=[[(r,date)] for r in available if r['origin']==constraints['origin'] and r['destination']==constraints['destination']]
            if constraints.get('max_stops',0)==1:
                for first in available:
                    if first['origin']!=constraints['origin']: continue
                    nextdate=(datetime.fromisoformat(date)+timedelta(days=1)).date().isoformat() if '+1' in first['scheduled_arrival_time_est'] else date
                    for second in state['flights']:
                        # Preserve the source API's calendar and lexicographic connection filter.
                        if second['origin']==first['destination'] and second['destination']==constraints['destination'] and second['dates'].get(nextdate,{}).get('status')=='available' and second['scheduled_departure_time_est']>=first['scheduled_arrival_time_est'].split('+')[0]:
                            candidates.append([(first,date),(second,nextdate)])
            for itinerary in candidates:
                cabin=constraints['cabin']; count=constraints['passenger_count']
                if any(r['dates'][d]['available_seats'][cabin]<count for r,d in itinerary): continue
                cost=sum(r['dates'][d]['prices'][cabin] for r,d in itinerary)*count
                departure=minutes(itinerary[0][0]['scheduled_departure_time_est'])
                arrival=minutes(itinerary[-1][0]['scheduled_arrival_time_est'])+(datetime.fromisoformat(itinerary[-1][1])-datetime.fromisoformat(date)).days*1440
                if cost>constraints.get('max_fare',1000000000) or departure<minutes(constraints.get('depart_after','00:00')) or arrival>minutes(constraints.get('arrive_before','23:59+1')): continue
                mode=constraints.get('preference','cheapest')
                score=(len(itinerary)-1,cost) if mode=='direct_first' else (date,arrival,cost) if mode=='earliest_arrival' else (cost,len(itinerary)-1)
                ordered=score+(date,departure,tuple(r['flight_number'] for r,d in itinerary))
                ranked.append((ordered,itinerary,cost))
        return sorted(ranked,key=lambda v:v[0])
    matches=collect(p)
    if not matches and p.get('fallback'): matches=collect(dict(p,**p['fallback']))
    if not matches: return {'status':'no_match','reason':'not_available','flights':[],'dates':[],'fare':None}
    item=matches[0]
    return {'status':'completed','reason':None,'flights':[r['flight_number'] for r,d in item[1]],'dates':[d for r,d in item[1]],'fare':item[2]}
