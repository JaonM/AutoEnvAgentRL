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
