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
                    nextdate='2024-05-'+str(int(date[-2:])+1) if '+1' in first['scheduled_arrival_time_est'] else date
                    for second in state['flights']:
                        # Preserve the source API's calendar and lexicographic connection filter.
                        if second['origin']==first['destination'] and second['destination']==constraints['destination'] and second['dates'].get(nextdate,{}).get('status')=='available' and second['scheduled_departure_time_est']>=first['scheduled_arrival_time_est']:
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
