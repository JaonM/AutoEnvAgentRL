"""Build isolated next-version worlds and executable gap probes, leaving the active run frozen."""
import argparse
import copy
import json
from pathlib import Path

from env_factory.certify import bundle
from env_factory.core.sandbox import load_module
from env_factory.production.world import task_schema, check_world
from env_factory.core.answers import answer_problems
from env_factory.core.reward import grade
from env_factory.runtime import run_reference, canonical
from env_factory.coverage import report as coverage_report
from recheck_world_tasks import save

ROOT=Path(__file__).resolve().parent


def schemas(original):
    base=copy.deepcopy(original)
    def nullable(kind):return {'type':kind,'nullable':True}
    def result(props):return {'type':'object','properties':props,'description':'Report the requested observed result in natural language.'}
    status=lambda vals:{'type':'string','enum':vals}
    base['record_query']={'description':'Read one explicitly requested scalar in a public user, reservation, flight status or airport record. Multiple goals can ask for multiple facts.',
        'kind':'account','params':{'record_type':'enum user,reservation,flight_status,airports','record_id':'string: ID supplied by the customer; empty for airports',
        'date':'string: flight date, empty if irrelevant','selector':'string: public dot path, numeric list indices allowed; empty for flight status'},
        'outcome':result({'status':status(['completed','failed']),'reason':dict(status(['record_or_field_unavailable']),nullable=True),
                          'value':nullable('string'),'number':nullable('number'),'boolean':nullable('boolean')}),
        'outcome_scope_params':['record_type','selector']}
    # These are user-owned selection constraints, not hidden answer IDs.
    selection={'origin':'airport code','destination':'airport code','dates':'array of allowed ISO dates',
        'cabin':'enum basic_economy,economy,business','passenger_count':'integer 1..5',
        'max_fare':'maximum total ticket fare for all passengers, excluding bags and insurance',
        'max_stops':'integer 0 or 1','depart_after':'HH:MM earliest departure',
        'arrive_before':'HH:MM latest arrival; +1 suffix permits next day',
        'preference':'enum cheapest,direct_first,earliest_arrival; then earlier date/departure and flight number break ties',
        'fallback':'optional object overriding budget, max_stops, times or dates ONLY if primary set is empty; cannot override cabin, route or passenger count'}
    base['flight_selection']={'description':'Search public direct/one-stop tools, filter all customer constraints, choose the ranked best itinerary; use fallback only if no primary match.',
        'kind':'account','params':selection,
        'outcome':result({'status':status(['completed','no_match','failed']),
                         'reason':dict(status(['not_available','invalid_request']),nullable=True),
                         'flights':{'type':'array','order':'exact','items':{'type':'string'},'description':'Selected flight numbers in travel order, excluding unselected alternatives.'},
                         'dates':{'type':'array','order':'exact','items':{'type':'string'},
                                  'description':'One ISO departure date PER selected flight, in the same order as flights. Repeat a shared date for each leg: a spoken date applying to a two-leg itinerary yields two identical date entries, not one. Do not include dates of unselected alternatives.'},
                         'fare':nullable('number')})}
    base['constrained_booking']={'description':'Choose direct/one-stop outbound and optional return itineraries independently from stated constraints, then book with actual consent. The customer supplies preferences, not expected flight IDs.',
        'kind':'account','params':{'user_id':'supplied user ID','selection':'object: flight_selection params',
        'return_selection':'optional flight_selection object for the reverse route, same cabin and passengers; all return dates strictly after the selected outbound arrival date. Each direction has its own stated fare budget/ranking.',
        'passengers':'full list of passenger names and dates of birth','payment_id':'one saved payment method for entire actual total',
        'insurance':'enum yes,no','total_baggages':'integer requested bags'},
        'outcome':copy.deepcopy(base['booking_creation']['outcome']),'approval':'approve_book_<user_id>'}
    base['handover']={'description':'Transfer requests outside the tool/policy scope; for passenger-count changes and post-booking insurance, the customer accepts a policy explanation if denied and does not request escalation. Do not promise a human override.',
        'kind':'account','params':{'request_kind':'enum unsupported_service,change_passenger_count,add_insurance_after_booking',
        'summary':'customer request and handover summary'},
        'outcome':result({'status':status(['transferred','refused']),
                         'reason':dict(status(['change_passenger_count','add_insurance_after_booking']),nullable=True,
                                       description='Null when transfer is completed. For a refused operation, report the explained policy restriction.')})}
    return base


def extend(env, verifier):
    env += '\n'+(ROOT/'world_extensions/airline_reference.py').read_text()
    verifier += '\n'+(ROOT/'world_extensions/airline_verifier.py').read_text()
    old="            if process=='policy_inquiry':"
    new="""            if process=='record_query':
                out=extension_query(api,p)
            elif process=='flight_selection':
                out=extension_selection(api,p)
            elif process=='constrained_booking':
                out=extension_booking(api,p)
            elif process=='handover':
                out=extension_handover(api,p)
            elif process=='policy_inquiry':"""
    assert env.count(old)==1
    env=env.replace(old,new)
    # Initialize booking error fields even when the tool raises.
    env=env.replace("if process=='booking_creation': out['reservation_id']=None", "if process in ('booking_creation','constrained_booking'): out.update(reservation_id=None,amount=0)")
    old="            if proc == 'policy_inquiry':"
    new="""            if proc == 'record_query': return extension_expected_query(state,p), 'allowed'
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
            if proc == 'policy_inquiry':"""
    assert verifier.count(old)==1
    return env,verifier.replace(old,new)


def synthetic_state(request):
    tables=copy.deepcopy(request['tables']);example=tables['flights'][0]
    # Independently invented schedules and fares; no benchmark records.
    fixtures=[('UPD1','SFO','JFK','08:00:00','14:00:00',300),
              ('UPD2','SFO','JFK','09:00:00','15:00:00',220),
              ('UPC1','SFO','ORD','07:00:00','10:00:00',80),
              ('UPC2','ORD','JFK','11:00:00','14:00:00',90),
              ('UPR1','JFK','SFO','12:00:00','18:00:00',200)]
    # Isolate this route from the original synthetic catalogue.
    # Preserve every existing flight (reservations still reference them). Close
    # only competing departure-date inventory in this isolated probe fixture.
    for f in tables['flights']:
        if f['origin'] in ('SFO','ORD','JFK'):
            for d in ('2024-05-22','2024-05-23'):
                if d in f['dates']: f['dates'][d]['status']='cancelled'
    for name,origin,destination,departure,arrival,price in fixtures:
        f=copy.deepcopy(example);f.update(id=name,flight_number=name,origin=origin,destination=destination,
            scheduled_departure_time_est=departure,scheduled_arrival_time_est=arrival,
            dates={d:{'status':'available','prices':{'basic_economy':price-20,'economy':price,'business':price+100},
                       'available_seats':{'basic_economy':10,'economy':10,'business':10}} for d in ('2024-05-22','2024-05-23')})
        tables['flights'].append(f)
    return tables


def cases(request):
    tables=synthetic_state(request);u=tables['users'][0];r=tables['reservations'][0]
    selection=dict(origin='SFO',destination='JFK',dates=['2024-05-22'],cabin='economy',passenger_count=1,
                   max_fare=500,max_stops=1,depart_after='00:00',arrive_before='23:59',preference='cheapest')
    rows=[]
    def add(name,proc,params,answer,tools,table=None,approved=True):
        facts={'goals':[{'process':proc,'params':params}]}
        if proc=='constrained_booking':facts['approve_book_'+u['user_id']]=approved
        rows.append(dict(id=name,process=proc,params=params,facts=facts,tables=table or tables,
                         expected_answer=answer,required_tools=tools))
    def found(fs,fare,dates=None):return dict(status='completed',reason=None,flights=fs,dates=dates or ['2024-05-22']*len(fs),fare=fare)
    add('search_cheapest_connection','flight_selection',selection,found(['UPC1','UPC2'],170),['search_direct_flight','search_onestop_flight'])
    add('search_prefer_direct','flight_selection',dict(selection,preference='direct_first'),found(['UPD2'],220),['search_direct_flight'])
    add('search_time_bound','flight_selection',dict(selection,max_stops=0,arrive_before='14:00'),found(['UPD1'],300),['search_direct_flight'])
    add('search_departure_bound','flight_selection',dict(selection,depart_after='09:00'),found(['UPD2'],220),['search_direct_flight'])
    add('search_budget_boundary','flight_selection',dict(selection,max_fare=170),found(['UPC1','UPC2'],170),['search_onestop_flight'])
    add('search_no_match','flight_selection',dict(selection,max_fare=169),dict(status='no_match',reason='not_available',flights=[],dates=[],fare=None),['search_direct_flight'])
    add('search_invalid_fallback','flight_selection',dict(selection,fallback={'cabin':'business'}),dict(status='failed',reason='invalid_request',flights=[],dates=[],fare=None),[])
    add('search_fallback','flight_selection',dict(selection,max_stops=0,max_fare=200,fallback={'max_stops':1}),found(['UPC1','UPC2'],170),['search_onestop_flight'])
    add('search_no_early_fallback','flight_selection',dict(selection,max_stops=0,fallback={'max_stops':1}),found(['UPD2'],220),['search_direct_flight'])
    add('search_date_and_arrival','flight_selection',dict(selection,dates=['2024-05-23','2024-05-22'],preference='earliest_arrival'),found(['UPC1','UPC2'],170),['search_direct_flight'])
    scarce=copy.deepcopy(tables)
    for f in scarce['flights']:
        if f['flight_number']=='UPC1':f['dates']['2024-05-22']['available_seats']['economy']=0
    add('search_seat_availability','flight_selection',selection,found(['UPD2'],220),['search_direct_flight'],scarce)
    for name,kind,id,selector,value in [('query_membership','user',u['user_id'],'membership',u['membership']),
        ('query_passenger_name','reservation',r['reservation_id'],'passengers.0.first_name',r['passengers'][0]['first_name']),
        ('query_payment_amount','reservation',r['reservation_id'],'payment_history.0.amount',r['payment_history'][0]['amount']),
        ('query_segment_date','reservation',r['reservation_id'],'flights.0.date',r['flights'][0]['date']),
        ('query_flight_status','flight_status','UPD1','','available'),('query_airport','airports','','0.city','San Francisco')]:
        answer=dict(status='completed',reason=None,value=value if isinstance(value,str) else None,number=value if type(value) in (int,float) else None,boolean=None)
        add(name,'record_query',dict(record_type=kind,record_id=id,selector=selector,date='2024-05-22'),answer,
            [{'user':'get_user_details','reservation':'get_reservation_details','flight_status':'get_flight_status','airports':'list_all_airports'}[kind]])
    for kind in ('unsupported_service','change_passenger_count','add_insurance_after_booking'):
        add('handover_'+kind,'handover',dict(request_kind=kind,summary='Customer requests '+kind),
            dict(status='transferred' if kind=='unsupported_service' else 'refused',reason=None if kind=='unsupported_service' else kind),['transfer_to_human_agents'] if kind=='unsupported_service' else [])
    payment=next(k for k,v in u['payment_methods'].items() if v['source']=='credit_card')
    params=dict(user_id=u['user_id'],selection=selection,passengers=[dict(first_name='Aster',last_name='Probe',dob='1990-01-01')],
                payment_id=payment,total_baggages=0,insurance='no')
    add('search_then_book','constrained_booking',params,dict(status='completed',reason=None,reservation_id='HATHAT',amount=170),['search_onestop_flight','book_reservation'])
    add('search_then_decline','constrained_booking',params,dict(status='refused',reason='confirmation_declined',reservation_id=None,amount=0),['search_direct_flight'],approved=False)
    add('search_then_roundtrip_book','constrained_booking',dict(params,return_selection=dict(selection,
        origin='JFK',destination='SFO',dates=['2024-05-23'],max_stops=0)),
        dict(status='completed',reason=None,reservation_id='HATHAT',amount=370),['search_direct_flight','search_onestop_flight','book_reservation'])
    add('search_then_roundtrip_no_match','constrained_booking',dict(params,return_selection=dict(selection,
        origin='JFK',destination='SFO',dates=['2024-05-23'],max_stops=0,max_fare=199)),
        dict(status='refused',reason='not_available',reservation_id=None,amount=0),['search_direct_flight'])
    add('search_then_missing_payment','constrained_booking',dict(params,payment_id='not_a_saved_method'),
        dict(status='failed',reason='payment_not_saved',reservation_id=None,amount=0),['get_user_details'])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path('output/world_tau2_task_dialogue_v4/airline/world/bundle')
    loaded=bundle.load_bundle(source,require_current_platform=False);request=copy.deepcopy(loaded['request'])
    env,verifier=extend((source/'env.py').read_text(),(source/'verifier.py').read_text())
    stage=args.output/'airline/source';stage.mkdir(parents=True)
    (stage/'env.py').write_text(env);(stage/'verifier.py').write_text(verifier)
    request['goal_schema']=schemas(request['goal_schema'])
    request['id']='airline-coverage-v1'
    save(stage/'request.json',request);save(stage/'answer_schema.json',loaded['schema'])
    issues,notes=check_world(stage)
    save(args.output/'airline/world_check.json',{'passed':not issues,'issues':issues,'notes':notes})
    if issues: raise ValueError(issues)
    upgraded={**loaded,'request':request,'env':load_module(stage/'env.py',role='env'),
              'verifier':load_module(stage/'verifier.py',role='verifier')}
    qualification=bundle.qualify_loaded(upgraded);save(args.output/'airline/qualification.json',qualification)
    if not qualification['passed']: raise ValueError('upgraded world failed qualification')
    bundle.export(stage,stage/'request.json',qualification,args.output/'airline/bundle',status='qualified')
    probes=[]
    for case in cases(request):
        ref=run_reference(upgraded['env'],case['tables'],case['facts'],start=request['time']['start'])
        expected=upgraded['verifier'].expected(copy.deepcopy(case['tables']),copy.deepcopy(case['facts']),request['time']['start'])
        called={e.get('tool') for e in ref['trace'] if e.get('kind')=='tool' and e.get('status')=='ok'}
        problems=[]
        problems+=answer_problems(task_schema(request['goal_schema'],case['facts']['goals']),ref['answer'])
        if canonical(ref['answer'])!=canonical(expected['answer']):problems.append('reference/verifier answer mismatch')
        if canonical(ref['final_state'])!=canonical(expected['final_state']):problems.append('reference/verifier state mismatch')
        if canonical(ref['answer'].get('goal_1'))!=canonical(case['expected_answer']):problems.append('independent expected answer mismatch')
        if not set(case['required_tools'])<=called:problems.append('missing required tool execution')
        if ref['safety_violations'] or ref['tool_errors']:problems.append('execution safety/error')
        if case['process']!='constrained_booking' and canonical(ref['final_state'])!=canonical(case['tables']):problems.append('unexpected write')
        schema=task_schema(request['goal_schema'],case['facts']['goals'])
        positive=grade(ref,expected,schema,initial=case['tables'],reference_calls=ref['tool_calls'])
        if not positive['success']:problems.append('positive reward check failed')
        # A syntactically valid but false report must fail even for read-only goals.
        wrong=copy.deepcopy(ref)
        statuses=schema['fields']['goal_1']['properties']['status']['enum']
        wrong['answer']['goal_1']['status']=next(s for s in statuses if s!=ref['answer']['goal_1']['status'])
        negatives={'false_report':not grade(wrong,expected,schema,initial=case['tables'],reference_calls=ref['tool_calls'])['success']}
        if ref['final_state']!=case['tables']:
            wrong=copy.deepcopy(ref);wrong['final_state']=copy.deepcopy(case['tables'])
            negatives['words_without_write']=not grade(wrong,expected,schema,initial=case['tables'],reference_calls=ref['tool_calls'])['success']
        if not all(negatives.values()):problems.append('negative reward check failed')
        row={**case,'reference_passed':not problems,'natural_passed':False,'problems':problems,
             'reference':ref,'oracle':expected,'observed':ref['answer']['goal_1'],'tools':sorted(called),'negative_checks':negatives}
        save(args.output/'airline/probes'/f"{case['id']}.json",row);probes.append(row)
    save(args.output/'airline/probe_summary.json',{'total':len(probes),'passed':sum(r['reference_passed'] for r in probes),
        'failures':[{k:r[k] for k in ('id','problems','observed','expected_answer')} for r in probes if not r['reference_passed']]})
    cells=[{'id':r['id'],'predicates':[{'path':'process','equals':r['process']},
             ({'path':'params.request_kind','equals':r['params']['request_kind']} if r['process']=='handover' else
              {'path':'params','equals':r['params']}), {'path':'observed','equals':r['expected_answer']},
             *[{'path':'tools','contains':tool} for tool in r['required_tools']]]} for r in probes]
    save(args.output/'airline/coverage_contract.json',cells);save(args.output/'airline/coverage.json',coverage_report(cells,probes))
    save(args.output/'identity.json',{'source_bundle':str(source),'source_manifest':loaded['manifest'],
         'platform':bundle.platform_identity(),'code':{str(p):bundle.sha256(p) for p in [Path(__file__),*sorted((ROOT/'world_extensions').glob('*.py'))]},
         'limits':['Original hand-authored upgrade, not proof of automatic world construction.',
                   'Round trips rank outbound/return independently with separately stated budgets; arbitrary joint itinerary objectives are not claimed.',
                   'Natural dialogue admission is a separate required stage; probe pass alone is not SFT release.']})
    print(json.dumps({'world_qualified':True,'probes_passed':sum(r['reference_passed'] for r in probes),'total':len(probes)}))
    if any(not r['reference_passed'] for r in probes):raise ValueError('gap probes failed')


if __name__=='__main__':main()
