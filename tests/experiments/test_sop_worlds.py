from copy import deepcopy

import pytest

from env_factory import sop
from env_factory.core.sandbox import load_module
from env_factory.runtime import tool_specs, run_reference, Episode
from test_semantic_worlds import data, SOURCE
from world_extensions.semantic_cases import make
from world_extensions.sop_specs import blueprint, executable_contract, make_behavior


def tools(env):
    return [{'type':'function','function':t} for t in tool_specs(env.make_tools(),consent=False)]


@pytest.mark.parametrize('domain',['retail','airline'])
def test_domain_graphs_on_held_out_data(domain):
    env=load_module(SOURCE/f'{domain}_env.py',role='env')
    request={'public_contract':{'tools':tools(env)}}
    design=blueprint(domain,request)
    cases=executable_contract(make_behavior(domain,data(domain),seeds=(101,307)))['cases']
    for case in cases:
        result=sop.run_graph(env,design,case)
        assert result['passed'],(case['id'],result)


def test_booking_identifier_is_bound_at_runtime_and_ablated():
    env=load_module(SOURCE/'airline_env.py',role='env')
    verifier=load_module(SOURCE/'airline_verifier.py',role='verifier')
    case=make('airline',data('airline'),seeds=(103,))['cases'][0]
    tables=deepcopy(case['data']);u=tables['users'][0];uid=u['user_id']
    tables['users'].append({**deepcopy(u),'id':'other','user_id':'other','reservations':[]})
    booking=case['steps'][0]['args']
    facts={'user_id':uid,'approve_book_'+uid:True,'goals':[
        {'process':'booking_creation','params':{'user_id':uid,'booking':booking}},
        {'process':'reservation_inspection','params':{'user_id':uid,'reservation_id':None},
         'bindings':{'reservation_id':{'goal':1,'path':'reservation_id'}}}]}
    request={'public_contract':{'tools':tools(env)},'tables':tables,
             'user_facts':facts,'time':{'start':case['now']},'sop_method':'book_then_inspect'}
    request['sop_blueprint']=blueprint('airline',request)
    reference=run_reference(env,tables,facts,start=case['now'])
    expected=verifier.expected(tables,facts,case['now'])
    assert reference['answer']==expected['answer']
    assert reference['final_state']==expected['final_state']
    assert reference['answer']['goal_2']['status']=='completed'
    report=sop.task_report(env,verifier,request)
    assert report['passed'],report
    assert report['witnesses'][0]['observed']
    from env_factory.production.world import goal_sensitivity
    sensitivity=goal_sensitivity(verifier,request)
    assert sensitivity['passed'] and sensitivity['checked']==2,sensitivity


def verify_method(domain,name,tables,facts,now):
    env=load_module(SOURCE/f'{domain}_env.py',role='env')
    verifier=load_module(SOURCE/f'{domain}_verifier.py',role='verifier')
    request={'public_contract':{'tools':tools(env)},'tables':tables,'user_facts':facts,
             'time':{'start':now},'sop_method':name}
    request['sop_blueprint']=blueprint(domain,request)
    reference=run_reference(env,tables,facts,start=now)
    expected=verifier.expected(deepcopy(tables),deepcopy(facts),now)
    assert reference['answer']==expected['answer'],(name,reference['answer'],expected['answer'])
    assert reference['final_state']==expected['final_state']
    report=sop.task_report(env,verifier,request)
    assert report['passed'],(name,report,reference['answer'])
    assert sop.trace_report(env,request)['passed']


@pytest.mark.parametrize('name',['replace_funding_then_cancel','correct_address_then_funding',
    'unavailable_exchange_then_return','correct_address_then_verify'])
def test_retail_methods_have_real_executable_witnesses(name):
    case=next(c for c in make('retail',data('retail'),seeds=(109,))['cases'] if 'pair-prices' in c['id'])
    tables=deepcopy(case['data']);u=tables['users'][0];o=tables['orders'][0];oid=o['order_id']
    u.update(email='customer@example.test',name={'first_name':'Ada','last_name':'Example'})
    address=dict(address1='24 New Street',address2='',city='NewCity',state='CA',country='USA',zip='94000')
    funding={'process':'payment_method_change','params':{'order_id':oid,'payment_method_id':'gift'}}
    shipping={'process':'shipping_address_change','params':dict(order_id=oid,**address)}
    if name=='replace_funding_then_cancel':
        goals=[funding,{'process':'cancel','params':{'order_id':oid,'reason':'no longer needed'}}]
    elif name=='correct_address_then_funding':goals=[shipping,funding]
    elif name=='correct_address_then_verify':
        goals=[shipping,{'process':'account_lookup','params':{'record_type':'order','record_id':oid,'fields':['address.city'],'selector':''}}]
    else:
        o['status']='delivered';p=deepcopy(case['steps'][0]['args'])
        for product,variant in zip(tables['products'],p['new_item_ids']):product['variants'][variant]['available']=False
        goals=[{'process':'delivered_exchange_request','params':p},
               {'process':'delivered_return_request','params':{k:v for k,v in p.items() if k!='new_item_ids'}}]
    facts={'email':u['email'],'goals':goals,**{'approve_'+g['process']+'_'+oid:True for g in goals}}
    verify_method('retail',name,tables,facts,'2024-06-01T11:30:00')


@pytest.mark.parametrize('name',['change_then_cancel','cabin_then_baggage','change_then_delay_compensation',
    'cancel_then_delay_compensation','correct_passengers_then_baggage','compensate_then_book'])
def test_airline_methods_have_real_executable_witnesses(name):
    env=load_module(SOURCE/'airline_env.py',role='env')
    case=make('airline',data('airline'),seeds=(113,))['cases'][0]
    api=Episode(env.make_tools(),case['data'],{},start=case['now'],approve=lambda *_:True)
    booking=api.call('book_reservation',**case['steps'][0]['args'])
    tables=api.db;u=tables['users'][0];rid=booking['reservation_id'];uid=u['user_id'];date=booking['flights'][0]['date']
    common=dict(user_id=uid,reservation_id=rid)
    itinerary={'process':'itinerary_change','params':dict(common,cabin='economy',
        flights=[{'flight_number':tables['flights'][1]['flight_number'],'date':date}],payment_id='gift')}
    cancellation={'process':'eligible_cancellation','params':dict(common,cancellation_reason='health')}
    baggage={'process':'adding_baggage','params':dict(common,total_baggages=3,payment_id='gift')}
    compensation={'process':'requested_compensation','params':dict(common,requested=True,complaint='delayed')}
    if name=='change_then_cancel':goals=[itinerary,cancellation]
    elif name=='cabin_then_baggage':goals=[{'process':'cabin_change','params':dict(common,cabin='business',payment_id='gift')},baggage]
    elif name in ('change_then_delay_compensation','cancel_then_delay_compensation'):
        u['membership']='gold';tables['flights'][0]['dates'][date]['status']='delayed'
        goals=[itinerary if name.startswith('change') else cancellation,compensation]
    elif name=='correct_passengers_then_baggage':
        passenger=deepcopy(booking['passengers']);passenger[0]['first_name']='Corrected'
        goals=[{'process':'passenger_list_update','params':dict(common,passengers=passenger)},baggage]
    else:
        u['membership']='gold';tables['flights'][0]['dates'][date]['status']='cancelled'
        compensation['params']['complaint']='cancelled'
        tables['flights'][1]['dates'][date]['prices']['economy']=80
        goals=[compensation,{'process':'constrained_booking','params':dict(user_id=uid,
            selection={'origin':'SFO','destination':'JFK','dates':[date],'cabin':'economy','passenger_count':1,'max_fare':100,'max_stops':0,'preference':'cheapest'},
            passengers=booking['passengers'],total_baggages=0,insurance='no',payment_id=None),
            'bindings':{'payment_id':{'goal':1,'path':'certificate_id'}}}]
    prefixes={'itinerary_change':'itinerary','eligible_cancellation':'cancel','adding_baggage':'baggage',
              'requested_compensation':'compensation','cabin_change':'cabin','passenger_list_update':'passengers','constrained_booking':'book'}
    facts={'user_id':uid,'goals':goals,**{'approve_'+prefixes[g['process']]+'_'+(uid if g['process']=='constrained_booking' else rid):True for g in goals}}
    verify_method('airline',name,tables,facts,case['now'])
