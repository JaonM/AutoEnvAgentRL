"""Held-out synthetic property cases and deliberate semantic mutations; no provider calls."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from world_extensions.semantic_cases import make
from env_factory.certify import behavior
from env_factory.core.sandbox import load_module

SOURCE=Path(__file__).resolve().parents[2]/'experiments/world_extensions/coherent'


def data(domain):
    methods={'card':{'id':'card','source':'credit_card'},'gift':{'id':'gift','source':'gift_card','amount':10000,'balance':1000}}
    user={'id':'u','user_id':'u','payment_methods':methods,'address':{},'membership':'regular',
          'saved_passengers':[{'first_name':'A','last_name':'B','dob':'1990-01-01'}]}
    if domain=='airline':
        return {'users':[user],'reservations':[],'flights':[{'id':'f','flight_number':'f','origin':'SFO','destination':'JFK',
            'scheduled_departure_time_est':'09:00:00','scheduled_arrival_time_est':'14:00:00','dates':{}}]}
    products=[{'id':f'p{i}','product_id':f'p{i}','name':f'Item{i}', 'variants':{
        f'v{i}{j}':{'item_id':f'v{i}{j}','price':10+j,'available':True,'options':{'size':str(j)}} for j in range(2)}} for i in range(2)]
    return {'users':[user],'orders':[{'id':'o','order_id':'o','status':'pending','user_id':'u','items':[],
        'payment_history':[],'return_items':None,'return_payment_method_id':None}],'products':products}


@pytest.mark.parametrize('domain',['retail','airline'])
def test_unseen_ids_amounts_and_sequential_effects(domain):
    env=load_module(SOURCE/f'{domain}_env.py',role='env')
    contract=make(domain,data(domain),seeds=(101,307,997))
    report=behavior.run(env,contract)
    assert report['passed'],report


@pytest.mark.parametrize('domain,old,new',[
    ('retail',"item['price']=variant['price']","item['price']=variants[-1]['price']"),
    ('retail','_balance(method,-delta)','_balance(method,0)'),
    ('airline',"_inventory(db,r['flights'],[],len(r['passengers']),r['cabin'],r['cabin'])",
               "_inventory(db,[],[],len(r['passengers']),r['cabin'],r['cabin'])"),
    ('airline',"method['amount']=float(method['amount']+amount)","method['amount']=float(method['amount'])"),
    ('airline','created_at=datetime.fromisoformat(now).isoformat()',"created_at='2024-05-15T15:00:00'"),
    ('airline','rid=_identifier(db)',"rid=_identifier(db)\n        if len(db['reservations'])>=3: raise BusinessError('Too many reservations')"),
])
def test_acceptance_kills_deliberate_defects(tmp_path,domain,old,new):
    source=(SOURCE/f'{domain}_env.py').read_text();assert old in source
    candidate=tmp_path/'env.py';candidate.write_text(source.replace(old,new))
    env=load_module(candidate,role='env')
    assert not behavior.run(env,make(domain,data(domain),seeds=(101,)))['passed']
