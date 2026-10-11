import copy
import sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments'))
from prepare_v2_fresh20 import cases
from v2_natural_oracle import expected
from v2_fresh_business_oracle import transaction_group_checks


def truth(index, refuse=False):
    row=cases()[index-1]
    return expected(row['acceptance'],row['business'],row['user'],refuse=refuse)


def test_predeclared_mix_and_independent_arithmetic():
    rows=cases()
    assert len(rows)==20
    assert Counter(r['stratum'] for r in rows)==dict(direct=4,simple=4,ordinary=4,complex=8)
    assert [truth(i)['answer'] for i in [1,6,11,16]]==[
        {'charge_cents':790},{'charge_cents':900},{'credit_cents':245},{'retained_bytes':300}]


def test_capacity_and_wallet_failures_preserve_first_dispatch():
    result=truth(4)
    assert result['answer']=={'ids':['P1'],'wallet_cents':50}
    assert result['final_state']['parcels'][1]==result['initial']['parcels'][1]
    assert truth(5)['answer']=={'ids':['P1'],'wallet_cents':9697}


def test_half_open_intervals_and_extension_conflicts():
    assert truth(7)['answer']=={'eligible':True}
    assert truth(8)['answer']=={'ids':[]}
    assert truth(9)['answer']=={'ids':['N1'],'wallet_cents':580}
    assert truth(10)['answer']=={'ids':['N1'],'wallet_cents':880}


def test_fefo_orders_expiry_before_identity_and_rejects_ineligible_lots():
    result=truth(13)['answer']
    assert result=={'allocation':[{'lot_id':'L1','units':4},{'lot_id':'L3','units':7},{'lot_id':'L4','units':9}],'shortfall':5}
    assert truth(14)['answer']['reserved_units']==15
    assert [r['lot_id'] for r in truth(15)['answer']['allocation']]==['L0001','L0006','L0016','L0021','L0036','L0046']


def test_archive_transitive_dependency_and_shared_blob_accounting():
    assert truth(17)['answer']=={'eligible':False}
    assert truth(18)['answer']['unique_blob_bytes']==1161
    assert truth(18)['answer']['chain']==[f'V{i:02d}' for i in range(11,-1,-1)]
    assert truth(19)['answer']['used_bytes']==0
    assert len(truth(19)['answer']['ids'])==12
    result=truth(20)
    assert len(result['answer']['ids'])==22
    assert result['answer']['used_bytes']==20098
    assert result['final_state']['blobs'][0]['stored'] is True
    row=cases()[18]
    row['business']['tables'][0]['rows'][-1]['pinned']=True
    blocked=expected(row['acceptance'],row['business'],row['user'])
    assert blocked['answer']['ids']==[]
    assert blocked['final_state']==blocked['initial']


@pytest.mark.parametrize('index',[4,5,9,10,14,15,19,20])
def test_refusal_preserves_all_state(index):
    result=truth(index,True)
    assert result['final_state']==result['initial']
    assert result['transaction_groups']==[]


def test_reference_calculation_does_not_mutate_fixture():
    row=cases()[19];before=copy.deepcopy(row)
    expected(row['acceptance'],row['business'],row['user'])
    assert row==before


def test_atomic_receipt_check_rejects_split_or_out_of_order_transactions():
    world=SimpleNamespace(tools={'write':{'kind':'transaction'}},trace=[],
        initial={'versions':[{'id':'V1','live':True},{'id':'V2','live':True}]},
        source={'tables':[{'id':'versions','primary_key':'id'}]})
    group=[{'table':'versions','ids':['V1','V2']}]
    for key in ['V1','V2']:
        world.trace.append({'kind':'tool','tool':'write','status':'ok','result':{'changes':[{'table':'versions','key':key,'set':{'live':False}}]}})
    assert not transaction_group_checks(world,group)[0]['passed']
    world.trace=[{**world.trace[0],'result':{'changes':[{'table':'versions','key':k,'set':{'live':False}} for k in ['V1','V2']]}}]
    assert transaction_group_checks(world,group)[0]['passed']

    world.trace.append(copy.deepcopy(world.trace[0]))
    assert transaction_group_checks(world,group)[0]['passed']  # no-op is an efficiency issue


def test_repeated_business_requests_do_not_charge_or_reserve_again():
    for index,table_id,column,value in [(4,'parcels','status','sent'),(9,'bookings','status','booked'),(14,'orders','reserved',True)]:
        row=cases()[index-1]
        table=next(t for t in row['business']['tables'] if t['id']==table_id)
        targets=row['user']['facts'].get('parcel_ids',row['user']['facts'].get('booking_ids',['O1']))
        for record in table['rows']:
            if record['id'] in targets: record[column]=value
        result=expected(row['acceptance'],row['business'],row['user'])
        assert result['final_state']==result['initial']
        assert result['transaction_groups']==[]


def test_explicit_workspace_gc_scope_includes_orphans_only_after_confirmed_deletion():
    row=cases()[18]
    blobs=next(t for t in row['business']['tables'] if t['id']=='blobs')['rows']
    blobs.append({'id':'orphan','bytes':50,'stored':True})
    next(t for t in row['business']['tables'] if t['id']=='quotas')['rows'][0]['used_bytes']+=50
    for refused in [False,True]:
        result=expected(row['acceptance'],row['business'],row['user'],refuse=refused)
        assert result['final_state']['blobs'][-1]['stored'] is refused
        if refused: assert result['final_state']==result['initial']
