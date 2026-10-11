"""Independent truth for the new parcel/parking/FEFO/archive holdout.

No generated contract, platform expression evaluator or reward is imported.
"""
import copy
import math


def expected(spec, data, facts, *, refuse=False):
    family, mode = spec['kind'].removeprefix('fresh_').split('_', 1)
    initial, final = copy.deepcopy(data), copy.deepcopy(data)
    p = spec['params']
    def get(table, identity):
        return next(row for row in final[table] if row['id'] == identity)
    observations = copy.deepcopy(spec.get('observations', {}))
    answer = {}
    if family == 'parcel':
        def fee(parcel, service):
            units = math.ceil(max(parcel['grams'], parcel['volume_grams']) / 500)
            return units * service['unit_cents'] + (service['remote_cents'] if parcel['remote'] else 0)
        def options(parcel):
            return sorted((s for s in final['services'] if s['zone'] == parcel['zone']
                and s['remaining'] >= 1 and s['max_grams'] >= max(parcel['grams'], parcel['volume_grams'])
                and (not parcel['hazardous'] or s['hazardous'])), key=lambda s: (fee(parcel,s), s['id']))
        if mode == 'direct':
            answer = {'charge_cents': fee(p['parcel'], p['service'])}
        elif mode == 'simple':
            parcel = get('parcels', facts['parcel_id'])
            answer = {'eligible': any(s['id'] == facts['service_id'] for s in options(parcel))}
        elif mode == 'ordinary':
            eligible = options(get('parcels', facts['parcel_id']))
            answer = {'ids': [s['id'] for s in eligible], 'cheapest_cents': fee(get('parcels',facts['parcel_id']),eligible[0]) if eligible else 0}
            observations['services'] = answer['ids']
        else:
            sent = []
            for index, identity in enumerate(facts['parcel_ids']):
                parcel = get('parcels', identity)
                if parcel['status']!='pending' or get('dispatches',parcel['dispatch_id'])['done']: continue
                choices = options(parcel)
                if not choices: continue
                selected = choices[0]
                charge = fee(parcel, selected)
                wallet = get('wallets', facts['wallet_id'])
                if wallet['cents'] < charge: continue
                if not refuse and facts['approve_'+str(index)]:
                    selected['remaining'] -= 1
                    wallet['cents'] -= charge
                    parcel.update(status='sent', service_id=selected['id'])
                    get('dispatches', parcel['dispatch_id']).update(done=True, charge_cents=charge)
                    sent.append(identity)
            answer = {'ids': sent, 'wallet_cents': get('wallets', facts['wallet_id'])['cents']}
            if mode == 'large': observations['services'] = [s['id'] for s in initial['services']]
    elif family == 'parking':
        def tariff(minutes):
            return min(p['daily_cap_cents'], p['first_hour_cents'] + max(0, math.ceil((minutes-60)/30))*p['half_hour_cents'])
        def available(bay, start, end, exclude=None):
            return bay['active'] and (not facts['charger'] or bay['charger']) and not any(
                booking['status']=='booked' and booking['bay_id']==bay['id'] and booking['id']!=exclude
                and start < booking['end'] and booking['start'] < end for booking in final['bookings'])
        if mode=='direct': answer={'charge_cents':tariff(p['minutes'])}
        elif mode=='simple': answer={'eligible':available(get('bays',facts['bay_id']),facts['start'],facts['end'])}
        elif mode=='ordinary':
            answer={'ids':[b['id'] for b in final['bays'] if available(b,facts['start'],facts['end'])]}
            observations['bays']=[b['id'] for b in initial['bays']]
            observations['bookings']=[b['id'] for b in initial['bookings']]
        else:
            completed=[]
            for index, identity in enumerate(facts['booking_ids']):
                booking=get('bookings',identity)
                if booking['status']!=('booked' if mode=='extend' else 'pending'): continue
                new_end=booking['end']+facts['extension_minutes'] if mode=='extend' else booking['end']
                bay=get('bays',booking['bay_id'])
                charge=tariff(new_end-booking['start'])-tariff(booking['end']-booking['start']) if mode=='extend' else tariff(new_end-booking['start'])
                wallet=get('wallets',facts['wallet_id'])
                if available(bay,booking['start'],new_end,identity) and wallet['cents']>=charge and not refuse and facts['approve_'+str(index)]:
                    booking.update(status='booked',end=new_end,paid_cents=booking['paid_cents']+charge)
                    wallet['cents']-=charge
                    completed.append(identity)
            answer={'ids':completed,'wallet_cents':get('wallets',facts['wallet_id'])['cents']}
    elif family=='warehouse':
        def usable(lot):
            return not lot['quarantined'] and not lot['recalled'] and lot['expiry_day']>=p['day'] and lot['available']>0 and lot['sku']==facts['sku']
        def allocation():
            remaining=facts['units']; result=[]
            for lot in sorted((l for l in final['lots'] if usable(l)),key=lambda l:(l['expiry_day'],l['id'])):
                quantity=min(remaining,lot['available'])
                if quantity: result.append({'lot_id':lot['id'],'units':quantity})
                remaining-=quantity
                if not remaining: break
            return result,remaining
        if mode=='direct':
            gross=sum(row['units']*row['credit_cents'] for row in p['returns'] if row['quality']>=p['minimum_quality'])
            answer={'credit_cents':max(0,gross-p['handling_cents'])}
        elif mode=='simple': answer={'eligible':usable(get('lots',facts['lot_id']))}
        elif mode=='ordinary':
            result,remaining=allocation();answer={'allocation':result,'shortfall':remaining}
            observations['lots']=[r['id'] for r in final['lots'] if usable(r)]
        else:
            result,remaining=allocation()
            if remaining or refuse or not facts['approve_0'] or get('orders',facts['order_id'])['reserved']: result=[]
            else:
                for item in result: get('lots',item['lot_id'])['available']-=item['units']
                order=get('orders',facts['order_id']);order.update(reserved=True,units=facts['units'])
                get('capacity',facts['capacity_id'])['reserved_units']+=facts['units']
            answer={'allocation':result,'reserved_units':get('orders',facts['order_id'])['units']}
            observations['lots']=[r['id'] for r in initial['lots'] if usable(r)]
    elif family=='archive':
        if mode=='direct':
            unique={row['blob_id']:row['bytes'] for row in p['versions'] if row['age_days']<=p['retain_days'] or row['pinned']}
            answer={'retained_bytes':sum(unique.values())}
        else:
            def protected(version): return version['pinned'] or version['age_days']<=p['retain_days']
            def ancestors(identity):
                result=[]
                while identity:
                    if identity in result: raise ValueError('independent archive fixture must be acyclic')
                    result.append(identity);identity=get('versions',identity)['parent_id']
                return result
            if mode=='simple':
                v=get('versions',facts['version_id'])
                answer={'eligible':v['live'] and not protected(v) and not any(r['live'] and r['parent_id']==v['id'] for r in final['versions'])}
            elif mode=='ordinary':
                chain=ancestors(facts['version_id'])
                blobs={get('versions',i)['blob_id'] for i in chain}
                answer={'chain':chain,'unique_blob_bytes':sum(get('blobs',i)['bytes'] for i in blobs)}
                observations['versions']=chain
                observations['blobs']=sorted(blobs)
            else:
                selected=ancestors(facts['version_id']) if mode=='chain' else [r['id'] for r in final['versions'] if r['live'] and not protected(r)]
                # Greatest deletable subset: retaining a protected/referenced child
                # retains its parent transitively, even if the parent is old.
                deletable={i for i in selected if get('versions',i)['live'] and not protected(get('versions',i))}
                while True:
                    blocked={r['parent_id'] for r in final['versions'] if r['live'] and r['id'] not in deletable}
                    reduced=deletable-blocked
                    if reduced==deletable: break
                    deletable=reduced
                if refuse or not facts['approve_0']: deletable=set()
                for identity in deletable: get('versions',identity)['live']=False
                if deletable:
                    referenced={r['blob_id'] for r in final['versions'] if r['live']}
                    for blob in final['blobs']:
                        if blob['id'] not in referenced: blob['stored']=False
                    get('quotas',facts['quota_id'])['used_bytes']=sum(b['bytes'] for b in final['blobs'] if b['stored'])
                answer={'ids':sorted(deletable),'used_bytes':get('quotas',facts['quota_id'])['used_bytes']}
                observations['versions']=[r['id'] for r in initial['versions']]
    else: raise ValueError('unknown fresh business family')
    groups = None
    if mode not in {'direct','simple','ordinary'}:
        if family in {'parcel','parking'}:
            groups = [{'table':'parcels' if family=='parcel' else 'bookings','ids':[i]} for i in answer['ids']]
        else:
            groups = ([{'table':'orders','ids':[facts['order_id']]}] if answer.get('reserved_units',0)>0 and final!=initial
                      else [{'table':'versions','ids':answer['ids']}] if answer.get('ids') else [])
    return {'answer':answer,'initial':initial,'final_state':final,'observations':observations,
            **({'transaction_groups':groups} if groups is not None else {})}


def transaction_group_checks(world, groups):
    state=copy.deepcopy(world.initial)
    keys={table['id']:table['primary_key'] for table in world.source['tables']}
    writes=[]
    for turn in world.trace:
        if not (turn.get('kind')=='tool' and turn.get('status')=='ok'
                and world.tools[turn['tool']]['kind']=='transaction'):
            continue
        effective={}
        for change in turn['result']['changes']:
            table=change['table']
            row=next(r for r in state[table] if r[keys[table]]==change['key'])
            if any(row.get(column)!=value for column,value in change['set'].items()):
                effective.setdefault(table,set()).add(row['id'])
            row.update(copy.deepcopy(change['set']))
        if effective: writes.append(effective)
    # Redundant no-op calls belong to efficiency scoring, not business rejection.
    matches=len(writes)==len(groups) and all(write.get(group['table'],set())==set(group['ids'])
        for write,group in zip(writes,groups))
    return [{'id':'independent_declared_atomic_transaction_groups','passed':bool(matches)}]
