"""Public retail API with independent policy-following reference agent."""
from env_factory.runtime import write, BusinessError
import json
from copy import deepcopy
from collections import Counter

PROCESSES = {
 'cancel_pending_order':'cancel', 'modify_pending_order_address':'shipping_address_change',
 'modify_pending_order_items':'pending_items_change','modify_pending_order_payment':'payment_method_change',
 'return_delivered_order_items':'delivered_return_request','exchange_delivered_order_items':'delivered_exchange_request',
 'modify_user_address':'default_address_change'}
ADDRESS = ('address1','address2','city','state','country','zip')

def _row(db, table, key, value):
    for row in db[table]:
        if row[key] == value: return row
    raise BusinessError('record_not_found')

def _public(row):
    return deepcopy({k:v for k,v in row.items() if k != 'id'})

def _method(db, order, method):
    methods = _row(db,'users','user_id',order['user_id'])['payment_methods']
    if method not in methods: raise BusinessError('payment_not_owned')
    return methods[method]

def _balance(method, delta):
    if method['source']=='gift_card': method['balance']=round(method['balance']+delta,2)

def _enough(method, amount):
    if method['source']=='gift_card' and method['balance'] < amount: raise BusinessError('insufficient_gift_balance')

def _selection(order, ids):
    if Counter(ids) - Counter(x['item_id'] for x in order['items']): raise BusinessError('items_not_found')

def _variants(db, order, ids, new_ids, exchange=False):
    _selection(order,ids)
    if len(ids)!=len(new_ids): raise BusinessError('list_length_mismatch')
    delta=0
    variants=[]
    selected=set()
    for old,new in zip(ids,new_ids):
        if not exchange and old==new: raise BusinessError('same_item')
        index,item=next((i,x) for i,x in enumerate(order['items']) if x['item_id']==old and i not in selected)
        selected.add(index)
        product=_row(db,'products','product_id',item['product_id'])
        if new not in product['variants']: raise BusinessError('wrong_product')
        variant=product['variants'][new]
        if not variant['available']: raise BusinessError('unavailable')
        delta+=variant['price']-item['price']
        variants.append(variant)
    return (round(delta,2) if exchange else delta), variants

def calculate(db, now, expression: str):
    """Calculate numeric arithmetic safely, returning rounded decimal text.

    Args:
        expression: Digits, spaces, decimal points, parentheses and +, -, *, / operators; Python ** and // are supported.
    """
    if not all(c in '0123456789+-*/(). ' for c in expression): raise BusinessError('Invalid characters in expression')
    source=expression.strip()
    cursor=[0]
    def skip():
        while cursor[0]<len(source) and source[cursor[0]]==' ': cursor[0]+=1
    def number():
        start=cursor[0]
        while cursor[0]<len(source) and source[cursor[0]] in '0123456789.': cursor[0]+=1
        return float(source[start:cursor[0]])
    def atom():
        skip()
        if cursor[0]<len(source) and source[cursor[0]]=='(':
            cursor[0]+=1
            value=addition()
            skip()
            if cursor[0]>=len(source) or source[cursor[0]]!=')': raise ValueError('Missing parenthesis')
            cursor[0]+=1
            return value
        return number()
    def power():
        value=atom()
        skip()
        if source[cursor[0]:cursor[0]+2]=='**':
            cursor[0]+=2
            exponent=unary()
            if abs(exponent)>1000 or abs(value)>1e100: raise ValueError('Expression too large')
            value=value**exponent
        return value
    def unary():
        skip()
        if cursor[0]<len(source) and source[cursor[0]] in '+-':
            sign=source[cursor[0]];cursor[0]+=1
            value=unary()
            return value if sign=='+' else -value
        return power()
    def multiplication():
        value=unary()
        skip()
        while cursor[0]<len(source) and source[cursor[0]] in '*/':
            op=source[cursor[0]];cursor[0]+=1
            if op=='/' and source[cursor[0]:cursor[0]+1]=='/':
                cursor[0]+=1;op='//'
            other=unary()
            value=value*other if op=='*' else value//other if op=='//' else value/other
            skip()
        return value
    def addition():
        value=multiplication()
        skip()
        while cursor[0]<len(source) and source[cursor[0]] in '+-':
            op=source[cursor[0]];cursor[0]+=1;other=multiplication()
            value=value+other if op=='+' else value-other
            skip()
        return value
    try:
        value=addition()
        if cursor[0]!=len(source): raise ValueError('Unexpected token')
        return str(round(float(value),2))
    except Exception as exc: raise BusinessError('Invalid expression') from exc

def find_user_id_by_email(db, now, email: str):
    """Locate a customer by case-insensitive exact email.

    Args:
        email: Registered email, such as 'alex@example.test'; no whitespace normalization.
    """
    for u in db['users']:
        if u['email'].lower()==email.lower(): return u['user_id']
    raise BusinessError('record_not_found')

def find_user_id_by_name_zip(db, now, first_name: str, last_name: str, zip: str):
    """Locate a customer by name and exact postal code when email is unavailable.

    Args:
        first_name: First name, case insensitive.
        last_name: Last name, case insensitive.
        zip: Exact registered postal code, such as '12345'.
    """
    for u in db['users']:
        if u['name']['first_name'].lower()==first_name.lower() and u['name']['last_name'].lower()==last_name.lower() and u['address']['zip']==zip: return u['user_id']
    raise BusinessError('record_not_found')

def get_user_details(db, now, user_id: str):
    """Retrieve a customer's public profile, payment methods and order identifiers.

    Args:
        user_id: Customer identifier such as 'alex_elm_999', obtained through identity lookup.
    """
    return _public(_row(db,'users','user_id',user_id))

def get_order_details(db, now, order_id: str):
    """Retrieve an order including items, payments, status and tracking fulfillments.

    Args:
        order_id: Order identifier including #, such as '#W0000000'.
    """
    return _public(_row(db,'orders','order_id',order_id))

def get_product_details(db, now, product_id: str):
    """Retrieve one product and its variant options, prices and availability.

    Args:
        product_id: Opaque product identifier such as '6086499569', obtained from an order or product listing.
    """
    return _public(_row(db,'products','product_id',product_id))

def get_item_details(db, now, item_id: str):
    """Retrieve a variant's options, price and availability.

    Args:
        item_id: Opaque variant identifier such as '1008292230', obtained from order or product details.
    """
    for p in db['products']:
        if item_id in p['variants']: return deepcopy(p['variants'][item_id])
    raise BusinessError('record_not_found')

def list_all_product_types(db, now):
    """Return JSON text mapping alphabetically sorted product names to product IDs.

    Args:
    """
    return json.dumps(dict(sorted((p['name'],p['product_id']) for p in db['products'])))

@write
def cancel_pending_order(db, now, order_id: str, reason: str):
    """Cancel a strictly pending order and refund outstanding net funding to each method.

    Args:
        order_id: Order identifier such as '#W0000000'.
        reason: Exactly 'no longer needed' or 'ordered by mistake'.
    """
    o=_row(db,'orders','order_id',order_id)
    if o['status']!='pending': raise BusinessError('not_pending')
    if reason not in ('no longer needed','ordered by mistake'): raise BusinessError('invalid_cancel_reason')
    refunds=[];outstanding={}
    for p in o['payment_history']:
        mid=p['payment_method_id']
        outstanding[mid]=round(outstanding.get(mid,0)+p['amount']*(1 if p['transaction_type']=='payment' else -1),2)
    debt=round(-sum(min(v,0) for v in outstanding.values()),2)
    if round(sum(outstanding.values()),2)<0: raise BusinessError('invalid_payment_history')
    for mid,net in outstanding.items():
        amount=round(max(0,net-debt),2)
        debt=round(max(0,debt-max(net,0)),2)
        if amount:
            _balance(_method(db,o,mid),amount)
            refunds.append(dict(transaction_type='refund',amount=amount,payment_method_id=mid))
    o['status']='cancelled';o['cancel_reason']=reason;o['payment_history'].extend(refunds)
    return _public(o)

@write
def modify_pending_order_address(db, now, order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str):
    """Replace shipping address for an order whose status contains pending.

    Args:
        order_id: Order identifier such as '#W0000000'.
        address1: Exact primary address line.
        address2: Exact secondary address line, possibly empty.
        city: Exact city name.
        state: Exact state or province.
        country: Exact country.
        zip: Exact postal code string.
    """
    o=_row(db,'orders','order_id',order_id)
    if 'pending' not in o['status']: raise BusinessError('not_pending')
    o['address']=dict(address1=address1,address2=address2,city=city,state=state,country=country,zip=zip)
    return _public(o)

@write
def modify_user_address(db, now, user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str):
    """Replace the profile default address, independently of existing orders.

    Args:
        user_id: Customer identifier such as 'alex_elm_999', obtained by identity lookup.
        address1: Exact primary address line.
        address2: Exact secondary address line, possibly empty.
        city: Exact city name.
        state: Exact state or province.
        country: Exact country.
        zip: Exact postal code string.
    """
    u=_row(db,'users','user_id',user_id)
    u['address']=dict(address1=address1,address2=address2,city=city,state=state,country=country,zip=zip)
    return _public(u)

@write
def modify_pending_order_items(db, now, order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str):
    """Replace each selected occurrence with its positional variant and settle the signed price difference.

    Args:
        order_id: Order identifier such as '#W0000000'.
        item_ids: Old variant IDs such as '1008292230', from order details; duplicates select multiple occurrences.
        new_item_ids: Positional replacement variant IDs such as '1008292231', from product details; duplicates preserved.
        payment_method_id: Owned method ID such as 'gift_card_0000000', from customer details; pays positive or receives negative difference in public decimal units.
    """
    o=_row(db,'orders','order_id',order_id)
    if o['status']!='pending': raise BusinessError('not_pending')
    delta,variants=_variants(db,o,item_ids,new_item_ids)
    method=_method(db,o,payment_method_id);_enough(method,delta)
    o['payment_history'].append(dict(transaction_type='payment' if delta>0 else 'refund',amount=abs(delta),payment_method_id=payment_method_id))
    _balance(method,-delta)
    remaining=list(enumerate(o['items']));changes=[]
    for old,new,variant in zip(item_ids,new_item_ids,variants):
        selected=next(((i,x) for i,x in remaining if x['item_id']==old),None)
        if selected is None: raise BusinessError('items_not_found')
        index,item=selected;remaining=[(i,x) for i,x in remaining if i!=index]
        changes.append((item,new,variant))
    for item,new,variant in changes:
        item['item_id']=new;item['price']=variant['price'];item['options']=deepcopy(variant['options'])
    o['status']='pending (item modified)'
    return _public(o)

@write
def modify_pending_order_payment(db, now, order_id: str, payment_method_id: str):
    """Replace pending-order funding, charging the new method and refunding the original.

    Args:
        order_id: Order identifier such as '#W0000000'.
        payment_method_id: Owned payment ID such as 'credit_card_0000000', from customer details; gift balance must cover full decimal amount.
    """
    o=_row(db,'orders','order_id',order_id)
    if 'pending' not in o['status']: raise BusinessError('not_pending')
    method=_method(db,o,payment_method_id)
    if len(o['payment_history'])!=1 or o['payment_history'][0]['transaction_type']!='payment': raise BusinessError('payment_history_not_single_payment')
    p=o['payment_history'][0]
    if p['payment_method_id']==payment_method_id: raise BusinessError('same_payment_method')
    _enough(method,p['amount'])
    o['payment_history'].extend([dict(transaction_type='payment',amount=p['amount'],payment_method_id=payment_method_id),dict(transaction_type='refund',amount=p['amount'],payment_method_id=p['payment_method_id'])])
    _balance(method,-p['amount']);_balance(_method(db,o,p['payment_method_id']),p['amount'])
    return _public(o)

@write
def return_delivered_order_items(db, now, order_id: str, item_ids: list[str], payment_method_id: str):
    """Record a delivered-item return request; instructions follow by email and no refund is settled now.

    Args:
        order_id: Order identifier such as '#W0000000'.
        item_ids: Selected IDs such as '1008292230', from order details, preserving multiplicity.
        payment_method_id: Original payment method or owned gift card such as 'gift_card_0000000', from customer/order details.
    """
    o=_row(db,'orders','order_id',order_id)
    if o['status']!='delivered': raise BusinessError('not_delivered')
    method=_method(db,o,payment_method_id)
    if method['source']!='gift_card' and payment_method_id!=o['payment_history'][0]['payment_method_id']: raise BusinessError('invalid_refund_destination')
    _selection(o,item_ids)
    o['status']='return requested';o['return_items']=sorted(item_ids);o['return_payment_method_id']=payment_method_id
    return _public(o)

@write
def exchange_delivered_order_items(db, now, order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str):
    """Record a delivered-item exchange with rounded signed decimal difference; no settlement or item mutation occurs.

    Args:
        order_id: Order identifier such as '#W0000000'.
        item_ids: Old variant IDs such as '1008292230', from order details, with duplicates preserved.
        new_item_ids: Positional replacement IDs such as '1008292231', from product details, same product and available.
        payment_method_id: Owned method ID such as 'gift_card_0000000', from customer details, with sufficient gift balance for signed decimal difference.
    """
    o=_row(db,'orders','order_id',order_id)
    if o['status']!='delivered': raise BusinessError('not_delivered')
    delta,_=_variants(db,o,item_ids,new_item_ids,True)
    method=_method(db,o,payment_method_id);_enough(method,delta)
    o.update(status='exchange requested',exchange_items=sorted(item_ids),exchange_new_items=sorted(new_item_ids),exchange_payment_method_id=payment_method_id,exchange_price_difference=delta)
    return _public(o)

def transfer_to_human_agents(db, now, summary: str):
    """Transfer an unsupported issue to a human agent without database mutation.

    Args:
        summary: Factual summary of the customer's issue.
    """
    return 'Transfer successful'

def make_tools():
    return {
        'calculate': calculate,
        'find_user_id_by_email': find_user_id_by_email,
        'find_user_id_by_name_zip': find_user_id_by_name_zip,
        'get_user_details': get_user_details,
        'get_order_details': get_order_details,
        'get_product_details': get_product_details,
        'get_item_details': get_item_details,
        'list_all_product_types': list_all_product_types,
        'transfer_to_human_agents': transfer_to_human_agents,
        'cancel_pending_order': cancel_pending_order,
        'modify_pending_order_address': modify_pending_order_address,
        'modify_pending_order_items': modify_pending_order_items,
        'modify_pending_order_payment': modify_pending_order_payment,
        'return_delivered_order_items': return_delivered_order_items,
        'exchange_delivered_order_items': exchange_delivered_order_items,
        'modify_user_address': modify_user_address,
    }

def approval_key(tool, args, facts):
    return 'approve_'+PROCESSES[tool]+'_'+args.get('order_id',args.get('user_id',''))

def _lookup_facts(record, fields, kind):
    selected=fields or {'profile':['name.first_name','name.last_name','email'],'order':['status'],'product':['name'],'item':['available','price'],'product_types':['product_name']}[kind]
    values=[]
    if kind=='product_types':
        pairs=[('product_name',name) for name in sorted(record)]
    else:
        pairs=[]
        for field in selected:
            value=record
            for component in field.split('.'):
                if not isinstance(value,dict) or component not in value: raise BusinessError('record_not_found')
                value=value[component]
            for scalar in value if isinstance(value,list) else [value]: pairs.append((field,scalar))
    for field,value in pairs:
        item={'field':field,'string_value':None,'number_value':None,'boolean_value':None}
        if isinstance(value,bool): item['boolean_value']=value
        elif isinstance(value,(int,float)): item['number_value']=value
        elif isinstance(value,str): item['string_value']=value
        elif value is not None: raise BusinessError('tool_error')
        values.append(item)
    return values

def reference(api):
    goals=api.ask('goals')
    answer={}
    uid=None
    attempted=False
    for index,goal in enumerate(goals,1):
        process=goal['process'];p=deepcopy(goal['params'])
        result={'status':'refused','reason':None}
        if process in ('pending_items_change','delivered_exchange_request'): result['price_difference']=None
        if process=='delivered_return_request': result['instructions_pending']=False
        if process=='account_lookup': result['facts']=[]
        if process=='handover': result['message']=''
        answer['goal_'+str(index)]=result
        if not attempted:
            attempted=True
            try: uid=api.call('find_user_id_by_email',email=api.ask('email'))
            except (KeyError,BusinessError):
                try: uid=api.call('find_user_id_by_name_zip',**{k:api.ask(k) for k in ('first_name','last_name','zip')})
                except (KeyError,BusinessError): pass
        if uid is None:
            result['reason']='identity_not_verified';continue
        try:
            user=api.call('get_user_details',user_id=uid)
            if process=='handover':
                api.call('transfer_to_human_agents',summary=p['summary'])
                result.update(status='transferred',message='YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.');continue
            if process=='account_lookup':
                kind=p['record_type'];rid=p['record_id']
                if kind=='profile':
                    if rid and rid!=uid: raise BusinessError('other_customer')
                    record=user
                elif kind=='order':
                    if rid not in user['orders']:
                        raise BusinessError('record_not_found')
                    record=api.call('get_order_details',order_id=rid)
                elif kind=='product_types': record=json.loads(api.call('list_all_product_types'))
                elif kind=='product': record=api.call('get_product_details',product_id=rid)
                else: record=api.call('get_item_details',item_id=rid)
                selector=p.get('selector','')
                if selector and kind=='product_types':
                    if selector not in record: raise BusinessError('record_not_found')
                    record={'name':selector,'product_id':record[selector]};kind='product'
                elif selector:
                    for component in selector.split('.'):
                        if isinstance(record,list) and component.isdigit() and int(component)<len(record): record=record[int(component)]
                        elif isinstance(record,dict) and component in record: record=record[component]
                        else: raise BusinessError('record_not_found')
                result.update(status='answered',facts=_lookup_facts(record,p['fields'],kind));continue
            tool=next(t for t,v in PROCESSES.items() if v==process)
            if process=='default_address_change':
                if p['user_id']!=uid: raise BusinessError('other_customer')
                before=user
            else:
                # Ownership is checked before exposing an unrelated order's details.
                if p['order_id'] not in user['orders']: raise BusinessError('record_not_found')
                before=api.call('get_order_details',order_id=p['order_id'])
                if before['user_id']!=uid: raise BusinessError('other_customer')
                if process in ('delivered_return_request','delivered_exchange_request'):
                    if before['status']!='delivered': raise BusinessError('not_delivered')
                else:
                    if before['status']=='pending (item modified)': raise BusinessError('item_edit_lock')
                    if before['status']!='pending': raise BusinessError('not_pending')
            db={'users':[deepcopy(user)],'orders':[deepcopy(before)] if 'order_id' in before else [],'products':[]}
            delta=None;variants=[]
            expected=deepcopy(before)
            if process=='cancel':
                if p['reason'] not in ('no longer needed','ordered by mistake'): raise BusinessError('invalid_cancel_reason')
                expected['status']='cancelled';expected['cancel_reason']=p['reason']
                paid={}
                for payment in before['payment_history']:
                    mid=payment['payment_method_id'];paid[mid]=round(paid.get(mid,0)+payment['amount']*(1 if payment['transaction_type']=='payment' else -1),2)
                debt=round(-sum(min(v,0) for v in paid.values()),2)
                if round(sum(paid.values()),2)<0: raise BusinessError('invalid_payment_history')
                for mid,net in paid.items():
                    amount=round(max(0,net-debt),2)
                    debt=round(max(0,debt-max(net,0)),2)
                    if amount:
                        expected['payment_history'].append(dict(transaction_type='refund',amount=amount,payment_method_id=mid))
                        _balance(_method(db,before,mid),amount)
            elif process in ('shipping_address_change','default_address_change'):
                expected['address']={k:p[k] for k in ADDRESS}
            elif process in ('pending_items_change','delivered_exchange_request'):
                _selection(before,p['item_ids'])
                if len(p['item_ids'])!=len(p['new_item_ids']): raise BusinessError('list_length_mismatch')
                for old,new in zip(p['item_ids'],p['new_item_ids']):
                    if old==new: raise BusinessError('same_item')
                    item=next(x for x in before['items'] if x['item_id']==old)
                    product=api.call('get_product_details',product_id=item['product_id'])
                    db['products'].append(product)
                    if new not in product['variants']: raise BusinessError('wrong_product')
                    if not product['variants'][new]['available']: raise BusinessError('unavailable')
                delta,variants=_variants(db,before,p['item_ids'],p['new_item_ids'],process=='delivered_exchange_request')
                method=_method(db,before,p['payment_method_id']);_enough(method,delta)
                if process=='pending_items_change':
                    available=list(enumerate(expected['items']));selected=[]
                    for old,new,variant in zip(p['item_ids'],p['new_item_ids'],variants):
                        index,item=next((i,x) for i,x in available if x['item_id']==old)
                        available=[(i,x) for i,x in available if i!=index];selected.append((item,new,variant))
                    for item,new,variant in selected:
                        item.update(item_id=new,price=variant['price'],options=deepcopy(variant['options']))
                    expected['status']='pending (item modified)'
                    expected['payment_history'].append(dict(transaction_type='payment' if delta>0 else 'refund',amount=abs(delta),payment_method_id=p['payment_method_id']))
                    _balance(method,-delta)
                else:
                    expected.update(status='exchange requested',exchange_items=sorted(p['item_ids']),exchange_new_items=sorted(p['new_item_ids']),exchange_payment_method_id=p['payment_method_id'],exchange_price_difference=delta)
            elif process=='payment_method_change':
                method=_method(db,before,p['payment_method_id'])
                history=before['payment_history']
                if len(history)!=1 or history[0]['transaction_type']!='payment': raise BusinessError('payment_history_not_single_payment')
                original=history[0]
                if original['payment_method_id']==p['payment_method_id']: raise BusinessError('same_payment_method')
                _enough(method,original['amount'])
                expected['payment_history'].extend([dict(transaction_type='payment',amount=original['amount'],payment_method_id=p['payment_method_id']),dict(transaction_type='refund',amount=original['amount'],payment_method_id=original['payment_method_id'])])
                _balance(method,-original['amount']);_balance(_method(db,before,original['payment_method_id']),original['amount'])
            elif process=='delivered_return_request':
                method=_method(db,before,p['payment_method_id'])
                if method['source']!='gift_card' and p['payment_method_id']!=before['payment_history'][0]['payment_method_id']: raise BusinessError('invalid_refund_destination')
                _selection(before,p['item_ids'])
                expected.update(status='return requested',return_items=sorted(p['item_ids']),return_payment_method_id=p['payment_method_id'])
            if 'item_ids' in p and not p['item_ids']: raise BusinessError('items_not_found')
            topic='approve_'+process+'_'+p.get('order_id',p.get('user_id',''))
            proposal={'tool':tool,'args':deepcopy(p)}
            try: token=api.confirm(topic,proposal)
            except KeyError:
                result['reason']='approval_missing';continue
            if token is None:
                try: api.ask(topic)
                except KeyError:
                    result['reason']='approval_missing';continue
                result['reason']='approval_declined';continue
            try: api.call(tool,consent=token,**p)
            except BusinessError as exc:
                result.update(status='failed',reason=str(exc) if str(exc) in REASONS else 'tool_error');continue
            after=api.call('get_user_details',user_id=uid) if process=='default_address_change' else api.call('get_order_details',order_id=p['order_id'])
            after_user=after if process=='default_address_change' else api.call('get_user_details',user_id=uid)
            expected_user=expected if process=='default_address_change' else db['users'][0]
            if after!=expected or after_user!=expected_user:
                result.update(status='failed',reason='compatibility_mismatch');continue
            result['status']='requested' if process in ('delivered_return_request','delivered_exchange_request') else 'completed'
            if 'price_difference' in result: result['price_difference']=delta
            if 'instructions_pending' in result: result['instructions_pending']=True
        except BusinessError as exc:
            result['reason']=str(exc) if str(exc) in REASONS else 'tool_error'
    return answer

REASONS=set('identity_not_verified other_customer record_not_found not_pending item_edit_lock not_delivered invalid_cancel_reason items_not_found list_length_mismatch same_item wrong_product unavailable payment_not_owned invalid_refund_destination insufficient_gift_balance payment_history_not_single_payment same_payment_method approval_missing approval_declined outside_scope contract_ambiguity compatibility_mismatch tool_error'.split())
