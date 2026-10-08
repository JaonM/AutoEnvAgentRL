"""Exercise the materialized app in-process or via an HTTP URL; no model calls."""
import argparse
import json
import os
import urllib.request
import urllib.error


def exercise(call, token):
    headers={'Authorization':'Bearer '+token}
    episode=call('POST','/v1/reset',{'seed':17,'reward_mode':'episode_end'},headers)
    assert episode[0]==200,episode
    headers['X-Episode-ID']=episode[1]['episode_id']
    conversation=[];trace=[]
    def tool(name,args,key=None,expected=200):
        h={'X-Episode-ID':headers['X-Episode-ID']}
        if key:h['Idempotency-Key']=key
        status,body,_=call('POST','/v1/tools/'+name,args,h)
        trace.append({'tool':name,'status':status,'result':body})
        assert status==expected,(name,status,body)
        return body
    def user(text,stage=None,branch=None):
        assert call('POST','/v1/agent_response',{'content':text},headers)[0]==200
        conversation.append({'role':'assistant','content':text})
        status,body,_=call('POST','/v1/user_simulator',{'messages':conversation},headers)
        assert status==200,body
        assert body.get('interaction_stage')==stage,body
        assert body.get('interaction_branch')==branch,body
        conversation.append({'role':'user','content':body['user_query']})
        trace.append({'user':body})
    stock=tool('read_inventory',{'id':1})
    assert stock['records'][0]['available']==40
    tool('reserve',{'id':1,'status':'reserved','available':20,'reserved':20},'reserve',409)
    user('Please approve split orders.',branch='reject_split')
    user('Please approve one order.',stage='approval')
    tool('reserve',{'id':1,'status':'reserved','available':20,'reserved':20},'reserve')
    tool('place_order',{'id':1,'status':'placed'},'order',504)
    assert tool('read_orders',{'id':1})['records'][0]['status']=='placed'
    tool('place_order',{'id':1,'status':'placed'},'order')
    tool('pay',{'id':1,'status':'paid'},'payment',503)
    assert tool('read_payments',{'id':1})['records'][0]['status']=='unpaid'
    tool('pay',{'id':1,'status':'paid'},'payment')
    user('The payment is confirmed. Any changes?',stage='cancel_request')
    tool('place_order',{'id':1,'status':'placed'},'unauthorized-new-order',403)
    tool('cancel_order',{'id':1,'status':'cancelled'},'cancel')
    tool('refund',{'id':1,'status':'refunded'},'refund')
    # Partial compensation must not earn full reward.
    assert call('GET','/v1/reward',None,headers)[1]['reward']==0
    tool('release',{'id':1,'status':'available','available':40,'reserved':0},'release')
    call('POST','/v1/agent_response',{'content':'Cancelled order, refunded payment, and released all 20 reserved units.'},headers)
    conversation.append({'role':'assistant','content':'Cancelled order, refunded payment, and released all 20 reserved units.'})
    assert call('POST','/v1/user_simulator',{'messages':conversation},headers)[1]['should_end']
    status,reward,_=call('GET','/v1/reward',None,headers)
    assert status==200 and reward['reward']==1,reward
    replay=call('GET','/v1/replay',None,headers)[1]
    transitions=[e for e in replay['events'] if e['event']=='business_transition']
    assert len(transitions)==6,len(transitions)
    return {'reward':reward,'trace':trace,'business_transition_count':len(transitions),'replay':replay}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',help='HTTP sandbox address; otherwise import local app.py')
    parser.add_argument('--report',default='realism_report.json')
    args=parser.parse_args()
    token=os.environ.get('SANDBOX_TRAINER_API_KEY','realism-local-test')
    if args.url:
        def call(method,path,body,headers):
            request=urllib.request.Request(args.url.rstrip('/')+path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json',**headers},method=method)
            try:response=urllib.request.urlopen(request,timeout=10)
            except urllib.error.HTTPError as exc:response=exc
            with response:return response.status,json.load(response),dict(response.headers)
    else:
        os.environ['SANDBOX_TRAINER_API_KEY']=token
        from app import create_app
        call=create_app().handle
    report=exercise(call,token)
    with open(args.report,'w') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'passed':True,'reward':report['reward'],'business_transition_count':report['business_transition_count']}))
