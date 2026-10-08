import pytest
from tests.runtime.test_reward_state_snapshots import runtime,context
from env_factory.sandbox_runtime import RewardStateSnapshots,DeclarativeMetricEvaluator as E


def make_snapshots(tmp_path, script, three_hops=False):
    store,data,_=runtime(tmp_path)
    tools=[{'tool_name':name,'operation':'select'} for name in ['lookup','middle','read_stock']]
    expression={'tool_result':{'name':'read_stock','path':'$.records[0].available'}}
    root={'tool_name':'lookup','arguments':{'region':'A'},'by_user_script':{
        'script-1':[{'tool_name':'lookup','arguments':{'region':'A'}}],
        'script-2':[{'tool_name':'lookup','arguments':{'region':'B'}}]}}
    child={'tool_name':'read_stock','arguments':{'id':{'$ref':'selected'}},'captures':[
        {'name':'selected','tool_name':'middle' if three_hops else 'lookup','path':'$.records[0].id'}]}
    expected=[root,child]
    if three_hops:
        expected.append({'tool_name':'middle','arguments':{'id':{'$ref':'upstream'}},'captures':[
            {'name':'upstream','tool_name':'lookup','path':'$.records[0].id'}]})
    bindings=[{'source':'trajectory','operator':'contains_tool_call','expected':value} for value in expected]
    snapshots=RewardStateSnapshots(data,tools,[expression,*bindings])
    store.event('user_protocol',{'script_id':script},{})
    def call(name,args,record):
        result={'records':[record]}
        snapshots.capture(name,result,args)
        store.event('tool_call',{'tool_name':name,'arguments':args},result)
    return call,lambda:E._value_expression(expression,context(store,data))


@pytest.mark.parametrize('script,chosen,other',[('script-1','A','B'),('script-2','B','A')])
@pytest.mark.parametrize('three_hops',[False,True])
def test_exploration_cannot_bind_consumer_to_wrong_producer(tmp_path,script,chosen,other,three_hops):
    call,value=make_snapshots(tmp_path,script,three_hops)
    call('lookup',{'region':chosen},{'id':1})
    call('lookup',{'region':other},{'id':2})
    if three_hops:call('middle',{'id':2},{'id':22})
    call('read_stock',{'id':22 if three_hops else 2},{'available':99})
    assert value() is None
    if three_hops:call('middle',{'id':1},{'id':11})
    call('read_stock',{'id':11 if three_hops else 1},{'available':20})
    assert value()==20
    call('read_stock',{'id':22 if three_hops else 2},{'available':88})
    assert value()==20
