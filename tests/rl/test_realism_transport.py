import pytest
from rl.tool_transport import is_authored_tool_fault, is_business_conflict


def test_only_declared_faults_are_policy_observations():
    task={'business_lifecycle':{'dynamics':{'faults':[{'tool':'pay','phase':'before','status':503}]}}}
    response={'error':{'code':'TRANSIENT_FAILURE'}}
    assert is_authored_tool_fault(task,'/v1/tools/pay',503,response)
    assert not is_authored_tool_fault({},'/v1/tools/pay',503,response)
    assert not is_authored_tool_fault(task,'/v1/tools/order',503,response)
    assert not is_authored_tool_fault(task,'/v1/reset',503,response)
    assert not is_authored_tool_fault(task,'/v1/tools/pay',503,{'error':{'code':'INTERNAL_ERROR'}})
    assert is_business_conflict('/v1/tools/pay',409,{'error':{'code':'INVALID_TRANSITION'}})
    assert not is_business_conflict('/v1/tools/pay',409,{'error':{'code':'SESSION_EXPIRED'}})


@pytest.mark.parametrize('remote',[False,True])
def test_episode_transport_exposes_authored_fault_but_propagates_real_failure(remote):
    from types import SimpleNamespace
    from rl.environment import SandboxEpisode
    from rl.remote_environment import RemoteSandboxEpisode
    from rl.environment import InfrastructureError
    cls=RemoteSandboxEpisode if remote else SandboxEpisode
    episode=object.__new__(cls)
    episode.task={'business_lifecycle':{'dynamics':{'faults':[{'tool':'pay','phase':'before','status':503}]}}}
    episode.trace=[];episode.headers={}
    result={'error':{'code':'TRANSIENT_FAILURE'}}
    if remote:episode._call=lambda *args:(503,result)
    else:episode.app=SimpleNamespace(handle=lambda *args:(503,result,{}))
    assert episode.request('POST','/v1/tools/pay',{})==(503,result)
    result['error']['code']='INTERNAL_ERROR'
    with pytest.raises(InfrastructureError):episode.request('POST','/v1/tools/pay',{})
