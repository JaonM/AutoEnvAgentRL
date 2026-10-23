"""Causal business progression, isolation, idempotency and independent certification."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from env_factory import lifecycle
from env_factory.core import spec, reward
from env_factory.core.sandbox import load_module
from env_factory.runtime import Episode, BusinessError, ToolError, run_reference, write
from env_factory.serving.server import Service
from env_factory.certify import gates, bundle

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples' / 'returns_lifecycle'


@pytest.fixture
def setup():
    return (spec.load(EXAMPLE / 'request.json'), load_module(EXAMPLE / 'env.py', role='env'),
            load_module(EXAMPLE / 'verifier.py', role='verifier'), json.loads((EXAMPLE / 'answer_schema.json').read_text()))


def episode(setup, **changes):
    request, env, _, _ = setup
    request = copy.deepcopy(request)
    request['scenario']['parameters'].update(changes)
    return Episode(env.make_tools(), request['tables'], request['user_facts'],
                   approve=lambda *_: True, **lifecycle.options(request))


def apply(api, name='request_return', key='K1', entity='R1'):
    return api.call(name, return_id=entity, request_id=key)


def status(api):
    return api.db['returns'][0]['status']


def test_no_action_no_event_and_words_are_not_shipping(setup):
    api = episode(setup)
    api.trace.append({'kind': 'chat', 'reply': '已经发货，仓库已经收到了', 'time': 0})
    api.wait(100)
    assert status(api) == 'none'
    assert not api.lifecycle.pending and not api.lifecycle.log
    with pytest.raises(BusinessError, match='precondition'):
        apply(api, 'request_refund')


def test_causal_progress_and_actual_timestamps_on_large_wait(setup):
    api = episode(setup)
    apply(api)
    assert status(api) == 'requested'
    api.wait(40)
    assert status(api) == 'accepted'
    entries = [e for e in api.lifecycle.log if e['kind'] == 'transition']
    assert [(e['transition'], e['time']) for e in entries] == [('apply',0),('handover',10),('receive',30),('accept',40)]
    apply(api,'request_refund',key='PAY')
    assert api.db['returns'][0]['refund'] == 0
    api.wait(10)
    assert status(api) == 'refunded' and api.db['returns'][0]['refund'] == 1000
    assert api.db['returns'][1] == setup[0]['tables']['returns'][1]


def test_cancel_invalidates_pending_event_without_resurrection(setup):
    api = episode(setup)
    apply(api); apply(api,'cancel_return',key='C')
    api.wait(100)
    assert status(api) == 'cancelled'
    assert any(e['kind'] == 'event_cancelled' for e in api.lifecycle.log)
    assert not api.lifecycle.pending


@pytest.mark.parametrize('params,target', [({'will_ship':False},'requested'), ({'inspection_pass':False},'rejected')])
def test_external_scenarios_do_not_skip_preconditions(setup,params,target):
    api = episode(setup,**params)
    apply(api);api.wait(100)
    assert status(api) == target
    before = api.result()
    with pytest.raises(BusinessError):apply(api,'request_refund',key='PAY')
    assert api.result()['lifecycle'] == before['lifecycle']
    assert api.db == before['final_state']


def test_retries_return_original_receipt_and_never_emit_twice(setup):
    api = episode(setup)
    original=apply(api);api.wait(40)
    snapshot=api.lifecycle.snapshot()
    assert apply(api)==original
    assert status(api)=='accepted' and snapshot==api.lifecycle.snapshot()
    with pytest.raises(BusinessError,match='different operation'):apply(api,entity='R2')
    with pytest.raises(BusinessError,match='precondition'):apply(api,key='other')


def test_missing_consent_cannot_schedule_an_event(setup):
    request,env,_,_=setup
    api=Episode(env.make_tools(),request['tables'],request['user_facts'],**lifecycle.options(request))
    with pytest.raises(BusinessError):apply(api)
    assert not api.lifecycle.pending and status(api)=='none'
    assert api.safety_violations==1


def test_receipt_failure_rolls_back_database_outbox_and_idempotency(setup):
    api=episode(setup)
    @write
    def bad(db,now,return_id:str,request_id:str):
        raise RuntimeError('receipt crashed')
    api.tools={**api.tools,'request_return':bad}
    before=api.result()
    with pytest.raises(ToolError):apply(api)
    assert api.db==before['final_state'] and api.lifecycle.snapshot()==before['lifecycle']


def test_generated_tool_cannot_write_managed_rows_directly(setup):
    api=episode(setup)
    @write
    def bypass(db,now):
        db['returns'][0]['status']='refunded'
        return {}
    api.tools={**api.tools,'bypass':bypass}
    with pytest.raises(ToolError,match='managed table'):api.call('bypass')
    assert status(api)=='none' and not api.lifecycle.log


def test_invariant_failure_rolls_back_effects_and_emissions(setup):
    request,env,_,_=setup
    request=copy.deepcopy(request)
    request['lifecycle_schema']['transitions']['apply']['effects']=[{'table':'returns','entity':{'path':'entity'},'set':{'refund':1000}}]
    api=Episode(env.make_tools(),request['tables'],{},approve=lambda *_:True,**lifecycle.options(request))
    with pytest.raises(ToolError,match='invariant'):apply(api)
    assert status(api)=='none' and api.db['returns'][0]['refund']==0 and not api.lifecycle.pending


def test_wait_is_atomic_on_scheduler_fault_and_cycle_is_bounded(setup):
    request,env,_,_=setup
    request=copy.deepcopy(request)
    request['lifecycle_schema']['max_events']=3
    rule=request['lifecycle_schema']['transitions']['handover']
    rule['from']=['requested','in_transit']
    rule['emit']=[{'transition':'handover','entity':{'path':'entity'},'delay':0}]
    api=Episode(env.make_tools(),request['tables'],{},approve=lambda *_:True,**lifecycle.options(request))
    apply(api);before=api.result()
    with pytest.raises(ToolError,match='budget'):api.wait(20)
    assert api.db==before['final_state'] and api.lifecycle.snapshot()==before['lifecycle']
    assert api.tool_errors==1 and api.now==before['now']


def test_later_initial_stage_is_built_from_legal_history(setup):
    request,env,_,_=setup
    request=copy.deepcopy(request)
    request['scenario']['history']=[{'offset':-35,'transition':'apply','entity':'R1','args':{},'key':'old-request'}]
    api=Episode(env.make_tools(),request['tables'],{},**lifecycle.options(request))
    assert status(api)=='received'
    assert api.lifecycle.pending[0]['at']==5
    api.wait(5);assert status(api)=='accepted'
    request['scenario']['history'][0]['transition']='settle'
    with pytest.raises(BusinessError):Episode(env.make_tools(),request['tables'],{},**lifecycle.options(request))


def test_calendar_clock_advances_only_for_opted_in_lifecycle(setup):
    request,env,_,_=setup
    api=Episode(env.make_tools(),request['tables'],{},start='2026-10-22T12:00',approve=lambda *_:True,**lifecycle.options(request))
    apply(api);api.wait(40)
    assert api.now=='2026-10-22T12:00:40' and status(api)=='accepted'


@pytest.mark.parametrize('wait',[float('nan'),float('inf'),-1,True])
def test_invalid_wait_rejected(setup,wait):
    with pytest.raises(ValueError):episode(setup).wait(wait)


def test_reset_fork_replay_preserve_hidden_queue_and_receipts(setup):
    request,env,verifier,schema=setup
    service=Service(env,verifier,schema,request,perturbed=0)
    identifier=service.reset()['episode']
    args={'return_id':'R1','request_id':'K1'}
    actions=[{'type':'confirm','key':'approve_return','proposal':{'tool':'request_return','args':args}},
             {'type':'tool','name':'request_return','args':{**args,'consent':'consent-1'}},
             {'type':'wait','seconds':15}]
    for action in actions:assert service.step(identifier,action)['ok']
    fork=service.fork(identifier)['episode']
    restored=service.restore('original',[{'action':a} for a in actions])['episode']
    original=service.episodes[identifier]['episode']
    for other in (fork,restored):assert service.episodes[other]['episode'].result()==original.result()
    service.step(fork,{'type':'wait','seconds':25})
    assert status(original)=='in_transit' and status(service.episodes[fork]['episode'])=='accepted'
    fresh=service.reset()['episode']
    assert status(service.episodes[fresh]['episode'])=='none' and not service.episodes[fresh]['episode'].lifecycle.pending
    assert request['tables']['returns'][0]['status']=='none'


def test_same_episode_concurrent_retries_are_serialized(setup):
    request,env,verifier,schema=setup
    service=Service(env,verifier,schema,request,perturbed=0)
    identifier=service.reset()['episode']
    args={'return_id':'R1','request_id':'K1'}
    actions=[]
    for _ in range(8):
        token=service.step(identifier,{'type':'confirm','key':'approve_return','proposal':{'tool':'request_return','args':args}})['result']
        actions.append({'type':'tool','name':'request_return','args':{**args,'consent':token}})
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(lambda a:service.step(identifier,a),actions))
    assert all(r['ok'] for r in results)
    api=service.episodes[identifier]['episode']
    assert len(api.lifecycle.pending)==1 and len(api.lifecycle.receipts)==1


def test_world_and_scenario_validation_prevents_episode_rule_override(setup):
    request=copy.deepcopy(setup[0])
    request['scenario']['transitions']={}
    assert spec.validate(request)
    request=copy.deepcopy(setup[0]);request['tables']['returns'][0]['status']='received'
    assert any('history' in p for p in spec.validate(request))
    request=copy.deepcopy(setup[0]);request['lifecycle_schema']['tools']['request_return']['transition']='receive'
    assert spec.validate(request)


def test_assets_hash_roundtrip_and_case_gate(setup,tmp_path):
    request,env,verifier,_=setup
    document=spec.write(request,tmp_path/'request.json')
    assert spec.load(tmp_path/'request.json')==request
    assert {'lifecycle_schema','scenario','lifecycle_contract'}<=set(document['request_assets']['files'])
    assert lifecycle.qualify(env,verifier,request)['passed']
    (tmp_path/'scenario.json').write_text('{}')
    with pytest.raises(spec.RequestError,match='hash mismatch'):spec.load(tmp_path/'request.json')


def test_independent_verifier_rejects_teleportation_and_grade_is_zero(setup):
    request,env,verifier,schema=setup
    result=run_reference(env,request['tables'],request['user_facts'],request=request)
    assert not lifecycle.audit(verifier,request['tables'],request['user_facts'],0,result,request)
    result['lifecycle']['log']=[e for e in result['lifecycle']['log'] if e.get('transition')!='receive']
    result['lifecycle_errors']=lifecycle.audit(verifier,request['tables'],request['user_facts'],0,result,request)
    expected=lifecycle.expected(verifier,request['tables'],request['user_facts'],0,request)
    grade=reward.grade(result,expected,schema,initial=request['tables'],reference_calls=result['tool_calls'])
    assert grade['state_ok'] and not grade['success'] and grade['reward']==0


def test_full_gates_export_http_and_reload(setup,tmp_path):
    request,env,verifier,schema=setup
    report=gates.qualify(env,verifier,schema,request,perturbed=20)
    assert report['passed'],report
    bundle.export(EXAMPLE,EXAMPLE/'request.json',report,tmp_path/'bundle',status='ready')
    loaded=bundle.load_bundle(tmp_path/'bundle')
    assert loaded['request']['lifecycle_schema']==request['lifecycle_schema']
    assert loaded['gates']['lifecycle']['passed']


def test_two_business_objects_compete_for_one_resource_atomically(setup):
    request,env,_,_=setup
    request=copy.deepcopy(request)
    request['tables']['stock']=[{'id':'SKU','status':'active','units':1}]
    schema=request['lifecycle_schema']
    schema['machines']['stock']={'table':'stock','field':'status','initial':'active','states':['active']}
    rule=schema['transitions']['apply']
    rule['guard']={'op':'ge','args':[{'path':'db.stock.0.units'},1]}
    rule['effects']=[{'table':'stock','entity':'SKU','set':{'units':{'op':'sub','args':[{'path':'db.stock.0.units'},1]}}}]
    api=Episode(env.make_tools(),request['tables'],{},approve=lambda *_:True,**lifecycle.options(request))
    apply(api,entity='R1')
    before=api.result()
    with pytest.raises(BusinessError):apply(api,key='K2',entity='R2')
    assert api.db['stock'][0]['units']==0
    assert api.db['returns'][1]['status']=='none'
    assert api.lifecycle.snapshot()==before['lifecycle']
    # The failed second request emitted no customer/carrier event for R2.
    assert all(e['entity']=='R1' for e in api.lifecycle.pending)


def test_contract_coverage_cannot_be_claimed_without_execution(setup):
    request,env,verifier,_=setup
    request=copy.deepcopy(request)
    request['lifecycle_contract']['cases']=request['lifecycle_contract']['cases'][:1]
    request['lifecycle_contract']['cases'][0]['transitions']=list(request['lifecycle_schema']['transitions'])
    report=lifecycle.qualify(env,verifier,request)
    assert not report['passed'] and any('coverage' in p for p in report['problems'])


def test_world_task_composition_selects_scenario_without_rewriting_machine(setup):
    from env_factory.production.world import compose
    world=copy.deepcopy(setup[0]);world['request_kind']='world'
    scenario=copy.deepcopy(world['scenario']);scenario['parameters']['will_ship']=False
    task={'title':'等待客户寄回','user_facts':world['user_facts'], 'user_scene':{'goal':'申请退货'},'scenario':scenario}
    composed=compose(world,task,'T2')
    assert composed['lifecycle_schema']==world['lifecycle_schema']
    assert composed['scenario']['parameters']['will_ship'] is False
    assert world['scenario']['parameters']['will_ship'] is True
    task.pop('scenario')
    with pytest.raises(ValueError,match='scenario'):compose(world,task,'T3')


def test_unknown_initial_history_is_rejected_before_generation(setup):
    request=copy.deepcopy(setup[0])
    request['scenario']['history']=[{'offset':0,'transition':'receive','entity':'R1','args':{},'key':'bad'}]
    assert any('illegal lifecycle initial history' in p for p in spec.validate(request))


def test_natural_chat_claims_do_not_create_shipping_events(setup):
    class User:
        def chat(self,*args,**kwargs):
            return {'content':'我已经寄回了，你直接给我退款吧。'}
    request,env,verifier,schema=setup
    service=Service(env,verifier,schema,request,perturbed=0,user_mode='natural',user_client=User(),audit=False)
    identifier=service.reset()['episode']
    service.step(identifier,{'type':'chat','message':'您是否已经寄回？'})
    api=service.episodes[identifier]['episode']
    assert status(api)=='none' and not api.lifecycle.pending


def test_request_only_completion_does_not_drain_pending_fulfillment(setup):
    request,env,verifier,schema=setup
    request=copy.deepcopy(request)
    request['scenario']['scope']=request['user_facts']['scope']='request_only'
    result=run_reference(env,request['tables'],request['user_facts'],request=request)
    assert result['answer']=={'status':'requested','refund':0} and result['lifecycle']['pending']
    expected=lifecycle.expected(verifier,request['tables'],request['user_facts'],0,request)
    assert reward.grade(result,expected,schema,initial=request['tables'],reference_calls=result['tool_calls'])['success']


def test_continuation_identity_includes_platform_even_after_recertification(monkeypatch):
    from rl.environments.sandbox import bundle_identity
    manifest={'files':{'env.py':'same-content'},'platform_identity':'platform-A'}
    monkeypatch.setattr(bundle,'load_bundle',lambda root:{'manifest':manifest})
    before=bundle_identity('unused')
    manifest['platform_identity']='platform-B'
    assert bundle_identity('unused')!=before


def test_calendar_overflow_cannot_partially_commit_due_events(setup):
    request,env,_,_=setup
    api=Episode(env.make_tools(),request['tables'],{},start='2026-10-22T12:00',approve=lambda *_:True,**lifecycle.options(request))
    apply(api);before=api.result()
    with pytest.raises(ToolError,match='range'):api.wait(1e300)
    assert api.db==before['final_state'] and api.lifecycle.snapshot()==before['lifecycle']


def test_author_and_verifier_selfchecks_accept_only_legal_initial_variants():
    from env_factory import checks
    assert checks.check_env(EXAMPLE)[0]==[]
    assert checks.check_verifier(EXAMPLE)[0]==[]
