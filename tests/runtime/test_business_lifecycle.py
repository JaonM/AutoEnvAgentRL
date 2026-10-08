import json
import pytest
from env_factory.sandbox_runtime import (BusinessLifecycle, ContractToolRegistry,
    EpisodeStore, ManifestDataStore, SandboxError)


def setup(tmp_path, *, wrong=False, dynamics=None, guards=None):
    schema={'columns':[{'name':'id','type':'INTEGER','nullable':False},
                       {'name':'status','type':'TEXT','nullable':False}], 'primary_key':['id']}
    (tmp_path/'schema.json').write_text(json.dumps(schema))
    (tmp_path/'rows.jsonl').write_text(json.dumps({'id':1,'status':'draft'})+'\n')
    store=EpisodeStore(tmp_path/'episodes.sqlite3')
    data=ManifestDataStore({'environment_mode':'stateful','tables':[{'table_name':'orders',
        'schema_file':'schema.json','rows_file':'rows.jsonl'}]},tmp_path,store)
    data.reset()
    config={'version':'1.0','roles':['buyer'],'tools':{'submit':{'roles':['buyer'],
        'requires_grants':['purchase'],'require_idempotency':True,
        'transition':{'table':'orders','selector':{'id':{'argument':'id'}},
                      'column':'status','from':['draft'],'to':'submitted'}}}}
    if dynamics is not None:config['dynamics']=dynamics
    if guards is not None:config['tools']['submit']['guards']=guards
    lifecycle=BusinessLifecycle(config,data,store,['submit'])
    tools=[{'type':'function','function':{'name':'submit','parameters':{'type':'object',
        'properties':{'id':{'type':'integer'}},'required':['id'],'additionalProperties':False}}}]
    def submit(args):
        data.update('orders',{'id':args['id']},{'status':'wrong' if wrong else 'submitted'})
        return {'ok':True}
    registry=ContractToolRegistry(tools,{'submit':submit},event_recorder=store.event,lifecycle=lifecycle)
    return store,data,registry


def test_commit_guard_rejects_changed_state_before_write(tmp_path):
    guard = {'id': 'still_draft', 'table': 'orders', 'selector': {'id': {'argument': 'id'}},
             'conditions': {'status': {'eq': 'draft'}}}
    store,data,registry=setup(tmp_path,guards=[guard],dynamics={'events':[
        {'id':'cancel', 'at_tick':1, 'changes':[{'table':'orders','selector':{'id':1},
                                              'values':{'status':'cancelled'}}]}]})
    store.set_state('business_grants',['purchase'])
    with pytest.raises(SandboxError) as error:
        registry.execute('submit',{'id':1},idem_key='guarded')
    assert error.value.code == 'BUSINESS_PRECONDITION_FAILED'
    assert error.value.status == 409
    assert error.value.details['guard_id'] == 'still_draft'
    assert error.value.details['matched_count'] == 1
    assert error.value.details['failed_fields'] == [
        {'field': 'status', 'operator': 'eq', 'expected': 'draft', 'actual': 'cancelled'}]
    assert data.table('orders')[0]['status'] == 'cancelled'
    assert not [e for e in store.replay()['events'] if e['event']=='business_transition']


def test_guard_missing_target_reports_cardinality_without_inventing_field_values(tmp_path):
    guard = {'id': 'still_draft', 'table': 'orders', 'selector': {'id': {'argument': 'id'}},
             'conditions': {'status': {'eq': 'draft'}}}
    store,data,registry=setup(tmp_path,guards=[guard])
    store.set_state('business_grants',['purchase'])
    with pytest.raises(SandboxError) as error:
        registry.execute('submit',{'id':99},idem_key='missing')
    assert error.value.status == 409
    assert error.value.details['matched_count'] == 0
    assert error.value.details['expected_count'] == 1
    assert error.value.details['failed_fields'] == []
    assert data.table('orders') == [{'id':1,'status':'draft'}]


def test_committed_retry_does_not_recheck_obsolete_guard(tmp_path):
    guard = {'id': 'still_draft', 'table': 'orders', 'selector': {'id': {'argument': 'id'}},
             'conditions': {'status': {'eq': 'draft'}}}
    store,data,registry=setup(tmp_path,guards=[guard])
    store.set_state('business_grants',['purchase'])
    assert registry.execute('submit',{'id':1},idem_key='guarded') == {'ok':True}
    assert registry.execute('submit',{'id':1},idem_key='guarded') == {'ok':True}
    assert len([e for e in store.replay()['events'] if e['event']=='business_transition']) == 1


def test_authorization_transition_and_idempotent_retry(tmp_path):
    store,data,registry=setup(tmp_path)
    with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1},idem_key='a')
    assert error.value.code=='AUTHORIZATION_REQUIRED'
    assert data.table('orders')[0]['status']=='draft'
    store.set_state('business_grants',['purchase'])
    with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1})
    assert error.value.code=='IDEMPOTENCY_REQUIRED'
    assert registry.execute('submit',{'id':1},idem_key='a')=={'ok':True}
    assert registry.execute('submit',{'id':1},idem_key='a')=={'ok':True}
    assert len([e for e in store.replay()['events'] if e['event']=='business_transition'])==1
    with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1},idem_key='b')
    assert error.value.code=='INVALID_TRANSITION'


def test_failed_postcondition_rolls_back_write_and_idempotency(tmp_path):
    store,data,registry=setup(tmp_path,wrong=True)
    store.set_state('business_grants',['purchase'])
    for _ in range(2):
        with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1},idem_key='a')
        assert error.value.code=='TRANSITION_POSTCONDITION_FAILED'
        assert data.table('orders')[0]['status']=='draft'
    assert not [e for e in store.replay()['events'] if e['event']=='business_transition']


def test_response_loss_commits_over_http_and_retry_does_not_repeat(tmp_path):
    from env_factory.sandbox_runtime import SandboxApplication
    store,data,registry=setup(tmp_path,dynamics={'faults':[{'id':'lost','tool':'submit',
        'call_index':1,'phase':'after_commit','status':504}]})
    store.set_state('business_grants',['purchase'])
    app=SandboxApplication(episode_store=store,tool_registry=registry,
        observation=lambda:{},reward=lambda:{},user_turn=lambda messages:{})
    status,body,_=app.handle('POST','/v1/tools/submit',{'id':1},{'Idempotency-Key':'order-1'})
    assert status==504
    assert data.table('orders')[0]['status']=='submitted'
    status,body,_=app.handle('POST','/v1/tools/submit',{'id':1},{'Idempotency-Key':'order-1'})
    assert status==200 and body=={'ok':True}
    assert store.get_state('scenario_dynamics')['tick']==1
    assert len([e for e in store.replay()['events'] if e['event']=='business_transition'])==1


def test_transient_failure_consumed_and_external_update_persists(tmp_path):
    store,data,registry=setup(tmp_path,dynamics={'tick_seconds':10,'faults':[{'id':'busy',
        'tool':'submit','call_index':1,'phase':'before','status':503}],
        'events':[{'id':'cancel','at_tick':2,'changes':[{'table':'orders','selector':{'id':1},
                                                    'values':{'status':'cancelled'}}]}]})
    store.set_state('business_grants',['purchase'])
    with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1},idem_key='a')
    assert error.value.code=='TRANSIENT_FAILURE'
    assert data.table('orders')[0]['status']=='draft'
    with pytest.raises(SandboxError) as error:registry.execute('submit',{'id':1},idem_key='a')
    assert error.value.code=='INVALID_TRANSITION'
    assert data.table('orders')[0]['status']=='cancelled'
    assert store.get_state('scenario_dynamics')['elapsed_seconds']==20
    assert store.get_state('scenario_dynamics')['fired']==['cancel']


def test_external_update_is_not_agent_collateral_write():
    from env_factory.sandbox_runtime import BusinessGoalEvaluator as G
    baseline={'orders':[{'id':1,'status':'draft'},{'id':2,'status':'draft'}]}
    event={'event':'external_business_change','payload':{'table':'orders','selector':{'id':2},
           'values':{'status':'cancelled'}},'result':{'updated_count':1}}
    adjusted=G.external_baseline(baseline,[event])
    goal={'row_predicates':[{'table':'orders','where':{'id':1},'values':{'status':'submitted'},'count':1}],
          'table_primary_keys':{'orders':['id']}}
    assert G.preserves_unrelated(goal,adjusted,{'orders':[{'id':1,'status':'submitted'},{'id':2,'status':'cancelled'}]})
    assert not G.preserves_unrelated(goal,adjusted,{'orders':[{'id':1,'status':'submitted'},{'id':2,'status':'submitted'}]})
    assert baseline['orders'][1]['status']=='draft'


def test_user_rejects_partial_order_then_grants_after_replanning(tmp_path):
    from env_factory.sandbox_runtime import ContractUserSimulator
    store,data,_=setup(tmp_path)
    stage={'id':'approve','kind':'execution_confirmation','assistant_contains_all':['approve'],
        'user_reply':'approved','retry_reply':'ask approval','grant_permissions':['purchase'],
        'branches':[{'id':'reject_split','when':[{'source':'assistant_text','path':'$',
            'operator':'contains','expected':'split'}],'user_reply':'No split orders; revise the plan.',
            'outcome':'user_rejection','advance':False,'grant_permissions':[], 'revoke_permissions':['purchase']}]}
    script={'script_id':'s','states':[],'transitions':[],'interaction_protocol':{'stages':[stage]}}
    user=ContractUserSimulator(store,profiles=[{'profile_id':'p'}],scripts=[script])
    state={'script_id':'s','protocol_index':0,'turn_index':0}
    result=user._protocol_turn(script,state,[{'role':'assistant','content':'approve split order?'}])
    assert result['interaction_branch']=='reject_split' and result['interaction_stage'] is None
    assert store.get_state('business_grants')==[]
    result=user._protocol_turn(script,state,[{'role':'assistant','content':'approve single order?'}])
    assert result['interaction_stage']=='approve'
    assert store.get_state('business_grants')==['purchase']


def test_clock_faults_and_grants_survive_reopen_and_isolate_episodes(tmp_path):
    from env_factory.sandbox_runtime import ScenarioDynamics
    store,data,registry=setup(tmp_path,dynamics={'faults':[{'id':'busy','tool':'submit',
        'call_index':1,'phase':'before','status':503}]})
    episode=store.current()
    store.set_state('business_grants',['purchase'])
    with pytest.raises(SandboxError):registry.execute('submit',{'id':1},idem_key='a')
    reopened=EpisodeStore(tmp_path/'episodes.sqlite3')
    with reopened.episode_context(episode.episode_id):
        assert reopened.get_state(ScenarioDynamics.KEY)['calls']['submit']==1
        assert reopened.get_state('business_grants')==['purchase']
    registry.execute('submit',{'id':1},idem_key='a')
    assert data.table('orders')[0]['status']=='submitted'
    other=store.reset(seed=99)
    data.reset(other)
    assert store.get_state('business_grants',[])==[]
    assert store.get_state(ScenarioDynamics.KEY) is None
    assert data.table('orders')[0]['status']=='draft'
    with store.episode_context(episode.episode_id):
        assert data.table('orders')[0]['status']=='submitted'


def test_seeded_schedule_is_reproducible_and_does_not_use_wall_clock(tmp_path):
    store,data,registry=setup(tmp_path,dynamics={'events':[{'id':'later','at_tick':100,
        'jitter_ticks':100,'changes':[{'table':'orders','selector':{'id':1},'values':{'status':'cancelled'}}]}]})
    schedules=[]
    for seed in (17,17,18):
        data.reset(store.reset(seed=seed));store.set_state('business_grants',['purchase'])
        registry.execute('submit',{'id':1},idem_key='a')
        schedules.append(store.get_state('scenario_dynamics')['schedule'])
    assert schedules[0]==schedules[1]
    assert schedules[0]!=schedules[2]
