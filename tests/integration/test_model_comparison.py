import copy
import json
from pathlib import Path
from unittest.mock import patch

from scripts.experiments.compare_fixed_repairs import counterexample_check


def test_positive_rename_updates_reference_report_but_preserves_source(tmp_path):
    source={'tables':[{'table_name':'artifacts','rows':[{'artifact_id':'A-103','title':'old title'}]}],
            'reference':{'answer':json.dumps({'artifact':'old title','status':'ready'})},
            'scenarios':[{'kind':'goal_success','steps':[{'operation':'agent_response','content':'Placed old title.'}]}]}
    original=copy.deepcopy(source)
    with patch('scripts.experiments.compare_fixed_repairs.compile_source') as compile:
        result=counterexample_check('guarded_transition',source,{},tmp_path)
    supplied=compile.call_args.args[0]
    title=supplied['tables'][0]['rows'][0]['title']
    assert json.loads(supplied['reference']['answer'])['artifact']==title
    assert supplied['scenarios'][0]['steps'][0]['content']=='Placed '+title+'.'
    assert source==original
    assert result['passed']


def test_positive_control_does_not_turn_reward_failure_into_success(tmp_path):
    source={'tables':[{'table_name':'artifacts','rows':[{'artifact_id':'A-103','title':'old title'}]}],
            'reference':{'answer':'old title'}}
    with patch('scripts.experiments.compare_fixed_repairs.compile_source',side_effect=ValueError('reward remained wrong')):
        assert not counterexample_check('guarded_transition',source,{},tmp_path)['passed']


def test_negative_control_requires_executed_client_rejection(tmp_path):
    source={'tables':[], 'business_lifecycle':{'dynamics':{'events':[{
        'id':'storm_tile_allocation','changes':[{'table':'materials','values':{'available':26}}]}]}}}
    for status in [None,400,409,500]:
        root=tmp_path/str(status)
        def compile(candidate,*,root,**kwargs):
            assert candidate['business_lifecycle']['dynamics']['events'][0]['changes'][0]['values']['available']==10
            (root/'preflight_failure.json').write_text(json.dumps({'details':{'actual_status':status}}))
            raise ValueError('failed trajectory')
        with patch('scripts.experiments.compare_fixed_repairs.compile_source',side_effect=compile):
            report=counterexample_check('reward_goal_drift',source,{},root)
        assert report['passed']==(status in [400,409])


def test_summary_excludes_pending_infrastructure_and_invalidated(tmp_path):
    import pytest
    from scripts.experiments.summarize_fixed_repairs import summarize
    (tmp_path/'plan.json').write_text(json.dumps({'models':['luna','astra'],
        'cases':[{'id':'first'},{'id':'second'}],'scope':'fixed_source_test'}))
    p=tmp_path/'first/luna';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps({'case':'first','model':'luna','passed':True,
        'platform_unchanged':True,'attempts':[{'status':'passed'}],'seconds':10}))
    p=tmp_path/'first/astra';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps({'case':'first','model':'astra','passed':False,
        'platform_unchanged':True,'attempts':[{'status':'infrastructure_error'}],'seconds':10}))
    report=summarize(tmp_path)
    assert not report['complete']
    assert report['models']['luna']['pass_rate_among_evaluable']==1
    assert report['models']['luna']['pending']==1
    assert report['models']['astra']['pass_rate_among_evaluable'] is None
    assert report['paired_cases']==[]
    (tmp_path/'invalidated.json').write_text('{}')
    with pytest.raises(ValueError,match='Invalidated'):
        summarize(tmp_path)


def test_network_timeout_is_not_model_failure_after_unrecovered_disconnect(tmp_path):
    from scripts.experiments.summarize_fixed_repairs import network_interrupted
    p=tmp_path/'events.jsonl'
    events=[{'type':'item.completed'}, {'type':'error','message':'Reconnecting... waiting for network (Connection failed: error sending request)'}]
    p.write_text('\n'.join(json.dumps(e) for e in events))
    assert network_interrupted(p)
    events.append({'type':'item.completed'})
    p.write_text('\n'.join(json.dumps(e) for e in events))
    assert not network_interrupted(p)


def test_selected_case_with_terminal_infrastructure_is_not_complete(tmp_path):
    from scripts.experiments.summarize_fixed_repairs import summarize
    (tmp_path/'plan.json').write_text(json.dumps({'models':['luna','astra'],'cases':[{'id':'one'},{'id':'excluded'}],
        'selected_cases':['one'],'scope':'fixed_source_test'}))
    for model in ['luna','astra']:
        p=tmp_path/'one'/model;p.mkdir(parents=True)
        (p/'result.json').write_text(json.dumps({'case':'one','model':model,'passed':False,'platform_unchanged':True,
            'attempts':[{'status':'infrastructure_error'}]}))
    result=summarize(tmp_path)
    assert result['all_trials_terminal']
    assert not result['complete']
    assert result['models']['luna']['planned']==1
    assert result['models']['luna']['pending']==0
