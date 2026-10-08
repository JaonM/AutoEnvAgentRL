import copy
from env_factory.generation.review_replay import observe_counterexample


def test_current_row_reward_tracks_mutation_without_overriding_verdict():
    bundle={'source':{'tables':[{'table_name':'layouts','columns':[
        {'name':'name','type':'text'},{'name':'eligible','type':'boolean'}],
        'rows':[{'name':'old','eligible':True},{'name':'new','eligible':False}]}]},
        'compiled_reward_rules':[{'metric_id':'result','operator':'value_targets','expected':{'targets':[
            {'key':'name','expression':{'lookup':{'table':'layouts','field':'name','where':{'eligible':True}}}}]}}]}
    finding={'counterexample':{'basis':'business_data','script_assumptions':[],'public_paths':[],
        'data_mutations':[{'table':'layouts','row_index':0,'column':'eligible','value_json':'false'},
                          {'table':'layouts','row_index':1,'column':'eligible','value_json':'true'}]}}
    before=copy.deepcopy(bundle)
    result=observe_counterexample(finding,bundle)
    assert result['targets']==[{'metric_id':'result','field':'name','original_target':'old','mutated_target':'new','resolved':True}]
    assert result['verdict_override'] is False
    assert bundle==before
    finding['counterexample']['data_mutations'].pop()
    assert observe_counterexample(finding,bundle)['targets'][0]['resolved'] is False
    finding['counterexample']['data_mutations'][0]['value_json']='42'
    assert observe_counterexample(finding,bundle)['status']=='invalid_counterexample'


def test_baseline_capacity_lookup_excludes_small_venue():
    from env_factory.generation.review_replay import baseline_reward_observations
    bundle={'source':{'tables':[{'table_name':'venues','rows':[
        {'name':'large','capacity':60},{'name':'small','capacity':32}]}]},
        'compiled_reward_rules':[{'metric_id':'venue','operator':'value_targets','expected':{'targets':[
            {'key':'name','expression':{'lookup':{'table':'venues','field':'name','where':{'capacity':{'gte':40}}}}}]}}]}
    observed=baseline_reward_observations(bundle)['targets'][0]
    assert observed['value']=='large'
    assert observed['lookups'][0]['matched_row_count']==1
    assert observed['lookups'][0]['matched_values']==['large']


def test_review_receives_executed_state_after_absolute_external_change(tmp_path):
    import json
    from tests.generation.test_dynamic_state_goal import dynamic_source
    from env_factory.generation.agent_authoring import compile_source
    source, request = dynamic_source()
    source['business_lifecycle'] = {'version': '1.0', 'tools': {}, 'dynamics': {'events': [{
        'id': 'configuration_refresh', 'at_tick': 1, 'changes': [{
            'table': 'zones', 'selector': {'id': 821}, 'values': {'configured_temperature': 31}}]}]}}
    source['scenarios'][0]['steps'][-1]['content'] = '温度：31摄氏度。'
    result = compile_source(source, root=tmp_path, request=request)
    observed = result['generation_pipeline']['verification']['reference_observations'][0]
    assert observed['reward']['reward'] == 1
    assert observed['state_predicates'][0]['expected_values'] == {'temperature': 31}
    assert observed['state_predicates'][0]['actual_rows'][0]['temperature'] == 31
    assert observed['external_changes'][0]['payload']['values'] == {'configured_temperature': 31}
    assert observed['unrelated_state_preserved']
    # Captured at the successful reference, not overwritten by subsequent ablations.
    assert observed['state_predicates'][0]['satisfied']
    json.dumps(observed)


def test_executable_review_evidence_preserves_pending_semantic_verdict(tmp_path):
    from tests.generation.test_dynamic_state_goal import dynamic_source
    from env_factory.generation.agent_authoring import compile_source
    from env_factory.generation.review_replay import replay_review_findings
    source, request = dynamic_source()
    original = compile_source(source, root=tmp_path/'original', request=request)
    report = {'findings': [{'code': 'changed_requirement', 'source_paths': ['tables[0]'],
        'counterexample': {'data_mutations': [{'table': 'zones', 'row_index': 0,
            'column': 'configured_temperature', 'value_json': '31'}]}}]}
    replay_review_findings(report, source=source, artifacts=original, request=request, root=tmp_path/'replay')
    evidence = report['findings'][0]['execution_evidence']
    assert evidence['observation'] in {'mutated_reference_and_ablations_passed', 'mutated_validation_rejected'}
    assert evidence['status'] == 'pending_adjudication'
    assert evidence['verdict_override'] is False
    assert source['tables'][0]['rows'][0]['configured_temperature'] != 31
