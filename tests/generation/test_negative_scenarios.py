import copy
import pytest
from env_factory.generation.negative_scenarios import derive_negatives

BASE=[{'scenario_id':'success','kind':'goal_success','steps':[{'operation':'tool_call','tool_name':'buy','arguments':{'sku':'good'},'expected_status':200},{'operation':'reward','step_id':'reward'}]}]
DEF={'scenario_id':'bad','kind':'goal_failure','argument_mutations':[{'tool_name':'buy','arguments':{'sku':'bad'}}],'assertions':[{'source':'step:reward','path':'$.reward','operator':'lte','expected':0}]}

def test_preserves_runtime_assertions_and_original():
    original=copy.deepcopy(BASE)
    case=derive_negatives(BASE,[DEF])[-1]
    assert case['steps'][0]['expected_status']==200
    assert case['assertions']==DEF['assertions']
    assert BASE==original


def test_explicit_rejection_replaces_only_the_mutated_calls_inherited_timeout():
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner, SandboxError
    original = copy.deepcopy(BASE)
    original[0]['steps'][0]['expected_status'] = 504
    definition = copy.deepcopy(DEF)
    definition['argument_mutations'][0]['expected_status'] = 403
    case = derive_negatives(original, [definition])[-1]
    assert original[0]['steps'][0]['expected_status'] == 504
    assert case['assertions'] == definition['assertions']
    def call(method, path, body, headers):
        return (403, {'error': 'AUTHORIZATION_REQUIRED'}, {}) if path.startswith('/v1/tools/') else (200, {'reward': 0}, {})
    AcceptanceScenarioRunner(call).run(case)
    with pytest.raises(SandboxError, match='expected HTTP 403, got 200'):
        AcceptanceScenarioRunner(lambda *args: (200, {'reward': 0}, {})).run(case)


@pytest.mark.parametrize('status', [True, '403', 403.0, 99, 600, None])
def test_negative_status_must_be_one_explicit_valid_http_code(status):
    definition = copy.deepcopy(DEF)
    definition['argument_mutations'][0]['expected_status'] = status
    with pytest.raises(ValueError, match='expected_status'):
        derive_negatives(BASE, [definition])


def test_status_override_does_not_turn_an_unchanged_request_into_a_negative():
    definition = copy.deepcopy(DEF)
    definition['argument_mutations'][0].update(arguments={'sku': 'good'}, expected_status=403)
    with pytest.raises(ValueError, match='must change reference'):
        derive_negatives(BASE, [definition])

@pytest.mark.parametrize('change',['no_change','unknown','duplicate','ambiguous','missing','manual_steps'])
def test_rejects_invalid_mutations(change):
    base=copy.deepcopy(BASE);d=copy.deepcopy(DEF)
    if change=='no_change':d['argument_mutations'][0]['arguments']['sku']='good'
    if change=='unknown':d['argument_mutations'][0]['arguments']={'absent':'bad'}
    if change=='duplicate':d['argument_mutations']*=2
    if change=='ambiguous':base[0]['steps'].insert(0,copy.deepcopy(base[0]['steps'][0]))
    if change=='missing':d['argument_mutations'][0]['tool_name']='absent'
    if change=='manual_steps':d['steps']=[]
    with pytest.raises(ValueError):derive_negatives(base,[d])


def test_compact_negative_keeps_dialogue_and_reaches_reward(tmp_path, monkeypatch):
    from tests.generation.test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner
    full, request = interactive_source()
    compact = {key:copy.deepcopy(full[key]) for key in
               ('version','description','environment_plan','tables','interaction_contract')}
    compact['business_tools'] = [
        {'name':tool['function']['name'], 'description':tool['function']['description'],
         'parameters':tool['function']['parameters'], 'implementation':implementation}
        for tool,implementation in zip(full['tools'],full['tool_implementations'])]
    compact['outcomes'] = [{'id':full['metrics'][0]['id'], 'rubric':full['metrics'][0]['rubric'],
                            'rule':full['metric_implementations'][0]}]
    steps = full['scenarios'][0]['steps']
    compact['reference'] = {'calls':[step for step in steps if step['operation']=='tool_call'],
                            'answer':steps[-1]['content']}
    compact['negative_scenarios'] = [{'scenario_id':'wrong_zone','kind':'goal_failure','from_reference':True,
        'argument_mutations':[{'tool_name':'read_temperature','arguments':{'id':547}}],
        'assertions':[{'source':'step:reward','path':'$.reward','operator':'lte','expected':0}]}]
    runs=[]
    original=AcceptanceScenarioRunner.run
    def observed(self,scenario, **kwargs):
        result=original(self,scenario, **kwargs)
        if scenario.get('scenario_id','').startswith('wrong_zone--'):runs.append(result)
        return result
    monkeypatch.setattr(AcceptanceScenarioRunner,'run',observed)
    result=compile_source(compact,root=tmp_path,request=request)
    assert result['generation_pipeline']['verification']['passed']
    assert len(runs)==3
    for run in runs:
        assert all(step['status']==200 for step in run['history'])
        assert sum(step['operation']=='dialogue_turn' for step in run['history'])==2
        assert run['step_results']['reward']['reward']==0
