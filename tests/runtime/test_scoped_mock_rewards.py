import json, random
import pytest
from env_factory import sandbox_runtime as runtime

@pytest.mark.parametrize('inventory', [True, False])
def test_mock_does_not_borrow_another_user_choice(tmp_path, monkeypatch, inventory):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '1')
    ids = ['alpha', 'beta', 'gamma']
    seeds = {name: next(n for n in range(100) if random.Random(n).randrange(3) == i) for i,name in enumerate(ids)}
    contract = {'metrics': [{'id':'choice', 'category':'outcome', 'scope':'terminal',
        'semantic_fields':['selection'], 'evaluation_inputs':['final_agent_response'],
        'evaluator': {'kind':'external_llm_judge','score_mapping':{'pass':1,'fail':0}}}],
        'acceptance_contract': {'executable_scenarios': [
            {'kind':'goal_success','steps':[{'operation':'reset','body':{'seed':seeds[name]}},
              {'operation':'agent_response','content':json.dumps({'selection':name})}]} for name in ids]}}
    if inventory:
        contract['task_spec']={'interaction_complexity':{'behavioral_diversity':{'variants':[{'script_id':x} for x in ids]}}}
    store=runtime.EpisodeStore(tmp_path/'episodes.sqlite3');store.reset(seed=seeds['alpha'])
    evaluator=runtime.ContractModelMetricEvaluator(contract,store)
    context={'trajectory':{'events':[{'event':'user_protocol','payload':{'script_id':'alpha'}}]},
             'final_agent_response':json.dumps({'selection':'beta'})}
    assert evaluator.evaluate_all(context,{})['choice']==0
    context['final_agent_response']=json.dumps({'selection':'alpha'})
    assert evaluator.evaluate_all(context,{})['choice']==1
    if inventory:
        alternate=next(n for n in range(100) if n != seeds['alpha'] and random.Random(n).randrange(3)==0)
        store.reset(seed=alternate)
        assert evaluator.evaluate_all(context,{})['choice']==1
    assert all(event['payload']['semantic_verification'] is False for event in store.replay()['events']
               if event['event']=='evaluator_call')
