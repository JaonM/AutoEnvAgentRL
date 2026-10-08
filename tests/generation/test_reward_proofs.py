import copy
import itertools
import pytest
from env_factory.generation.reward_proofs import reward_upper_bound
from env_factory.sandbox_runtime import ContractRewardGate,ContractRewardAggregator


def metrics():
    return [{'id':name,'category':'outcome','weight':weight,'score_range':[0,1]}
            for name,weight in [('rule',.6),('meaning',.2),('reason',.2)]]


@pytest.mark.parametrize('known',[0,.5,1])
def test_bound_dominates_all_possible_unknown_scores(known):
    contract=metrics();gate=ContractRewardGate({},contract)
    proof=reward_upper_bound(contract,{'rule':known},gate,{})
    assert proof['scope']=='reward_upper_bound_not_observed_reward'
    assert set(proof['assumed_full_credit'])=={'meaning','reason'}
    for meaning,reason in itertools.product([0,.25,.5,.75,1],repeat=2):
        actual=ContractRewardAggregator(contract).aggregate(gate.apply({'rule':known,'meaning':meaning,'reason':reason},{}))
        assert actual['reward'] <= proof['reward_upper_bound']+1e-12


@pytest.mark.parametrize('change',['negative_weight','nan_weight','penalty','range','duplicate','invalid_known','custom_gate'])
def test_unsupported_contract_uses_actual_evaluation(change):
    contract=metrics();known={'rule':0}
    if change=='negative_weight':contract[0]['weight']=-.1
    if change=='nan_weight':contract[0]['weight']=float('nan')
    if change=='penalty':contract[0]['category']='penalty'
    if change=='range':contract[0]['score_range']=[-1,1]
    if change=='duplicate':contract[1]['id']='rule'
    if change=='invalid_known':known['rule']=float('inf')
    class CustomGate(ContractRewardGate):pass
    gate=(CustomGate if change=='custom_gate' else ContractRewardGate)({},contract)
    assert reward_upper_bound(contract,known,gate,{}) is None


def test_runtime_prerequisite_gate_is_part_of_bound():
    contract=metrics()
    gate=ContractRewardGate({'training_contract':{'category':'multi_step_agentic'},
                             'capability_dag':{'nodes':['query'],'edges':[]}},contract)
    context={'trajectory':{'events':[]}}
    proof=reward_upper_bound(contract,{'rule':1},gate,context)
    assert proof['reward_upper_bound']==0


@pytest.mark.parametrize('penalty', [-1, -.4, 0])
@pytest.mark.parametrize('weight', [0, .5, 2])
def test_known_penalty_bound_dominates_actual_unknown_scores(penalty, weight):
    contract = metrics() + [{'id': 'stop', 'category': 'penalty', 'weight': weight, 'score_range': [-1, 0]}]
    gate = ContractRewardGate({}, contract)
    known = {'rule': .5, 'stop': penalty}
    proof = reward_upper_bound(contract, known, gate, {})
    assert proof['assumed_full_credit'] == ['meaning', 'reason']
    assert proof['deterministic_components']['stop'] == penalty
    for meaning, reason in itertools.product([0, .25, .5, .75, 1], repeat=2):
        scores = {**known, 'meaning': meaning, 'reason': reason}
        actual = ContractRewardAggregator(contract).aggregate(gate.apply(scores, {}))
        assert actual['reward'] <= proof['reward_upper_bound'] + 1e-12


@pytest.mark.parametrize('known', [{}, {'stop': .1}, {'stop': -1.1}, {'stop': True}, {'stop': float('nan')}])
def test_unknown_or_invalid_penalty_requires_actual_evaluation(known):
    contract = metrics() + [{'id': 'stop', 'category': 'penalty', 'weight': 1, 'score_range': [-1, 0]}]
    assert reward_upper_bound(contract, known, ContractRewardGate({}, contract), {}) is None


def test_penalty_does_not_bypass_missing_runtime_prerequisites():
    contract = metrics() + [{'id': 'stop', 'category': 'penalty', 'weight': 2, 'score_range': [-1, 0]}]
    gate = ContractRewardGate({'training_contract': {'category': 'multi_step_agentic'},
                              'capability_dag': {'nodes': ['query'], 'edges': []}}, contract)
    proof = reward_upper_bound(contract, {'rule': 1, 'stop': -1}, gate, {'trajectory': {'events': []}})
    assert proof['reward_upper_bound'] == -1


def test_preflight_skips_only_provable_negative_model_evaluations(tmp_path,monkeypatch):
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source,verify_execution
    from env_factory.generation import reward_proofs
    from env_factory.sandbox_runtime import ContractModelMetricEvaluator
    source,request=fixture('simple_agentic')
    source['description']['public_input']['initial_user_message']+='温度不得超过30摄氏度。'
    schema=source['tools'][0]['function']['parameters']
    schema['properties']['max_temperature']={'type':'integer','description':'最高温度'}
    schema['required'].append('max_temperature')
    source['scenarios'][0]['steps'][0]['arguments']['max_temperature']=30
    source['business_conditions']=[{'id':'temperature_limit','public_requirement':'温度不得超过30摄氏度',
        'tool':'read_temperature','argument':'max_temperature','column':'temperature','operator':'lte'}]
    source['metric_implementations'][0]['expected']['targets'][0]['expression']={
        'from_tool':{'name':'read_temperature','field':'temperature'}}
    artifacts=compile_source(source,root=tmp_path,request=request)
    for metric in artifacts['metrics']:metric['weight']*=.9
    synthetic=copy.deepcopy(artifacts['metrics'][0])
    synthetic.update(id='test_semantic',category='outcome',type='model-based',weight=.1,score_range=[0,1])
    artifacts['metrics'].append(synthetic)
    calls=[]
    def fake(self,context,scores):
        calls.append(context.get('final_agent_response'))
        return {'test_semantic':float(context.get('final_agent_response')=='温度：18摄氏度。')}
    monkeypatch.setattr(ContractModelMetricEvaluator,'evaluate_all',fake)
    real_proof=reward_proofs.reward_upper_bound
    monkeypatch.setattr(reward_proofs,'reward_upper_bound',lambda *args:None)
    baseline=verify_execution(artifacts,tmp_path,condition_probes=artifacts['generation_pipeline']['business_condition_probes'])
    baseline_calls=len(calls);calls.clear()
    monkeypatch.setattr(reward_proofs,'reward_upper_bound',real_proof)
    optimized=verify_execution(artifacts,tmp_path,condition_probes=artifacts['generation_pipeline']['business_condition_probes'])
    assert optimized['checks']==baseline['checks']
    assert 0<len(calls)<baseline_calls
    execution=optimized['business_condition_execution'][0]['execution']
    assert execution['kind']=='reward_upper_bound'
    assert execution['proof']['assumed_full_credit']==['test_semantic']
    assert execution['proof']['reward_upper_bound']<1
def test_terminal_reward_conflict_reports_both_definitions():
    import copy
    import pytest
    from env_factory.generation.reward_proofs import validate_reward_goal_consistency
    goal = {'table': 'stock', 'where': {'sku': 'A'}, 'count': 1, 'values': {'reserved': 20}}
    rule = {'metric_id': 'reserved', 'operator': 'state_predicates',
            'expected': [copy.deepcopy(goal)], 'score_mapping': {'pass': 1, 'fail': 0}}
    source = {'metrics': [{'id': 'reserved', 'weight': 1}],
              'semantic_goal': {'row_predicates': [goal]}, 'metric_implementations': [rule]}
    validate_reward_goal_consistency(source)
    rule['expected'][0]['values']['reserved'] = 24
    with pytest.raises(ValueError, match='REWARD_GOAL_CONFLICT') as error:
        validate_reward_goal_consistency(source)
    assert 'reward_path' in str(error.value) and 'goal_path' in str(error.value)
    rule['expected'][0]['where'] = {'sku': 'B'}
    validate_reward_goal_consistency(source)
    rule['expected'][0]['where'] = {'sku': 'A'}
    source['metrics'][0]['weight'] = 0
    validate_reward_goal_consistency(source)
