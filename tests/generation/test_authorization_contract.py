from copy import deepcopy
import json
import pytest
from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_interaction_contract import interactive_source


def approved_source():
    from tests.generation.test_dynamic_state_goal import dynamic_source
    interactive, request = interactive_source()
    source, _ = dynamic_source()
    source['interaction_contract'] = interactive['interaction_contract']
    source['description']['public_input'] = interactive['description']['public_input']
    source['tools'][0] = interactive['tools'][0]
    source['tool_implementations'][0] = interactive['tool_implementations'][0]
    source['tables'][0]['columns'].append({'name': 'region', 'type': 'text', 'nullable': False})
    for row in source['tables'][0]['rows']:
        row['region'] = '东区'
    source['scenarios'][0]['steps'][0]['arguments']['region'] = '东区'
    request['require_bound_authorization'] = True
    source['business_lifecycle'] = {'version': '1.0', 'tools': {
        'update_temperature': {'requires_grants': ['adjust']}}}
    for variant in source['interaction_contract']['variants']:
        variant['stages'].append({'id': 'approve', 'kind': 'execution_confirmation',
            'purpose': '确认调整温度', 'assistant_contains_all': ['25'],
            'reference_response': '确认调整到25度吗？', 'user_reply': '我同意调整到25度。',
            'retry_reply': '请先确认调整目标。', 'requires_tools': ['read_temperature'],
            'before_tool': 'update_temperature', 'grant_permissions': ['adjust'],
            'approval_terms': {'adjust': {'scope': 'terms', 'postconditions': [
                {'table': 'zones', 'where': {'name': '青松库'}, 'values': {'temperature': 25}, 'count': 1}]}}})
    return source, request


def test_compiled_user_simulator_grants_terms_and_reference_passes(tmp_path):
    source, request = approved_source()
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']


def test_wrong_approved_temperature_rejects_reference(tmp_path):
    source, request = approved_source()
    for variant in source['interaction_contract']['variants']:
        variant['stages'][-1]['approval_terms']['adjust']['postconditions'][0]['values']['temperature'] = 24
    with pytest.raises(ValueError, match='PREFLIGHT|SCENARIO'):
        compile_source(source, request=request, root=tmp_path)


@pytest.mark.parametrize('change', ['missing', 'unenforced'])
def test_missing_or_unenforced_terms_fail_before_execution(tmp_path, change):
    source, request = approved_source()
    if change == 'missing':
        source['interaction_contract']['variants'][0]['stages'][-1].pop('approval_terms')
    else:
        source['business_lifecycle']['tools']['update_temperature']['requires_grants'] = []
    with pytest.raises(ValueError, match='AUTHORIZATION_TERMS_'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def proposal_source(target=25):
    source, request = approved_source()
    request['require_structured_interactions'] = True
    for row in source['tables'][0]['rows']:
        row['configured_temperature'] = target
    steps = source['scenarios'][0]['steps']
    write = next(step for step in steps if step.get('tool_name') == 'update_temperature')
    steps[-1]['content'] = f'温度：{target}摄氏度。'
    for variant in source['interaction_contract']['variants']:
        stage = variant['stages'][-1]
        stage.update(reference_response='请确认将所示库区调整到显示的配置温度。',
                     user_reply='确认，仅同意本次展示的库区和温度。',
                     assistant_contains_all=['配置温度'],
                     reference_proposal=deepcopy(write['arguments']))
        stage['approval_terms'] = {'adjust': {'scope': 'terms', 'proposal_arguments': ['id', 'value'],
            'postconditions': [{'table': 'zones', 'where': {'id': {'proposal_argument': 'id'}},
                'values': {'temperature': {'proposal_argument': 'value'}}, 'count': 1}]}}
    return source, request


@pytest.mark.parametrize('target', [25, 30])
def test_compiled_consent_follows_observed_proposal_and_exposes_only_format(tmp_path, target):
    source, request = proposal_source(target)
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']
    public = result['public_input']
    material = next(m for m in public['materials'] if m['name'] == 'approval_proposal_tools')
    assert json.loads(material['content']) == {'update_temperature': ['id', 'value']}
    assert 'reference_proposal' not in json.dumps(public)


def test_proposal_expression_is_bound_before_removing_required_reads(tmp_path):
    source, request = proposal_source()
    write = next(step for step in source['scenarios'][0]['steps']
                 if step.get('tool_name') == 'update_temperature')
    write['arguments']['value'] = {'$expr': {'op': 'add', 'args': [
        {'$ref': 'target_temperature'}, 0]}}
    for variant in source['interaction_contract']['variants']:
        variant['stages'][-1]['reference_proposal'] = deepcopy(write['arguments'])
    result = compile_source(source, request=request, root=tmp_path)
    verification = result['generation_pipeline']['verification']
    assert verification['passed']
    no_tools = [passed for name, passed in verification['checks'].items()
                if name.startswith('no_tools_')]
    assert no_tools and all(no_tools)


def test_ablation_binding_uses_capture_values_at_each_displayed_proposal():
    from env_factory.generation.agent_authoring import _bind_reference_for_ablations
    proposal = {'request': {'proposal': {'value': {'$expr': {
        'op': 'add', 'args': [{'$ref': 'temperature'}, 1]}}}}}
    steps = [
        {'operation': 'tool_call', 'arguments': {}, 'capture': {'temperature': '$.value'}},
        {'operation': 'dialogue_turn', 'content': json.dumps(proposal)},
        {'operation': 'tool_call', 'arguments': {}, 'capture': {'temperature': '$.value'}},
        {'operation': 'tool_call', 'arguments': {'value': {'$ref': 'temperature'}}}]
    original = deepcopy(steps)
    execution = {'history': [{'body': {'value': 7}}, {'body': {}},
                             {'body': {'value': 11}}, {'body': {}}],
                 'variables': {'temperature': 11}}
    bound = _bind_reference_for_ablations(steps, execution)
    assert json.loads(bound[1]['content'])['request']['proposal']['value'] == 8
    assert bound[3]['arguments']['value'] == 11
    assert steps == original


@pytest.mark.parametrize('defect', ['missing_reference', 'missing_parameter', 'legacy_request', 'mixed_format'])
def test_invalid_proposal_contract_rejected_before_execution(tmp_path, defect):
    source, request = proposal_source()
    stage = source['interaction_contract']['variants'][0]['stages'][-1]
    if defect == 'missing_reference':
        stage.pop('reference_proposal')
    elif defect == 'missing_parameter':
        stage['approval_terms']['adjust']['proposal_arguments'] = ['value']
        stage['approval_terms']['adjust']['postconditions'][0]['where'] = {'name': '青松库'}
    elif defect == 'legacy_request':
        request['require_structured_interactions'] = False
    else:
        stage['approval_terms']['adjust'] = {'scope': 'operation'}
    with pytest.raises(ValueError, match='AUTHORIZATION_PROPOSAL_INVALID'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def grouped_source():
    source, request = proposal_source()
    source['description']['public_input']['initial_user_message'] += '同时将备用温度设置为配置温度加1度；分别确认两个操作的目标值。'
    for row in source['tables'][0]['rows']:
        row['backup_temperature'] = 0
    source['tables'][0]['columns'].append({'name': 'backup_temperature', 'type': 'integer', 'nullable': False})
    original, additional = 'update_temperature', 'update_backup_temperature'
    tool = deepcopy(source['tools'][-1])
    tool['function'].update(name=additional, description='更新备用温度')
    source['tools'].append(tool)
    implementation = deepcopy(source['tool_implementations'][-1])
    implementation.update(tool_name=additional, changes={'value': 'backup_temperature'})
    source['tool_implementations'].append(implementation)
    source['actions'].append({**deepcopy(source['actions'][-1]), 'name': additional, 'description': '更新备用温度'})
    source['tool_bindings'].append({'tool_name': additional, 'action_name': additional})
    source['capability_plan'].append({**deepcopy(source['capability_plan'][-1]), 'action_name': additional})
    source['reward_key_steps'].append({**deepcopy(source['reward_key_steps'][-1]),
                                      'step_id': additional, 'action_name': additional})
    steps = source['scenarios'][0]['steps']
    position = next(i for i, step in enumerate(steps) if step.get('tool_name') == original)
    write = deepcopy(steps[position])
    write.update(tool_name=additional)
    write['arguments']['value'] = {'$expr': {'op': 'add', 'args': [{'$ref': 'target_temperature'}, 1]}}
    steps.insert(position + 1, write)
    goal = source['semantic_goal']['row_predicates'][0]['value_expressions']
    goal['backup_temperature'] = {'op': 'add', 'args': [deepcopy(goal['temperature']), {'literal': 1}]}
    source['business_lifecycle']['tools'][additional] = {'requires_grants': ['adjust_backup']}
    for variant in source['interaction_contract']['variants']:
        stage = variant['stages'][-1]
        stage['grant_permissions'].append('adjust_backup')
        stage['approval_terms']['adjust']['tool'] = original
        term = deepcopy(stage['approval_terms']['adjust'])
        term['tool'] = additional
        term['postconditions'][0]['values'] = {'backup_temperature': {'proposal_argument': 'value'}}
        stage['approval_terms']['adjust_backup'] = term
        stage['reference_proposal'] = {original: deepcopy(steps[position]['arguments']),
                                       additional: deepcopy(write['arguments'])}
        stage['reference_response'] = '请确认展示的主温度和备用温度两项操作及其各自参数。'
        stage['user_reply'] = '确认这两项操作，仅同意所展示的各自参数。'
    return source, request


def test_grouped_confirmation_compiles_and_executes_within_existing_turn_bound(tmp_path):
    source, request = grouped_source()
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']
    assert all(len(v['stages']) <= 5 for v in source['interaction_contract']['variants'])
    public = next(m for m in result['public_input']['materials'] if m['name'] == 'approval_proposal_tools')
    assert json.loads(public['content']) == {'update_temperature': {
        'update_temperature': ['id', 'value'], 'update_backup_temperature': ['id', 'value']}}
    assert 'target_temperature' not in public['content']


@pytest.mark.parametrize('defect', ['unknown_tool', 'unguarded', 'shared_grant', 'partial_parameters',
                                  'unrestricted', 'missing_operation', 'crossed_parameters', 'wrong_literal_type'])
def test_invalid_grouped_contract_fails_before_model_scoring(tmp_path, defect):
    source, request = grouped_source()
    stage = source['interaction_contract']['variants'][0]['stages'][-1]
    term = stage['approval_terms']['adjust_backup']
    if defect == 'unknown_tool': term['tool'] = 'unknown'
    if defect == 'unguarded': source['business_lifecycle']['tools']['update_backup_temperature']['requires_grants'] = []
    if defect == 'shared_grant': source['business_lifecycle']['tools']['update_backup_temperature']['requires_grants'].append('adjust')
    if defect == 'partial_parameters': term['proposal_arguments'] = ['value']
    if defect == 'unrestricted': stage['approval_terms']['adjust_backup'] = {'scope': 'operation'}
    if defect == 'missing_operation': stage['reference_proposal'].pop('update_backup_temperature')
    if defect == 'crossed_parameters': stage['reference_proposal']['update_backup_temperature'] = {'different': 25}
    if defect == 'wrong_literal_type': stage['reference_proposal']['update_backup_temperature']['value'] = [25]
    from env_factory.sandbox_runtime import SandboxError
    with pytest.raises((ValueError, SandboxError)):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)
