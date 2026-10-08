import copy
import pytest
from env_factory.generation.business_conditions import compile_business_conditions, validate_condition_bindings
from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator


def design():
    return {
        'description': {'public_input': {'initial_user_message': 'Venue must support Persian and fit the requested attendance.'}},
        'tools': [{'function': {'name': 'venues', 'parameters': {'properties': {
            'language_support': {'type': 'boolean'}, 'attendance': {'type': 'integer'}},
            'required': ['language_support', 'attendance']}}}],
        'tables': [{'table_name': 'venues', 'columns': [{'name': key} for key in ['name', 'supports_persian', 'capacity']]}],
        'tool_implementations': [{'tool_name': 'venues', 'operation': 'select', 'table': 'venues',
            'projection': ['name', 'supports_persian', 'capacity']}],
        'business_conditions': [
            {'id': 'language', 'public_requirement': 'must support Persian', 'tool': 'venues',
             'argument': 'language_support', 'column': 'supports_persian', 'operator': 'eq'},
            {'id': 'capacity', 'public_requirement': 'fit the requested attendance', 'tool': 'venues',
             'argument': 'attendance', 'column': 'capacity', 'operator': 'gte'}],
        'scenarios': [{'kind': 'goal_success', 'steps': [{'operation': 'tool_call', 'tool_name': 'venues',
            'arguments': {'language_support': True, 'attendance': 60}}]}],
        'metric_implementations': [{'operator': 'value_targets', 'expected': {'targets': [
            {'key': 'venue', 'expression': {'from_tool': {'name': 'venues', 'field': 'name'}}}]}}]}


def test_conditions_compile_once_into_query_and_reward_and_follow_changed_data():
    source = design()
    conditions = compile_business_conditions(source)
    # Compilation is idempotent; redundant identical source filters do not multiply.
    compile_business_conditions(source)
    assert len(source['tool_implementations'][0]['filters']) == 2
    bindings = bind_reward_queries(source)
    validate_condition_bindings(conditions, bindings, source)
    expression = source['metric_implementations'][0]['expected']['targets'][0]['expression']
    def evaluate(rows):
        return DeclarativeMetricEvaluator._value_expression(expression, {'venues': rows})
    good = {'name': 'Hall A', 'supports_persian': True, 'capacity': 80}
    assert evaluate([good]) == 'Hall A'
    # One-condition-at-a-time negative controls, including the actual omitted-language defect.
    for mutation in [{'supports_persian': False}, {'capacity': 59}]:
        assert evaluate([{**good, **mutation}]) is None
    # A new eligible row becomes the answer; no fixture answer is frozen.
    assert evaluate([{**good, 'supports_persian': False},
        {'name': 'Hall B', 'supports_persian': True, 'capacity': 60}]) == 'Hall B'


@pytest.mark.parametrize('change', ['private', 'optional', 'conflict', 'duplicate', 'column'])
def test_invalid_condition_contract_fails_before_execution(change):
    source = design()
    if change == 'private': source['business_conditions'][0]['public_requirement'] = 'secret rule'
    if change == 'optional': source['tools'][0]['function']['parameters']['required'] = ['attendance']
    if change == 'conflict': source['tool_implementations'][0]['filters'] = [
        {'argument': 'language_support', 'column': 'supports_persian', 'operator': 'lte'}]
    if change == 'duplicate': source['business_conditions'].append(copy.deepcopy(source['business_conditions'][0]))
    if change == 'column': source['business_conditions'][0]['column'] = 'missing'
    with pytest.raises(ValueError, match='BUSINESS_CONDITION_INVALID'):
        compile_business_conditions(source)


def test_missing_reference_condition_cannot_silently_disappear():
    source = design()
    conditions = compile_business_conditions(source)
    del source['scenarios'][0]['steps'][0]['arguments']['language_support']
    bindings = bind_reward_queries(source)
    with pytest.raises(ValueError, match='BUSINESS_CONDITION_UNBOUND.*language'):
        validate_condition_bindings(conditions, bindings, source)


def test_declared_eligibility_must_reach_an_outcome_not_just_a_tool():
    source = design()
    conditions = compile_business_conditions(source)
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'literal': 'Hall A'}
    bindings = bind_reward_queries(source)
    with pytest.raises(ValueError, match='BUSINESS_CONDITION_UNSCORED.*language'):
        validate_condition_bindings(conditions, bindings, source)


def test_condition_contract_runs_through_full_compiler_and_reference(tmp_path):
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source
    source, request = fixture('simple_agentic')
    source['description']['public_input']['initial_user_message'] += '温度不得超过30摄氏度。'
    schema = source['tools'][0]['function']['parameters']
    schema['properties']['max_temperature'] = {'type': 'integer', 'description': '允许的最高温度'}
    schema['required'].append('max_temperature')
    for step in source['scenarios'][0]['steps']:
        if step.get('operation') == 'tool_call': step['arguments']['max_temperature'] = 30
    source['business_conditions'] = [{'id': 'temperature_limit', 'public_requirement': '温度不得超过30摄氏度',
        'tool': 'read_temperature', 'argument': 'max_temperature', 'column': 'temperature', 'operator': 'lte'}]
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'read_temperature', 'field': 'temperature'}}
    result = compile_source(source, root=tmp_path, request=request)
    assert result['generation_pipeline']['verification']['passed'] is True
    execution = result['generation_pipeline']['verification']['business_condition_execution']
    assert execution and all(case['status'] == 'passed' for case in execution)
    assert execution[0]['execution']['kind'] == 'completed'
    assert execution[0]['execution']['reward']['reward'] < 1
    assert (tmp_path / 'business_condition_execution.json').is_file()
    filters = result['tool_implementations'][0]['filters']
    assert {'argument': 'max_temperature', 'column': 'temperature', 'operator': 'lte'} in filters


def test_automatic_probes_preserve_source_and_isolate_each_condition():
    from env_factory.generation.business_conditions import probe_business_conditions
    source = design()
    source['tables'][0]['rows'] = [
        {'name': 'Hall A', 'supports_persian': True, 'capacity': 80},
        {'name': 'Hall B', 'supports_persian': False, 'capacity': 59}]
    conditions = compile_business_conditions(source)
    bindings = bind_reward_queries(source)
    before = copy.deepcopy(source)
    report = probe_business_conditions(conditions, bindings, source)
    assert source == before
    assert report['verified'] == 2
    assert report['needs_authored_case'] == 0
    assert all(case['other_predicates_match'] and not case['mutant_matches'] for case in report['cases'])
    assert report['scope'] == 'isolated_query_predicates_not_episode_reward'


def test_list_condition_probe_targets_the_reward_selected_alternative():
    from env_factory.generation.business_conditions import probe_business_conditions
    source = design()
    source['tables'][0]['rows'] = [
        {'name': 'Unused', 'supports_persian': True, 'capacity': 80},
        {'name': 'Chosen', 'supports_persian': True, 'capacity': 80}]
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['expression']['from_tool']['where'] = {'name': 'Chosen'}
    conditions = compile_business_conditions(source)
    bindings = bind_reward_queries(source)
    report = probe_business_conditions(conditions, bindings, source)
    assert all(case['row_index'] == 1 for case in report['cases'])
    for case in report['cases']:
        state = {'venues': copy.deepcopy(source['tables'][0]['rows'])}
        state['venues'][0][case['column']] = case['mutated_value']
        assert DeclarativeMetricEvaluator._value_expression(target['expression'], state) == 'Chosen'
        state['venues'][1][case['column']] = case['mutated_value']
        assert DeclarativeMetricEvaluator._value_expression(target['expression'], state) is None


def test_coupled_condition_is_reported_uncovered_instead_of_false_success():
    from env_factory.generation.business_conditions import probe_business_conditions
    source = design()
    source['tables'][0]['rows'] = [{'name': 'A', 'supports_persian': True, 'capacity': 80}]
    conditions = compile_business_conditions(source)
    bindings = bind_reward_queries(source)
    # A stronger conjunct makes it impossible to violate >=60 alone while keeping >=70.
    bindings['venues']['where']['capacity']['eq'] = 80
    report = probe_business_conditions(conditions, bindings, source)
    assert report['verified'] == 1
    assert report['needs_authored_case'] == 1
    assert report['cases'][1]['status'] == 'needs_authored_case'


def test_episode_counterfactual_detects_eligibility_removed_from_reward(tmp_path):
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source, verify_execution
    source, request = fixture('simple_agentic')
    source['description']['public_input']['initial_user_message'] += '库区必须支持波斯语。'
    source['tables'][0]['columns'].append({'name': 'supports_persian', 'type': 'boolean', 'nullable': False})
    for row in source['tables'][0]['rows']: row['supports_persian'] = True
    schema = source['tools'][0]['function']['parameters']
    schema['properties']['language_support'] = {'type': 'boolean', 'description': '是否要求波斯语支持'}
    schema['required'].append('language_support')
    for step in source['scenarios'][0]['steps']:
        if step.get('operation') == 'tool_call': step['arguments']['language_support'] = True
    source['business_conditions'] = [{'id': 'language', 'public_requirement': '必须支持波斯语',
        'tool': 'read_temperature', 'argument': 'language_support', 'column': 'supports_persian', 'operator': 'eq'}]
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'read_temperature', 'field': 'temperature'}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    probes = artifacts['generation_pipeline']['business_condition_probes']
    # Inject the historical defect: the answer value remains unchanged when only
    # eligibility changes, and the reward silently omits that condition.
    for rule in artifacts['metric_implementations']:
        if rule['operator'] == 'numeric_targets':
            rule['expected']['targets'][0]['expression']['lookup']['where'].pop('supports_persian')
    with pytest.raises(ValueError, match='AGENT_PREFLIGHT_FAILED:.*business_condition_0_language'):
        verify_execution(artifacts, tmp_path, condition_probes=probes)


def test_computed_decimal_operand_has_json_safe_evidence():
    import json
    from decimal import Decimal
    from env_factory.generation.business_conditions import probe_business_conditions, _report_value
    source = design()
    source['tables'][0]['rows'] = [{'name':'Hall A', 'supports_persian':True, 'capacity':80}]
    conditions = compile_business_conditions(source)
    bindings = bind_reward_queries(source)
    bindings['venues']['where']['capacity']['gte'] = {'op':'mul','args':[{'literal':6},{'literal':10}]}
    report = probe_business_conditions(conditions, bindings, source)
    assert json.loads(json.dumps(report))['cases'][1]['resolved_operand'] == 60
    assert _report_value(Decimal('0.123456789012345678901')) == {'decimal':'0.123456789012345678901'}
    with pytest.raises(ValueError,match='non-finite'):
        _report_value(Decimal('NaN'))


def test_condition_probe_invalidates_every_eligible_alternative():
    from env_factory.generation.business_conditions import probe_business_conditions
    source = design()
    source['tables'][0]['rows'] = [
        {'name': 'First', 'supports_persian': True, 'capacity': 80},
        {'name': 'Selected after external event', 'supports_persian': True, 'capacity': 65},
        {'name': 'Already ineligible', 'supports_persian': False, 'capacity': 50}]
    conditions = compile_business_conditions(source)
    bindings = bind_reward_queries(source)
    report = probe_business_conditions(conditions, bindings, source)
    for case in report['cases']:
        assert {change['row_index'] for change in case['row_mutations']} == {0, 1}
        rows = copy.deepcopy(source['tables'][0]['rows'])
        for change in case['row_mutations']:
            rows[change['row_index']][case['column']] = change['mutated_value']
        assert not any(DeclarativeMetricEvaluator._where_matches(row, case['query_where']) for row in rows)
        assert rows[2] == source['tables'][0]['rows'][2]


def test_scheduled_event_restoring_eligibility_is_inconclusive(tmp_path):
    import json
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source, verify_execution
    source, request = fixture('simple_agentic')
    source['description']['public_input']['initial_user_message'] += '温度不得超过30摄氏度。'
    schema = source['tools'][0]['function']['parameters']
    schema['properties']['max_temperature'] = {'type': 'integer', 'description': '允许的最高温度'}
    schema['required'].append('max_temperature')
    source['scenarios'][0]['steps'][0]['arguments']['max_temperature'] = 30
    source['business_conditions'] = [{'id': 'temperature_limit', 'public_requirement': '温度不得超过30摄氏度',
        'tool': 'read_temperature', 'argument': 'max_temperature', 'column': 'temperature', 'operator': 'lte'}]
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'read_temperature', 'field': 'temperature'}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    artifacts['business_lifecycle'] = {'version': '1.0', 'roles': [], 'tools': {}, 'dynamics': {
        'events': [{'id': 'restore_temperature', 'at_tick': 1, 'changes': [
            {'table': 'zones', 'selector': {'id': 821}, 'values': {'temperature': 18}}]}]}}
    with pytest.raises(ValueError, match='AGENT_PREFLIGHT_FAILED:.*business_condition'):
        verify_execution(artifacts, tmp_path,
            condition_probes=artifacts['generation_pipeline']['business_condition_probes'])
    evidence = json.loads((tmp_path / 'business_condition_execution.json').read_text())
    assert evidence[0]['status'] == 'needs_authored_case'
    assert evidence[0]['reason'] == 'eligible alternatives remain at query execution'


def test_actual_selected_record_carries_query_conditions_but_count_does_not():
    source = design()
    conditions = compile_business_conditions(source)
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'venues', 'path': '$.records[0].name'}}
    bindings = bind_reward_queries(source)
    validate_condition_bindings(conditions, bindings, source)
    source['metric_implementations'][0]['expected']['targets'][0]['expression']['tool_result']['path'] = '$.count'
    with pytest.raises(ValueError, match='BUSINESS_CONDITION_UNSCORED'):
        validate_condition_bindings(conditions, bindings, source)
