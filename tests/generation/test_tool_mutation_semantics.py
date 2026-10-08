import pytest
import copy

from env_factory.generation.contract_compiler import ContractCompiler, PipelineGenerationError


def validate(name, extra=False):
    properties = {'id': {'type': 'string'}, 'status': {'type': 'string'}}
    if extra:
        properties['ignored'] = {'type': 'string'}
    ContractCompiler._validate_business_tool_semantics(
        tools=[{'type': 'function', 'function': {'name': name, 'parameters': {
            'type': 'object', 'properties': properties}}}],
        implementations=[{'tool_name': name, 'operation': 'update', 'table': 'orders',
                          'selector': {'id': 'id'}, 'changes': {'status': 'status'}}])


@pytest.mark.parametrize('name', ['reserve_costume_kit', 'dispatch_order', 'approve_request',
                                 'cancel_reservation', 'update_search_index'])
def test_domain_mutations_do_not_require_a_closed_verb_vocabulary(name):
    validate(name)


@pytest.mark.parametrize('name', ['read_order', 'get_status', 'query_inventory'])
def test_explicit_read_verb_cannot_hide_a_mutation(name):
    with pytest.raises(PipelineGenerationError, match='advertises a read'):
        validate(name)


def test_domain_verb_does_not_bypass_parameter_consumption():
    with pytest.raises(PipelineGenerationError, match='ignored'):
        validate('reserve_costume_kit', extra=True)


@pytest.mark.parametrize('name,valid', [('check_costume_reservation', True),
                                      ('read_evaluate_results', True),
                                      ('calculate_average', False), ('delete_order', False)])
def test_select_checks_explicit_claims_instead_of_requiring_read_whitelist(name, valid):
    def run():
        ContractCompiler._validate_business_tool_semantics(
            tools=[{'function': {'name': name, 'parameters': {'properties': {}}}}],
            implementations=[{'tool_name': name, 'operation': 'select', 'filters': []}])
    if valid:
        run()
    else:
        with pytest.raises(PipelineGenerationError, match='overclaims'):
            run()


def disclosure_contract():
    return {
        'tools': [{'function': {'name': 'read_components', 'parameters': {'properties': {
            'id': {'type': 'string'}, 'choice': {'type': 'string'}}}}}],
        'implementations': [{'tool_name': 'read_components', 'operation': 'select',
                             'filters': [{'argument': 'id', 'column': 'id'}]}],
        'metrics': [{'id': 'read_process', 'category': 'process'}],
        'interaction_scripts': [{'script_id': f'script-{index}', 'interaction_protocol': {'stages': [{
            'id': 'choose', 'kind': 'option_selection', 'private_fact': value,
            'bind_to': {'tool': 'read_components', 'argument': 'choice'}}]}}
            for index, value in [(1, 'A'), (2, 'B')]],
        'metric_implementations': [{'metric_id': 'read_process', 'operator': 'contains_tool_call',
            'expected': {'tool_name': 'read_components', 'by_user_script': {
                f'script-{index}': [{'tool_name': 'read_components',
                    'arguments': {'id': {'$ref': 'selected_id'}, 'choice': value}}]
                for index, value in [(1, 'A'), (2, 'B')]}}}],
    }


def test_user_parameter_consumed_by_script_scoped_process_contract():
    ContractCompiler._validate_business_tool_semantics(**disclosure_contract())


def test_captured_disclosure_requires_same_call_value_constraint():
    contract = disclosure_contract()
    options = contract['metric_implementations'][0]['expected']['by_user_script']
    for index, value in [(1, 'A'), (2, 'B')]:
        option = options[f'script-{index}'][0]
        option['arguments']['choice'] = {'$ref': 'observed_choice'}
        option['disclosed_arguments'] = {'choice': value}
    ContractCompiler._validate_business_tool_semantics(**contract)
    options['script-2'][0].pop('disclosed_arguments')
    with pytest.raises(PipelineGenerationError, match='ignored'):
        ContractCompiler._validate_business_tool_semantics(**contract)


@pytest.mark.parametrize('defect', ['missing_binding', 'missing_process_rule', 'missing_script',
                                   'wrong_disclosure', 'unconstrained_alternative', 'nonprocess_metric'])
def test_declarations_or_partial_coverage_cannot_hide_unused_parameters(defect):
    contract = disclosure_contract()
    scoped = contract['metric_implementations'][0]['expected']['by_user_script']
    if defect == 'missing_binding':
        contract['interaction_scripts'][0]['interaction_protocol']['stages'] = []
    elif defect == 'missing_process_rule':
        contract['metric_implementations'] = []
    elif defect == 'missing_script':
        scoped.pop('script-2')
    elif defect == 'wrong_disclosure':
        scoped['script-2'][0]['arguments']['choice'] = 'A'
    elif defect == 'unconstrained_alternative':
        extra = copy.deepcopy(scoped['script-1'][0])
        extra['arguments']['choice'] = 'invented'
        scoped['script-1'].append(extra)
    elif defect == 'nonprocess_metric':
        contract['metrics'][0]['category'] = 'outcome'
    with pytest.raises(PipelineGenerationError, match='ignored'):
        ContractCompiler._validate_business_tool_semantics(**contract)


def test_runtime_lifecycle_guard_operand_is_a_parameter_use():
    contract = disclosure_contract()
    contract['interaction_scripts'] = []
    contract['business_lifecycle'] = {'tools': {'read_components': {'guards': [
        {'selector': {'choice': {'argument': 'choice'}}, 'conditions': {}}]}}}
    ContractCompiler._validate_business_tool_semantics(**contract)
    contract['business_lifecycle']['tools']['read_components'] = {'description': {'argument': 'choice'}}
    with pytest.raises(PipelineGenerationError, match='ignored'):
        ContractCompiler._validate_business_tool_semantics(**contract)
