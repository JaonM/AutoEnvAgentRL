import pytest

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
