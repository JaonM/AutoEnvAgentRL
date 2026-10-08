import copy
import json
from unittest.mock import Mock, patch

import pytest

from env_factory.generation.business_blueprint import validate_blueprint, validate_implementation


def blueprint():
    return {'version': '1.0', 'description': {
        'task': 'Query the chosen collection under the user limit', 'goal': 'Return a verified option',
        'expected_result': 'Verified option', 'task_intent': 'query', 'complexity': 'standard',
        'requirements': {'input_modalities': ['text'], 'output_modalities': ['text']},
        'public_input': {'initial_user_message': 'Find an option in the catalogue', 'materials': []}},
        'environment_mode': 'reference_data', 'requirements': [{'id': 'fit', 'statement': 'Respect the user limit'}],
        'tools': [{'name': 'lookup', 'purpose': 'Find collection identity', 'operation': 'select',
                   'inputs': {'name': {'kind': 'user', 'stage': 'collection'}}},
                  {'name': 'quote', 'purpose': 'Read option terms', 'operation': 'select',
                   'inputs': {'id': {'kind': 'tool', 'tool': 'lookup', 'field': 'id'},
                              'limit': {'kind': 'user', 'stage': 'limit'}}}],
        'interactions': [{'id': 'collection', 'kind': 'information_required', 'before_tool': 'lookup',
                          'argument': 'name', 'purpose': 'Choose collection'},
                         {'id': 'limit', 'kind': 'option_selection', 'before_tool': 'quote',
                          'argument': 'limit', 'purpose': 'Reveal limit'}]}


def request():
    return {'training_category': 'multi_step_agentic', 'available_environment_modes': ['reference_data'],
            'user_script_count': 1}


def implementation():
    plan = blueprint()
    return {'description': plan['description'], 'environment_plan': {'mode': 'reference_data'},
            'business_requirements': plan['requirements'],
            'tool_implementations': [{'tool_name': t['name'], 'operation': t['operation']} for t in plan['tools']],
            'tools': [{'function': {'name': t['name'], 'parameters': {'properties': dict.fromkeys(t['inputs'], {})}}}
                      for t in plan['tools']],
            'capability_plan': [{'action_name': 'lookup', 'dependencies': []},
                                {'action_name': 'quote', 'dependencies': ['lookup']}],
            'scenarios': [{'kind': 'goal_success', 'steps': [
                {'operation': 'tool_call', 'tool_name': 'lookup', 'arguments': {'name': 'chosen'},
                 'capture': {'id': '$.records[0].id'}},
                {'operation': 'tool_call', 'tool_name': 'quote', 'arguments': {'id': {'$ref': 'id'}, 'limit': 10}}]}],
            'interaction_contract': {'variants': [{'stages': [dict(stage, bind_to={
                'tool': stage['before_tool'], 'argument': stage['argument']}) for stage in plan['interactions']]}]}}


def test_blueprint_proves_only_structure_and_locks_materialization():
    result = validate_blueprint(blueprint(), request())
    assert result['tool_order'] == ['lookup', 'quote']
    assert result['scope'] == 'business_structure_not_executable_certification'
    validate_implementation(blueprint(), implementation(), request())
    changed = implementation()
    changed['description']['goal'] = 'Ignore limit'
    with pytest.raises(ValueError, match='original description'):
        validate_implementation(blueprint(), changed, request())


def test_cycles_fail_early():
    plan = blueprint()
    plan['tools'][0]['inputs']['name'] = {'kind': 'tool', 'tool': 'quote', 'field': 'name'}
    with pytest.raises(ValueError, match='cyclic'):
        validate_blueprint(plan, request())


def test_blueprint_rejects_interaction_kind_conflict_before_freezing():
    plan = blueprint()
    plan['interactions'][1]['kind'] = 'information_required'
    with pytest.raises(ValueError, match='two distinct interaction kinds'):
        validate_blueprint(plan, request())


def test_missing_provenance_and_unbound_user_inputs_fail_early():
    plan = blueprint()
    plan['tools'][1]['inputs']['id'] = {'kind': 'public', 'pointer': '/description/private/id'}
    with pytest.raises(ValueError, match='public surface'):
        validate_blueprint(plan, request())
    plan = blueprint()
    plan['tools'][1]['inputs']['limit'] = {'kind': 'user', 'stage': 'collection'}
    with pytest.raises(ValueError, match='not bound'):
        validate_blueprint(plan, request())


def test_blueprint_cannot_drop_obligations_dependencies_or_interaction_bindings():
    for mutate in (
        lambda s: s['business_requirements'].clear(),
        lambda s: s['capability_plan'][1]['dependencies'].clear(),
        lambda s: s['interaction_contract']['variants'][0]['stages'][1]['bind_to'].update(argument='wrong'),
    ):
        document = implementation()
        mutate(document)
        with pytest.raises(ValueError, match='BLUEPRINT_DRIFT'):
            validate_implementation(blueprint(), document, request())


def test_parent_validates_blueprint_before_source_compilation(tmp_path):
    from env_factory.generation.code_agent import generate
    from env_factory.generation.semantic_review import CHECKS
    phases = []
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            phases.append(prompt)
            workspace = tmp_path / 'authoring'
            if len(phases) == 1:
                (workspace / 'blueprint.json').write_text(json.dumps(blueprint()))
            else:
                (workspace / 'source.json').write_text(json.dumps(implementation()))
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(communicate=finish, returncode=0)
    artifact = {'generation_pipeline': {'verification': {}}, 'requirement_coverage': {'status': 'not_declared'}}
    review = {'status': 'pass', 'checks': {k: {'passed': True, 'evidence': 'checked'} for k in CHECKS},
              'findings': [], 'provenance': {'completed_turns': 1, 'usage': {}, 'events_sha256': 'hash'}}
    with patch('env_factory.generation.code_agent.subprocess.Popen', side_effect=launch), \
         patch('env_factory.generation.code_agent.compile_source', return_value=artifact) as compile_task, \
         patch('env_factory.generation.code_agent.verify_delivery'), \
         patch('env_factory.generation.code_agent.review_source', return_value=review):
        result = generate(request=request(), artifact_dir=tmp_path, script_count=1)
    assert len(phases) == 2
    assert 'Do not implement source.json yet' in phases[0]
    assert 'immutable blueprint.json' in phases[1]
    assert compile_task.call_count == 1
    assert compile_task.call_args.kwargs['blueprint'] == blueprint()
    assert result['generation_pipeline']['authoring_stages'] == ['business_blueprint', 'implementation']
    assert result['generation_pipeline']['blueprint_validation']['passed']


def test_blueprint_allows_declared_disclosure_variants_but_preserves_authorization():
    plan = blueprint()
    plan['interactions'][1]['allowed_kinds'] = ['option_selection', 'correction']
    source = implementation()
    source['interaction_contract']['variants'][0]['stages'][1]['kind'] = 'correction'
    validate_implementation(plan, source, request())
    plan['interactions'][1]['allowed_kinds'].append('execution_confirmation')
    with pytest.raises(ValueError, match='allowed interaction kinds'):
        validate_blueprint(plan, request())


def test_blueprint_rejects_read_write_conflicts_and_operation_drift():
    plan = blueprint()
    plan['tools'][1]['operation'] = 'update'
    with pytest.raises(ValueError, match='read-only'):
        validate_blueprint(plan, request())
    plan = blueprint()
    plan['environment_mode'] = 'stateful'
    with pytest.raises(ValueError, match='requires a write'):
        validate_blueprint(plan, dict(request(), available_environment_modes=['stateful']))
    plan = blueprint()
    plan['interactions'][1]['kind'] = 'execution_confirmation'
    with pytest.raises(ValueError, match='protect a write'):
        validate_blueprint(plan, request())
    source = implementation()
    source['tool_implementations'][1]['operation'] = 'update'
    with pytest.raises(ValueError, match='tool operation changed'):
        validate_implementation(blueprint(), source, request())


def test_blueprint_rejects_literal_replacement_of_tool_result():
    source = implementation()
    source['scenarios'][0]['steps'][1]['arguments']['id'] = 'known-fixture-id'
    with pytest.raises(ValueError, match='planned producer field'):
        validate_implementation(blueprint(), source, request())


@pytest.mark.parametrize('field,value', [('kind', []), ('before_tool', {}),
    ('purpose', 7), ('argument', []), ('allowed_kinds', ['option_selection', {}])])
def test_malformed_stage_returns_repairable_validation_error(field, value):
    plan = blueprint()
    plan['interactions'][1][field] = value
    with pytest.raises(ValueError, match='BLUEPRINT:'):
        validate_blueprint(plan, request())


@pytest.mark.parametrize('origin', [
    {'kind': 'public', 'pointer': None}, {'kind': 'user', 'stage': []},
    {'kind': 'tool', 'tool': {}, 'field': 'id'}])
def test_malformed_origin_returns_repairable_validation_error(origin):
    plan = blueprint()
    plan['tools'][1]['inputs']['id'] = origin
    with pytest.raises(ValueError, match='BLUEPRINT:'):
        validate_blueprint(plan, request())


def test_disclosed_selection_can_inform_derived_arguments_of_same_tool():
    plan = blueprint()
    plan['tools'][1]['inputs']['id'] = {'kind': 'derived', 'sources': [
        {'kind': 'tool', 'tool': 'lookup', 'field': 'id'},
        {'kind': 'user', 'stage': 'limit'}]}
    validate_blueprint(plan, request())
    # A stage protecting a different tool cannot be borrowed as an input.
    plan['tools'][1]['inputs']['id']['sources'][1]['stage'] = 'collection'
    with pytest.raises(ValueError, match='not bound'):
        validate_blueprint(plan, request())


def test_derived_capture_can_select_a_descendant_of_declared_result():
    plan = blueprint()
    plan['tools'][1]['inputs']['id'] = {'kind': 'derived', 'sources': [
        {'kind': 'tool', 'tool': 'lookup', 'field': '$.records'},
        {'kind': 'user', 'stage': 'limit'}]}
    validate_implementation(plan, implementation(), request())
    source = implementation()
    source['scenarios'][0]['steps'][0]['capture']['id'] = '$.other_records[0].id'
    with pytest.raises(ValueError, match='planned producer field'):
        validate_implementation(plan, source, request())
    plan['tools'][1]['inputs']['id'] = {'kind': 'tool', 'tool': 'lookup', 'field': '$.records'}
    with pytest.raises(ValueError, match='planned producer field'):
        validate_implementation(plan, implementation(), request())


def test_blueprint_structured_interactions_and_reward_coverage_compile_together(tmp_path):
    from tests.generation.test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    source, task_request = interactive_source()
    source['business_requirements'] = [{
        'id': 'temperature', 'statement': 'Return the actual queried temperature',
        'input_refs': ['/description/public_input/initial_user_message'],
        'tools': ['lookup_zone', 'read_temperature'], 'outcome_ids': ['temperature_answer'],
        'positive_scenarios': ['success'], 'negative_scenarios': ['failure']}]
    task_request.update(user_script_count=3, require_business_requirements=True,
                        require_structured_interactions=True)
    stages = source['interaction_contract']['variants'][0]['stages']
    plan = {'version': '1.0', 'description': copy.deepcopy(source['description']),
        'environment_mode': source['environment_plan']['mode'],
        'requirements': [{'id': 'temperature', 'statement': 'Return the actual queried temperature'}],
        'tools': [
            {'name': 'lookup_zone', 'operation': 'select', 'purpose': 'Locate chosen warehouse',
             'inputs': {'name': {'kind': 'user', 'stage': 'warehouse'},
                        'region': {'kind': 'user', 'stage': 'choice'}}},
            {'name': 'read_temperature', 'operation': 'select', 'purpose': 'Read actual temperature',
             'inputs': {'id': {'kind': 'tool', 'tool': 'lookup_zone', 'field': 'id'}}}],
        'interactions': [{key: stage[key] for key in ('id', 'kind', 'before_tool', 'purpose')}
                         | {'argument': stage['bind_to']['argument']} for stage in stages]}
    plan['interactions'][1]['allowed_kinds'] = ['option_selection', 'constraint_update', 'correction']
    artifact = compile_source(source, root=tmp_path, request=task_request, blueprint=plan)
    assert artifact['requirement_coverage']['paired_execution_verified']
    assert artifact['reward_pair_validation']['passed']
    report = json.loads((tmp_path / 'requirement_reward_validation.json').read_text())
    assert report['false_negative']['measured'] == 3
    # The authored failure has no successful interaction prefix to expand.
    assert report['false_positive']['measured'] == 1
    assert report['false_positive']['errors'] == 0


@pytest.mark.parametrize('failure', ['invalid_blueprint', 'accepted_blueprint_tamper'])
def test_parent_blocks_implementation_until_blueprint_is_valid_and_immutable(tmp_path, failure):
    from env_factory.generation.code_agent import generate
    from env_factory.generation.pipeline_errors import PipelineGenerationError
    phases = []
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            phases.append(prompt)
            workspace = tmp_path / 'authoring'
            plan = blueprint()
            if failure == 'invalid_blueprint':
                plan['tools'][0]['operation'] = 'unsupported'
            elif len(phases) > 1:
                plan['description']['goal'] = 'Simplified goal'
                (workspace / 'source.json').write_text(json.dumps(implementation()))
            (workspace / 'blueprint.json').write_text(json.dumps(plan))
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(communicate=finish, returncode=0)
    with patch('env_factory.generation.code_agent.subprocess.Popen', side_effect=launch), \
         patch('env_factory.generation.code_agent.compile_source') as compiler:
        with pytest.raises(PipelineGenerationError, match=(
                'unsupported tool operation' if failure == 'invalid_blueprint' else 'accepted blueprint changed')):
            generate(request=request(), artifact_dir=tmp_path, script_count=1)
    assert len(phases) == 2
    compiler.assert_not_called()


def test_blueprint_user_input_reuse_keeps_real_dependency_and_requires_binding():
    plan, source = blueprint(), implementation()
    plan['tools'][1]['inputs']['name'] = {'kind': 'user', 'stage': 'collection'}
    result = validate_blueprint(plan, request())
    assert result['reused_user_inputs'] == [{'stage': 'collection', 'producer': 'lookup', 'tool': 'quote', 'argument': 'name'}]
    source['tools'][1]['function']['parameters']['properties']['name'] = {}
    with pytest.raises(ValueError, match='reused user disclosure'):
        validate_implementation(plan, source, request())
    source['interaction_contract']['variants'][0]['stages'][0]['reuse_bindings'] = [{'tool': 'quote', 'argument': 'name'}]
    validate_implementation(plan, source, request())
    # Reuse itself cannot manufacture the hidden-data dependency.
    plan['tools'][1]['inputs']['id'] = {'kind': 'public', 'pointer': '/description/public_input/initial_user_message'}
    with pytest.raises(ValueError, match='real tool-result dependency'):
        validate_blueprint(plan, request())


def test_blueprint_rejects_impossible_or_ambiguous_select_echo_before_freeze():
    plan = blueprint()
    plan['tools'][0]['inputs']['id'] = {'kind': 'public', 'pointer': '/description/public_input/initial_user_message'}
    with pytest.raises(ValueError, match='does not echo'):
        validate_blueprint(plan, request())
    plan['tools'][1]['inputs']['id']['field'] = '$.id'
    with pytest.raises(ValueError, match='arbitrary root fields'):
        validate_blueprint(plan, request())
    plan['tools'][1]['inputs']['id']['field'] = '$.records[0].id'
    validate_blueprint(plan, request())
    plan['tools'][0]['result_field'] = 'items'
    plan['tools'][1]['inputs']['id']['field'] = '$.items[0].id'
    validate_blueprint(plan, request())


def test_user_reuse_requires_ancestry_not_just_an_unrelated_dependency():
    plan = blueprint()
    plan['tools'].append({'name': 'unrelated', 'purpose': 'Independent catalogue', 'operation': 'select',
                          'inputs': {'name': {'kind': 'user', 'stage': 'collection'}}})
    with pytest.raises(ValueError, match='must follow its disclosure tool'):
        validate_blueprint(plan, request())
