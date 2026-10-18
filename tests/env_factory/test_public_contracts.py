"""Public contracts apply equally to an arbitrary business, without target-domain branches."""
import copy

from env_factory.production import contracts


def lookup_member(db, now, member_id: str, include_expired: bool = False):
    """Read a member's benefits.

    Args:
        member_id: Member identifier from the customer.
        include_expired: Whether to include expired benefits.
    """


CONTRACT = {
    'policy': 'Only the member can request their benefits. Values use points, not currency.',
    'tools': [{'type': 'function', 'function': {'name': 'lookup_member', 'parameters': {
        'type': 'object', 'properties': {'member_id': {'type': 'string'}, 'include_expired': {'type': 'boolean'}},
        'required': ['member_id']}}}],
}


def request():
    result = {'business_rules': 'Benefits are read-only.'}
    contracts.attach(result, CONTRACT)
    return result


def test_contract_is_copied_preserved_and_checked():
    original = copy.deepcopy(CONTRACT)
    value = request()
    assert value['agent_policy'] == CONTRACT['policy']
    assert contracts.interface_problems({'lookup_member': lookup_member}, value) == []
    value['public_contract']['policy'] = 'Invented policy'
    errors = contracts.interface_problems({'lookup_member': lookup_member}, value)
    assert 'public contract content hash mismatch' in errors
    assert 'agent policy differs from supplied public policy' in errors
    assert CONTRACT == original


def test_extra_tool_missing_tool_and_extra_business_parameter_fail():
    assert any('public tool names differ' in e for e in contracts.interface_problems({'other': lookup_member}, request()))

    def extra(db, now, member_id: str, include_expired: bool = False, quote_id: str = ''):
        """Lookup."""

    assert any('parameter names differ' in e for e in contracts.interface_problems({'lookup_member': extra}, request()))


def test_requiredness_and_types_are_not_silently_changed():
    def changed(db, now, member_id: int, include_expired: bool):
        """Lookup."""

    errors = contracts.interface_problems({'lookup_member': changed}, request())
    assert any('required parameters differ' in e for e in errors)
    assert any('JSON types differ' in e for e in errors)


def test_invalid_input_rejected_before_authoring():
    from env_factory.production.world import build_world

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid contract must not call a model')

    import tempfile
    result = build_world({'name': 'unrelated business', 'public_contract': {'policy': 'x', 'tools': []}},
                         tempfile.mkdtemp(), runner=forbidden, client=None, python='python', language='en', index=0)
    assert result['failure_class'] == 'public_contract'
    assert not result['training_ready']


def test_semantic_review_fails_closed_and_retains_source_identity():
    class Client:
        content = 'not a verdict'

        def chat(self, messages, **kwargs):
            assert 'an inquiry performs NO identification' in messages[0]['content']
            assert CONTRACT['policy'] in messages[0]['content']
            return {'content': self.content}

    client = Client()
    assert not contracts.review(client, {}, CONTRACT)['passed']
    client.content = '{"problems": ["Generated policy invents an identity exemption"]}'
    result = contracts.review(client, {}, CONTRACT)
    assert not result['passed'] and result['contract_sha256'] == contracts.digest(CONTRACT)
    client.content = '{"problems": []}'
    assert contracts.review(client, {}, CONTRACT)['passed']


def test_declared_authentication_requirement_cannot_be_overruled_by_model_audit():
    contract = {**CONTRACT, 'constraints': {'allow_unauthenticated_inquiry': False}}
    request = {'goal_schema': {'benefits_question': {'kind': 'inquiry'}}}

    class MustNotCall:
        def chat(self, *args, **kwargs):
            raise AssertionError('deterministic contradiction must fail before calling a model')

    result = contracts.review(MustNotCall(), request, contract)
    assert not result['passed'] and result['stage'] == 'declared_constraints'
    request['goal_schema']['benefits_question']['kind'] = 'account'
    assert contracts.specification_problems(request, contract) == []
    assert contracts.specification_problems({'goal_schema': {'public_fees': {'kind': 'inquiry'}}}, CONTRACT) == []
