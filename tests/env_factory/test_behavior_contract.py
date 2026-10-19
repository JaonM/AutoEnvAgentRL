import copy
from types import SimpleNamespace

from env_factory.certify import behavior
from env_factory.runtime import write, BusinessError


def contract():
    return {'version': 1, 'cases': [{'id': 'library-stock', 'data': {'books': [{'stock': 2}]},
        'steps': [{'tool': 'borrow', 'args': {}, 'checks': [{'name': 'stock reduced',
            'assert': {'op': 'eq', 'args': [{'path': 'after.books.0.stock'}, 1]}}]},
                  {'tool': 'fail', 'args': {}, 'error': '*'}]}]}


def module(broken=False):
    @write
    def borrow(db, now):
        if not broken: db['books'][0]['stock'] -= 1
        return 'borrowed'
    @write
    def fail(db, now):
        db['books'][0]['stock'] = 0
        raise BusinessError('unavailable')
    return SimpleNamespace(make_tools=lambda: {'borrow': borrow, 'fail': fail})


def test_independent_properties_detect_a_plausible_receipt_with_missing_effect():
    value = contract();before = copy.deepcopy(value)
    assert behavior.run(module(), value)['passed']
    report = behavior.run(module(True), value)
    assert not report['passed']
    assert report['cases'][0]['failures'][0]['rule'] == 'stock reduced'
    assert value == before


def test_contract_tampering_and_empty_tests_fail_closed():
    value = contract();request = {'behavior_contract': value, 'behavior_contract_sha256': behavior.digest(value)}
    assert behavior.qualify(module(), request)['passed']
    value['cases'][0]['steps'] = []
    assert not behavior.qualify(module(), request)['passed']
    assert not behavior.run(module(), {'version': 1, 'cases': []})['passed']


def test_dependencies_and_unsupported_expressions():
    assert behavior.expression({'path': 'outputs.0.id'}, {'outputs': [{'id': 'new'}]}) == 'new'
    value = contract();value['cases'][0]['steps'][0]['checks'][0]['assert']['op'] = 'eval'
    assert not behavior.run(module(), value)['passed']
