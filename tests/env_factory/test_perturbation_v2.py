import copy
from env_factory.core.worlds import variants, request_variants, perturbation_coverage
from env_factory.runtime import canonical


def test_nested_amount_enum_perturbations_are_distinct_and_preserve_relations():
    data = {'orders': [{'id': 'a', 'customer_id': 'u', 'details': {'price': 10.5, 'status': 'paid'},
                         'items': [{'id': 'i', 'quantity': 2}]},
                       {'id': 'b', 'customer_id': 'v', 'details': {'price': 20.5, 'status': 'cancelled'},
                        'items': [{'id': 'j', 'quantity': 3}]}]}
    before = copy.deepcopy(data)
    worlds = variants(data, {}, version=2)
    assert len(worlds) == 21 and len({canonical(d) for _, d, _ in worlds}) == 21
    assert worlds == variants(data, {}, version=2) and data == before
    assert all([(r['id'], r['customer_id'], r['items'][0]['id']) for r in d['orders']] ==
               [('a', 'u', 'i'), ('b', 'v', 'j')] for _, d, _ in worlds)
    report = perturbation_coverage({'tables': data, 'user_facts': {}, 'perturbation_version': 2}, [], requested=20)
    assert report['duplicates'] == 0 and report['unique_changed'] == 20
    assert 'orders.details.price' in report['changed_paths']
    assert 'orders.details.status' in report['changed_paths']
    assert 'orders.items.0.quantity' in report['changed_paths']


def test_finite_boolean_world_does_not_pad_coverage_with_duplicate_variants():
    worlds = variants({'t': [{'id': 1, 'active': True}]}, {}, version=2)
    assert len(worlds) == 2
    # Legacy bundles retain the exact old count and seed naming.
    old = variants({'t': [{'id': 1, 'active': True}]}, {})
    assert len(old) == 21 and old[-1][0] == 'perturb:20'


def test_declared_cross_entity_mutations_can_preserve_a_shared_constraint():
    request = {'tables': {'accounts': [{'id': 'a', 'balance': 100}, {'id': 'b', 'balance': 200}]},
               'user_facts': {}, 'perturbation_version': 2,
               'perturbation_cases': [{'id': 'transfer', 'rule_ids': ['conservation'], 'changes': [
                   {'table': 'accounts', 'id': 'a', 'path': ['balance'], 'value': 90},
                   {'table': 'accounts', 'id': 'b', 'path': ['balance'], 'value': 210}]}]}
    name, data, _ = request_variants(request, perturbed=1)[-1]
    assert name == 'perturb:case:transfer' and sum(r['balance'] for r in data['accounts']) == 300
    assert request['tables']['accounts'][0]['balance'] == 100
