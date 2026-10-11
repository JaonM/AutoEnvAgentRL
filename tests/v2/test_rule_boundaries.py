import copy

from env_factory_v2.examples import direct, lit, ref
from env_factory_v2.validation.verification import qualify


def test_stateless_boundary_probe_detects_wrong_rule_without_changing_agent_answer():
    source = direct()
    source['rules'] = [{'id': 'strict_threshold', 'public': 'The predicate is true only above zero.',
                        'predicate': {'gt': [ref('args.value'), lit(0)]}}]
    source['probes'] = [{'id': 'zero_boundary', 'scenario': 'correct',
                         'rule_ids': ['strict_threshold'], 'mutations': [], 'expect_success': True,
                         'answer': {'temperature': {'if': {
                             'condition': {'eq': [{'rule': {'id': 'strict_threshold',
                                                            'arguments': {'value': lit(0)}}}, lit(False)]},
                             'then': lit(18), 'else': lit(19)}}}}]
    compiled, report = qualify(source)
    assert report['gates']['joint_execution_and_counterexamples']
    assert next(r for r in report['records'] if r['id'] == 'zero_boundary')['first']['answer'] == {'temperature': 18}
    assert not compiled['public']['tools']
    assert [a['id'] for a in compiled['public']['answer_fields']] == ['temperature']
    wrong = copy.deepcopy(source)
    wrong['rules'][0]['predicate'] = {'ge': [ref('args.value'), lit(0)]}
    _, report = qualify(wrong)
    assert not report['gates']['joint_execution_and_counterexamples']
    boundary = next(r for r in report['records'] if r['id'] == 'zero_boundary')
    assert not boundary['passed'] and boundary['first']['outcome']['reward'] == 0
