import copy
from unittest.mock import Mock, patch

from env_factory_v2.examples import direct, lit
from env_factory_v2.verification import qualify, run_scenario


def explained():
    source = direct()
    source['answers'][0]['weight'] = 0.5
    source['explanations'] = [{'id': 'reason', 'meaning': '解释公开温度', 'weight': 0.5,
        'criteria': [{'id': 'fact', 'rule_id': 'fixed_fact', 'obligation': '符合公开温度事实'}],
        'context_fields': ['temperature']}]
    for case in source['scenarios']:
        case['answer']['reason'] = lit('公开温度是18摄氏度。')
    extra = copy.deepcopy(source['scenarios'][0])
    extra['id'] = 'paraphrase'
    extra['answer']['reason'] = lit('18摄氏度是给定温度。')
    source['scenarios'].append(extra)
    return source


def judgment(passed=True):
    return {'criteria': {'fact': passed}, 'provenance': {'response_id': 'unit-only'}}


def test_unique_evidence_review_preserves_every_execution_and_scope_isolation():
    source = explained()
    primary = Mock(side_effect=lambda payload: judgment())
    independent = Mock(side_effect=lambda payload: judgment())
    with patch('env_factory_v2.verification.run_scenario', wraps=run_scenario) as execute:
        _, report = qualify(source, judge=primary, independent_judge=independent)
    assert execute.call_count == 6
    assert primary.call_count == 3
    assert independent.call_count == 2  # same text and evidence, then paraphrase
    assert report['gates']['joint_execution_and_counterexamples']
    evidence = report['records'][1]['repeat']['outcome']['semantic_evidence']['reason']
    assert evidence['provenance']['reuse_scope'] == 'qualification_independent_evidence'
    qualify(source, judge=primary, independent_judge=independent)
    assert independent.call_count == 4  # no cross-qualification reuse


def test_conflicting_independent_review_cannot_pass_by_reuse():
    _, report = qualify(explained(), judge=lambda p: judgment(),
                        independent_judge=lambda p: judgment(False))
    assert not report['gates']['joint_execution_and_counterexamples']
    assert not report['gates']['deterministic_reset_and_replay']
