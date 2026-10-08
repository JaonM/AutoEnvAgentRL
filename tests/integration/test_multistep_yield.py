from scripts.experiments.measure_multistep_yield import summarize


def test_failed_generation_stays_in_denominator():
    rows = [dict(status='passed', training_ready=True)] * 8
    rows += [dict(status='failed', error_type='TimeoutError'), dict(status='failed', failure_code='INFRA')]
    result = summarize(rows)
    assert result['end_to_end_yield'] == .8
    assert not result['target_met']


def test_incomplete_or_build_only_success_cannot_meet_target():
    for last in [dict(status='pending'), dict(status='running'), dict(status='passed', training_ready=False)]:
        result = summarize([dict(status='passed', training_ready=True)] * 8 + [last])
        assert not result['target_met']
        if last['status'] != 'passed':
            assert result['end_to_end_yield'] is None


def test_exact_ninety_percent_is_counted_with_scope():
    result = summarize([dict(status='passed', training_ready=True)] * 9 + [dict(status='failed')])
    assert result['target_met']
    assert result['end_to_end_yield'] == .9
    assert result['scope'] == 'fresh_authoring_on_frozen_requests_not_population_yield'
    assert not summarize([])['target_met']
