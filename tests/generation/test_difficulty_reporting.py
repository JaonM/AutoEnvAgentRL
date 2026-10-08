from datetime import datetime, timezone
import json

from env_factory.generation.difficulty_reporting import summarize_difficulty


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def test_parent_and_blueprint_feedback_include_retries_without_guessing_provider_failure(tmp_path):
    directory = sample(tmp_path, 1, 'standard', failed=True, seconds=100)
    (directory / 'execution_progress.jsonl').unlink()
    report = directory / 'code_agent_generation.json'
    report.write_text(json.dumps({'attempts': [
        {'phase_events': [{'phase': 'independent_review', 'status': 'failed', 'completed_at': 1001}]},
        {'phase_events': [{'phase': 'blueprint_validation', 'status': 'failed', 'completed_at': 1030}]},
        {'actionable_feedback_at': 1050}, {'actionable_feedback_at': 999},
        {'actionable_feedback_at': 1101}, {'actionable_feedback_at': True}]}))
    retry = directory / 'generation_retries' / 'attempt-2' / 'code_agent_generation.json'
    retry.parent.mkdir(parents=True)
    retry.write_text(json.dumps({'attempts': [{'actionable_feedback_at': 1020}]}))
    result = summarize_difficulty([directory], tmp_path, requested=1, wall_seconds=100)
    row = result['samples'][0]
    assert row['reservation_to_first_recorded_feedback_seconds'] == 20
    assert row['first_recorded_feedback'] == {'seconds': 20, 'timestamp': 1020,
        'source': 'parent_validation', 'file': 'generation_retries/attempt-2/code_agent_generation.json'}
    retry.write_text(json.dumps({'attempts': [{'phase_events': [
        {'phase': 'compile_and_execution', 'status': 'failed', 'completed_at': 1002}]}]}))
    row = summarize_difficulty([directory], tmp_path, requested=1, wall_seconds=100)['samples'][0]
    assert row['reservation_to_first_recorded_feedback_seconds'] == 30
    assert row['first_recorded_feedback']['source'] == 'blueprint_validation'
    report.write_text(json.dumps({'attempts': [None, {'phase_events': None}, {'status': 'failed',
        'error': 'CODE_AGENT_TIMEOUT', 'candidate_ready_at': 1001}]}))
    row = summarize_difficulty([directory], tmp_path, requested=1, wall_seconds=100)['samples'][0]
    assert row['reservation_to_first_recorded_feedback_seconds'] is None


def sample(root, number, profile, *, ready=False, failed=False, pending=False, seconds=30):
    directory = root / 'artifacts' / f'artifact-{number}' / 'task'
    directory.mkdir(parents=True)
    manifest = {'status': 'failed' if failed else 'reserved' if pending else 'completed',
                'reserved_at': stamp(1000), 'completed_at': stamp(1000 + seconds)}
    if profile is not None:
        manifest['difficulty_profile'] = profile
    (directory / 'sample_manifest.json').write_text(json.dumps(manifest))
    if not failed and not pending:
        (directory / 'task.json').write_text(json.dumps({'task': 'control', 'complexity': 'standard'}))
        output = directory.parent / 'sandbox'
        output.mkdir()
        (output / 'pipeline_result.json').write_text(json.dumps({'passed': ready, 'training_ready': ready,
                                                               'build_completed_at': stamp(1000 + seconds)}))
    (directory / 'execution_progress.jsonl').write_text('\n'.join(json.dumps({
        'status': 'actionable_feedback', 'timestamp': when}) for when in [900, 1010, 1000 + seconds + 1]))
    return directory


def test_frozen_profiles_include_failed_pending_and_untracked_denominators(tmp_path):
    directories = [sample(tmp_path, 1, 'standard', ready=True),
                   sample(tmp_path, 2, 'standard', failed=True, seconds=90),
                   sample(tmp_path, 3, 'complex_write_transactions', ready=True, seconds=60),
                   sample(tmp_path, 4, 'long_horizon_planning', pending=True)]
    result = summarize_difficulty(directories, tmp_path / 'sandbox', requested=5, wall_seconds=120)
    assert result['planned_denominator_valid']
    assert result['groups']['all']['training_ready_yield'] == 2 / 5
    assert result['groups']['all']['terminal'] == 3
    assert result['groups']['all']['pending_or_untracked'] == 2
    assert result['groups']['standard']['training_ready_yield'] == .5
    assert result['groups']['standard']['reservation_to_terminal_seconds'] == {'observed': 2, 'median': 60, 'max': 90}
    assert result['groups']['standard']['reservation_to_first_recorded_feedback_seconds'] == {'observed': 2, 'median': 10, 'max': 10}
    assert result['groups']['all']['ready_per_cohort_wall_hour'] == 60
    assert sum(group['requested'] for name, group in result['groups'].items() if name != 'all') == 5
    assert result['groups']['untracked']['training_ready_yield'] == 0
    assert result['groups']['long_horizon_planning']['reservation_to_terminal_seconds']['observed'] == 0


def test_missing_timestamps_and_historical_profiles_are_not_invented(tmp_path):
    directory = sample(tmp_path, 1, None, ready=True)
    manifest = json.loads((directory / 'sample_manifest.json').read_text())
    manifest.pop('reserved_at')
    (directory / 'sample_manifest.json').write_text(json.dumps(manifest))
    result = summarize_difficulty([directory], tmp_path / 'sandbox', requested=1, wall_seconds=0)
    assert 'standard' not in result['groups']
    assert result['groups']['unassigned']['training_ready_yield'] == 1
    assert result['groups']['all']['reservation_to_terminal_seconds'] == {'observed': 0, 'median': None, 'max': None}
    assert result['groups']['all']['ready_per_cohort_wall_hour'] is None


def test_duplicate_directories_do_not_inflate_and_excess_samples_invalidate_yield(tmp_path):
    first = sample(tmp_path, 1, 'standard', ready=True)
    second = sample(tmp_path, 2, 'standard', ready=True)
    result = summarize_difficulty([first, first], tmp_path / 'sandbox', requested=1, wall_seconds=20)
    assert result['groups']['all']['training_ready'] == 1
    result = summarize_difficulty([first, second], tmp_path / 'sandbox', requested=1, wall_seconds=20)
    assert result['planned_denominator_valid'] is False
    assert result['groups']['all']['training_ready_yield'] is None


def test_failed_generation_cannot_inherit_a_stale_passing_build(tmp_path):
    directory = sample(tmp_path, 1, 'standard', ready=True)
    manifest = json.loads((directory / 'sample_manifest.json').read_text())
    manifest['status'] = 'failed'
    (directory / 'sample_manifest.json').write_text(json.dumps(manifest))
    result = summarize_difficulty([directory], tmp_path / 'sandbox', requested=1, wall_seconds=20)
    assert result['groups']['all']['training_ready'] == 0
