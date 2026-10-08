import json
from env_factory.generation.realism_reporting import summarize_realism


def test_mechanism_rates_include_generation_failures_without_inflating_builds(tmp_path):
    directories = []
    for index in (1, 2):
        path = tmp_path / 'task' / f'task-{index}'
        path.mkdir(parents=True)
        directories.append(path)
        (path / 'sample_manifest.json').write_text(json.dumps({
            'status': 'completed' if index == 1 else 'failed',
            'realism_requirements': {'version': '1.0', 'mechanisms': ['fault_recovery']}}))
    (directories[0] / 'task.json').write_text(json.dumps({'realism_requirements': {
        'version': '1.0', 'mechanisms': ['fault_recovery', 'external_change']}}))
    output = tmp_path / 'sandbox/task-1'
    output.mkdir(parents=True)
    (output / 'pipeline_result.json').write_text(json.dumps({'passed': True, 'training_ready': False}))
    (directories[1] / 'pipeline_result.json').write_text(json.dumps({'failure_code': 'INFRA'}))
    result = summarize_realism(directories, tmp_path / 'sandbox', requested=3)
    assert result['untracked_requests'] == 1
    assert result['groups']['fault_recovery']['generation_yield'] == .5
    assert result['groups']['fault_recovery']['build_pass_rate'] == 1
    assert result['groups']['fault_recovery']['training_ready_yield'] == 0
    assert result['groups']['fault_recovery']['infrastructure_failures'] == 1
    assert result['groups']['external_change']['requested'] == 1
