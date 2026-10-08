"""Per-invocation mechanism counts, including failed generation requests."""
import json


def summarize_realism(task_dirs, sandbox_root, *, requested):
    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
    groups = {}
    for directory in task_dirs:
        manifest = read(directory / 'sample_manifest.json')
        task = read(directory / 'task.json')
        assigned = manifest.get('realism_requirements', {}).get('mechanisms', [])
        declared = task.get('realism_requirements', {}).get('mechanisms', [])
        build_path = (directory.parent / 'sandbox' if directory.name == 'task'
                      else sandbox_root / directory.name)
        build = read(build_path / 'pipeline_result.json')
        generation = read(directory / 'pipeline_result.json')
        generated = bool(task) and (not manifest or manifest.get('status') == 'completed')
        labels = sorted(set(assigned) | set(declared)) or ['normal']
        for label in ['all', *labels]:
            counts = groups.setdefault(label, dict(requested=0, generated=0, build_attempted=0,
                passed=0, training_ready=0, infrastructure_failures=0))
            counts['requested'] += 1
            counts['generated'] += int(generated)
            counts['build_attempted'] += int(bool(build))
            counts['passed'] += int(build.get('passed') is True)
            counts['training_ready'] += int(build.get('training_ready') is True)
            counts['infrastructure_failures'] += int((build or generation).get('failure_code') == 'INFRA')
    for counts in groups.values():
        counts['generation_yield'] = counts['generated'] / counts['requested']
        counts['build_pass_rate'] = counts['passed'] / counts['build_attempted'] if counts['build_attempted'] else None
        counts['training_ready_yield'] = counts['training_ready'] / counts['requested']
    return {'version': '1.0', 'scope': 'current_invocation', 'requested': requested,
            'untracked_requests': max(0, requested - len(task_dirs)),
            'groups': groups, 'note': 'Mechanism groups overlap; no attempts means null build pass rate.'}
