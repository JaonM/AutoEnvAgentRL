"""Summarize and seal a completed V2 round without changing its measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    protocol = read(root / 'protocol.json')
    baseline = read(root / 'baseline.json')
    samples = baseline['samples']
    expected_mix = {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}
    if (protocol['planned'] != 20 or protocol['mix'] != expected_mix
            or baseline['planned'] != 20 or baseline['terminal'] != 20
            or len(samples) != 20 or {s['id'] for s in samples} != set(range(1, 21))
            or Counter(s['stratum'] for s in samples) != expected_mix
            or any(s['status'] not in {'passed', 'failed'} for s in samples)):
        raise ValueError('A complete 20-sample round with 4:4:4:8 coverage is required')
    good, first_pass, repaired, recovered = [], [], [], []
    failures, phase_seconds = [], defaultdict(float)
    for sample in samples:
        number = sample['id']
        request = root / f'request-{number:02d}.json'
        if digest(request) != sample['request_sha256']:
            raise ValueError(f'Request changed: {number}')
        if (sample['status'] == 'passed') != (sample['qualified'] is True):
            raise ValueError(f'Status and qualification disagree: {number}')
        progress_path = root / f'sample-{number:02d}' / 'progress.json'
        progress = read(progress_path) if progress_path.exists() else {}
        attempts = progress.get('attempts', [])
        if any(a.get('status') == 'running' for a in attempts):
            raise ValueError(f'Unfinished attempt: {number}')
        for attempt in attempts:
            for phase, seconds in attempt.get('phases', {}).items():
                phase_seconds[phase] += seconds
        if len(attempts) > 1:
            repaired.append(number)
        if sample['qualified']:
            delivery_path = root / f'sample-{number:02d}' / 'sandbox' / 'delivery.json'
            delivery = read(delivery_path)
            expected_gates = {'structural_contract', 'joint_execution_and_counterexamples',
                              'independent_business_challenge', 'deterministic_reset_and_replay',
                              'container_execution'}
            container = delivery.get('container') or {}
            execution = container.get('execution') or {}
            if (sample.get('exit_code') != 0 or progress.get('qualified') is not True
                    or delivery.get('qualified') is not True
                    or delivery.get('platform_identity') != protocol['platform_identity']
                    or set(delivery.get('gates', {})) != expected_gates
                    or any(value is not True for value in delivery['gates'].values())
                    or execution.get('passed') is not True
                    or execution.get('http_api', {}).get('passed') is not True
                    or container.get('bundle_identity') != delivery.get('bundle_identity')
                    or progress.get('bundle_identity') != delivery.get('bundle_identity')):
                raise ValueError(f'Qualification evidence incomplete: {number}')
            good.append(number)
            (first_pass if len(attempts) == 1 else recovered).append(number)
        else:
            failures.append({'id': number, 'title': sample['title'],
                             'failure_class': sample.get('failure_class'),
                             'attempts': [{'index': a['index'], 'phase': a.get('failed_phase'),
                                           'error': a.get('error'),
                                           'evidence': f'sample-{number:02d}/attempt-{a["index"]}'}
                                          for a in attempts]})
    if baseline['qualified'] != len(good) or baseline['end_to_end_yield'] != len(good) / 20:
        raise ValueError('Recorded aggregate disagrees with verified sample results')
    groups = {}
    for name, planned in expected_mix.items():
        selected = [s for s in samples if s['stratum'] == name]
        count = sum(s['id'] in good for s in selected)
        groups[name] = {'planned': planned, 'qualified': count, 'yield': count / planned,
                        'mean_seconds': statistics.mean(s['seconds'] for s in selected)}
    native = [read(path)['provenance'] for path in sorted(root.glob('sample-*-native/*.json'))]
    return {'scope': protocol['scope'], 'platform_identity': protocol['platform_identity'],
            'planned': 20, 'qualified': len(good), 'yield': len(good) / 20,
            'target_reached': len(good) >= 18, 'groups': groups,
            'first_pass': len(first_pass), 'repair_attempted': len(repaired),
            'repair_recovered': len(recovered), 'repair_recovered_ids': recovered,
            'wall_seconds': baseline['wall_seconds'],
            'mean_sample_seconds': statistics.mean(s['seconds'] for s in samples),
            'median_sample_seconds': statistics.median(s['seconds'] for s in samples),
            'phase_seconds': dict(phase_seconds),
            'recorded_host_native_completions': len(native),
            'recorded_host_native_seconds': sum(p['seconds'] for p in native),
            'terminal_failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=Path)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    root = args.round.resolve()
    analysis = summarize(root)
    if args.seal:
        if (root / 'sealed_manifest.json').exists():
            raise ValueError('Round is already sealed; refuse to overwrite')
        with (root / 'measurement_analysis.json').open('x') as stream:
            stream.write(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')
        files = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*'))
                 if path.is_file() and '__pycache__' not in path.parts
                 and path.suffix != '.pyc' and path.name != 'sealed_manifest.json'}
        with (root / 'sealed_manifest.json').open('x') as stream:
            stream.write(json.dumps({'scope': 'completed_round_evidence', 'files': files},
                                    ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
