"""Compare sealed, complete V2 rounds only when workloads and budgets match."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from env_factory_v2.contract import require
from measure_v2_natural import read, digest

MATCH_FIELDS = ('assets_identity', 'planned', 'mix', 'model', 'concurrency', 'author_tools',
    'candidate_repairs', 'base_image', 'bridge_timeout_seconds', 'sample_timeout_seconds',
    'policy_turn_budget', 'policy_calls_per_turn', 'user_mode', 'measurement',
    'qualification', 'acceptance_budget')


def load_round(root):
    root = Path(root).resolve()
    files = read(root / 'sealed_manifest.json')['files']
    require({'protocol.json', 'results.json'} <= set(files), 'round protocol and results must be sealed')
    require(all(not Path(path).is_absolute() and (root/path).resolve().is_relative_to(root)
                for path in files), 'sealed evidence must remain inside its root')
    require(all(digest(root/path) == expected for path, expected in files.items()), 'sealed round changed')
    return read(root/'protocol.json'), read(root/'results.json')


def round_metrics(protocol, result):
    planned = protocol['planned']; samples = result['samples']
    require(type(planned) is int and planned > 0 and result['planned'] == planned
            and result['terminal'] == planned and len(samples) == planned, 'round is not complete')
    require(len({s['id'] for s in samples}) == planned
            and all(s['status'] not in {'pending', 'running'} for s in samples), 'invalid terminal sample set')
    def duration(value):
        require(type(value) in {int, float} and math.isfinite(value) and value >= 0, 'invalid duration')
        return value
    wall = duration(result['wall_seconds'])
    require(wall > 0, 'round wall time must be positive')
    qualified = []
    for sample in samples:
        duration(sample['seconds'])
        if sample.get('qualified') is True:
            require(sample['status'] == 'passed' and all(sample.get(gate) is True for gate in
                ('sandbox_qualified', 'external_business_qualified', 'training_ready', 'policy_success')),
                'qualified sample is missing a delivery gate')
            qualified.append(sample)
    groups = {}
    for group, count in protocol['mix'].items():
        selected = [s for s in samples if s['stratum'] == group]
        require(len(selected) == count and count > 0, 'stratum mix changed')
        groups[group] = sum(s in qualified for s in selected) / count
    require(sum(protocol['mix'].values()) == planned, 'stratum denominator mismatch')
    return {'wall_seconds': wall, 'yield': len(qualified) / planned, 'groups': groups,
            'median_sample_seconds': median(s['seconds'] for s in samples),
            'sample_keys': sorted((s['id'], s['stratum'], s['raw_request']) for s in samples)}


def compare(before_protocol, before_result, after_protocol, after_result):
    reasons = [f'{key} differs or is missing' for key in MATCH_FIELDS
               if key not in before_protocol or key not in after_protocol
               or before_protocol[key] != after_protocol[key]]
    try:
        before, after = round_metrics(before_protocol, before_result), round_metrics(after_protocol, after_result)
        if before['sample_keys'] != after['sample_keys']:
            reasons.append('sample identities, requests or difficulty strata differ')
    except (ValueError, KeyError, TypeError) as error:
        reasons.append(str(error))
    if reasons:
        return {'comparable': False, 'reasons': reasons, 'wall_reduction': None,
                'scope': 'no_speed_or_yield_comparison_claim'}
    reduction = 1 - after['wall_seconds'] / before['wall_seconds']
    no_regression = after['yield'] >= before['yield'] and all(
        after['groups'][group] >= before['groups'][group] for group in before['groups'])
    return {'comparable': True, 'wall_reduction': reduction,
            'yield_before': before['yield'], 'yield_after': after['yield'],
            'yield_change_percentage_points': 100 * (after['yield'] - before['yield']),
            'groups_before': before['groups'], 'groups_after': after['groups'],
            'speed_target_met_without_stratum_yield_regression': reduction >= 0.30 and no_regression,
            'scope': 'matched_workload_observation_not_statistical_causality_or_unseen_generalization',
            'limitations': ['Infrastructure and model latency variation remain possible.',
                           'Repeated known workloads do not establish held-out generalization.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = load_round(args.before), load_round(args.after)
    report = compare(*before, *after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['comparable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
