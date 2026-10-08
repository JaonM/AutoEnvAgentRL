"""Trace authored business obligations to observable inputs and executable evidence.

This validates references, not the truth/completeness of natural-language claims.
Independent semantic review and paired execution remain separate obligations.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping


def resolve_pointer(document, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('requirement input must use an absolute JSON pointer')
    current = document
    for raw in pointer[1:].split('/'):
        key = raw.replace('~1', '/').replace('~0', '~')
        try:
            if isinstance(current, list):
                if not key.isdigit() or (len(key) > 1 and key.startswith('0')):
                    raise ValueError('array index must be a canonical nonnegative integer')
                current = current[int(key)]
            else:
                current = current[key]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ValueError(f'requirement input pointer does not resolve: {pointer}') from exc
    return current


def validate_requirement_coverage(source: dict, *, required=False) -> dict:
    rows = source.get('business_requirements')
    if rows is None and not required:
        return {'version': '1.0', 'status': 'not_declared', 'requirements': []}
    if not isinstance(rows, list) or not rows:
        raise ValueError('BUSINESS_REQUIREMENTS: provide a nonempty business_requirements list')
    outcomes = {m['id'] for m in source['metrics'] if m.get('category') == 'outcome'}
    tools = {t['function']['name'] for t in source['tools']}
    scenarios = {s['scenario_id']: s for s in source['scenarios']}
    identifiers, covered = set(), set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('BUSINESS_REQUIREMENTS: each requirement must be an object')
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError('BUSINESS_REQUIREMENTS: requirement IDs must be nonempty and unique')
        identifiers.add(identifier)
        if not isinstance(row.get('statement'), str) or not row['statement'].strip():
            raise ValueError(f'BUSINESS_REQUIREMENTS/{identifier}: missing business statement')
        inputs = row.get('input_refs')
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f'BUSINESS_REQUIREMENTS/{identifier}: missing input_refs')
        for pointer in inputs:
            resolve_pointer(source, pointer)
        for field, known, nonempty in (('outcome_ids', outcomes, True), ('tools', tools, False)):
            values = row.get(field)
            if (not isinstance(values, list) or (nonempty and not values)
                    or any(not isinstance(v, str) or v not in known for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f'BUSINESS_REQUIREMENTS/{identifier}: invalid {field}; '
                                 f'expected unique declared identifiers from {sorted(known)}')
        covered.update(row['outcome_ids'])
        for field, kind in (('positive_scenarios', 'goal_success'), ('negative_scenarios', 'goal_failure')):
            values = row.get(field)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(v, str) or scenarios.get(v, {}).get('kind') != kind for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f'BUSINESS_REQUIREMENTS/{identifier}: invalid {field}; '
                    f'expected nonempty unique {kind} scenario IDs; declared candidates='
                    f'{sorted(name for name, case in scenarios.items() if case.get("kind") == kind)}')
    if covered != outcomes:
        raise ValueError(f'BUSINESS_REQUIREMENTS: uncovered outcomes {sorted(outcomes - covered)}')
    return {'version': '1.0', 'status': 'structurally_valid',
            'semantic_completeness_verified': False, 'paired_execution_verified': False,
            'requirements': copy.deepcopy(rows)}


def bind_scenarios(coverage, scenarios, script_ids=()):
    """Resolve compiler-owned interaction expansion without prefix guessing."""
    if coverage['status'] == 'not_declared':
        return coverage
    result = copy.deepcopy(coverage)
    known = {s['scenario_id'] for s in scenarios}
    for requirement in result['requirements']:
        for field in ('positive_scenarios', 'negative_scenarios'):
            resolved = []
            for original in requirement[field]:
                candidates = [original, *(f'{original}-{script}' for script in script_ids)]
                matches = [name for name in candidates if name in known]
                if not matches:
                    raise ValueError(f'BUSINESS_REQUIREMENTS: scenario disappeared during compilation: {original}')
                resolved.extend(matches)
            requirement[field] = list(dict.fromkeys(resolved))
    return result


def score_requirement_pairs(coverage, metrics, observations):
    """Measure labeled requirement/scenario pairs; missing evidence fails closed.

    Negatives must discriminate at least one bound outcome (normalized <= .2).
    This measures authored probes, not accuracy on arbitrary policy trajectories.
    Infrastructure failures and missing components are inconclusive, never passes.
    """
    import math
    if coverage['status'] == 'not_declared':
        return {'status': 'not_declared', 'passed': False, 'pairs': []}
    ranges = {m['id']: m['score_range'] for m in metrics}
    pairs = []
    for requirement in coverage['requirements']:
        for polarity, field in (('positive', 'positive_scenarios'), ('negative', 'negative_scenarios')):
            for scenario in requirement[field]:
                evidence = observations.get(scenario, {})
                scores = {}
                complete = evidence.get('kind') == 'completed'
                for metric_id in requirement['outcome_ids']:
                    value = evidence.get('reward', {}).get('components', {}).get(metric_id)
                    low, high = ranges[metric_id]
                    valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and high > low and low <= value <= high)
                    complete = complete and valid
                    if valid:
                        scores[metric_id] = (value - low) / (high - low)
                classified = (all(value >= 1 - 1e-9 for value in scores.values()) if polarity == 'positive'
                              else any(value <= .2 + 1e-9 for value in scores.values())) if complete else None
                pairs.append({'requirement_id': requirement['id'], 'scenario_id': scenario,
                              'polarity': polarity, 'normalized_outcomes': scores,
                              'passed': classified, 'status': 'measured' if complete else 'inconclusive'})
    def statistics(polarity):
        selected = [p for p in pairs if p['polarity'] == polarity]
        measured = [p for p in selected if p['status'] == 'measured']
        errors = sum(p['passed'] is False for p in measured)
        return {'declared': len(selected), 'measured': len(measured),
                'inconclusive': len(selected) - len(measured), 'errors': errors,
                'error_rate': errors / len(measured) if measured else None}
    passed = bool(pairs) and all(p['passed'] is True for p in pairs)
    return {'version': '1.0', 'status': 'passed' if passed else 'failed', 'passed': passed,
            'scope': 'authored_requirement_scenario_pairs_not_population_accuracy',
            'false_negative': statistics('positive'), 'false_positive': statistics('negative'),
            'pairs': pairs}
