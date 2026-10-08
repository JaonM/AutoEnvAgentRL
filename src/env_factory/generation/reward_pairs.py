"""Bounded positive variations of references; original necessity checks stay intact."""
from __future__ import annotations

import copy
import json
import math


def derive_reward_pairs(artifacts, definitions=None, *, max_probes=8):
    references = [s for s in artifacts['acceptance_contract']['executable_scenarios']
                  if s.get('kind') == 'goal_success']
    tools = {item['tool_name']: item for item in artifacts['tool_implementations']}
    # Explicit mappings are produced before this helper by the compiler; exact IDs only.
    known = {s['scenario_id']: s for s in references}
    definitions = [] if definitions is None else definitions
    if not isinstance(definitions, list) or len(definitions) > max_probes:
        raise ValueError(f'REWARD_PAIRS: at most {max_probes} expanded declarative probes are supported')
    pairs, ids = [], set()
    for definition in definitions:
        if (not isinstance(definition, dict) or not isinstance(definition.get('id'), str)
                or not definition['id'] or definition['id'] in ids):
            raise ValueError('REWARD_PAIRS: probe IDs must be nonempty and unique')
        ids.add(definition['id'])
        reference = known.get(definition.get('reference_id'))
        if reference is None:
            raise ValueError('REWARD_PAIRS: reference_id must name a successful compiled scenario')
        steps = copy.deepcopy(reference['steps'])
        kind = definition.get('kind')
        if kind == 'equivalent_answer':
            if set(definition) != {'id', 'reference_id', 'kind', 'answer'} or not isinstance(definition['answer'], str):
                raise ValueError('REWARD_PAIRS: equivalent_answer requires a string answer')
            answers = [step for step in steps if step.get('operation') == 'agent_response']
            if not answers or answers[-1]['content'] == definition['answer']:
                raise ValueError('REWARD_PAIRS: answer probe must change the terminal answer')
            answers[-1]['content'] = definition['answer']
        elif kind == 'readonly_recovery':
            if set(definition) != {'id', 'reference_id', 'kind', 'before_tool', 'calls'}:
                raise ValueError('REWARD_PAIRS: unsupported recovery fields')
            calls = definition['calls']
            if not isinstance(calls, list) or not 1 <= len(calls) <= 3:
                raise ValueError('REWARD_PAIRS: recovery needs one to three read calls')
            for call in calls:
                if (not isinstance(call, dict) or set(call) != {'tool_name', 'arguments'}
                        or not isinstance(call['arguments'], dict)
                        or tools.get(call['tool_name'], {}).get('operation') not in {'select', 'aggregate_count'}):
                    raise ValueError('REWARD_PAIRS: recovery can only add read-only calls without captures')
            positions = [i for i, step in enumerate(steps)
                         if step.get('operation') == 'tool_call' and step.get('tool_name') == definition['before_tool']]
            if len(positions) != 1:
                raise ValueError('REWARD_PAIRS: before_tool must identify exactly one call')
            steps[positions[0]:positions[0]] = [dict(copy.deepcopy(call), operation='tool_call', expected_status=200)
                                                for call in calls]
        else:
            raise ValueError('REWARD_PAIRS: unsupported probe kind')
        pairs.append({'id': definition['id'], 'reference_id': reference['scenario_id'],
                      'kind': kind, 'steps': steps, 'provenance': 'authored'})
    # An automatically equivalent JSON serialization cannot alter business facts.
    for reference in references:
        answer = next((s for s in reversed(reference['steps']) if s.get('operation') == 'agent_response'), None)
        if answer:
            try:
                parsed = json.loads(answer['content'])
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, (dict, list)):
                continue
            alternate = json.dumps(parsed, ensure_ascii=False, indent=3, sort_keys=True)
            if alternate == answer['content']:
                alternate = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            if alternate == answer['content']:
                continue
            steps = copy.deepcopy(reference['steps'])
            next(s for s in reversed(steps) if s.get('operation') == 'agent_response')['content'] = alternate
            pairs.append({'id': 'platform-json-format:' + reference['scenario_id'],
                          'reference_id': reference['scenario_id'], 'kind': 'equivalent_answer',
                          'steps': steps, 'provenance': 'platform_json_value_preserving'})
    lifecycle = artifacts.get('business_lifecycle', {})
    static_readonly = (artifacts.get('environment_plan', {}).get('mode') == 'reference_data'
                       and not lifecycle)
    if static_readonly:
        for reference in references:
            first = next((step for step in reference['steps'] if step.get('operation') == 'tool_call'), None)
            if not first or tools.get(first['tool_name'], {}).get('operation') not in {'select', 'aggregate_count'}:
                continue
            if first.get('expected_status', 200) != 200:
                continue
            steps = copy.deepcopy(reference['steps'])
            position = next(i for i, step in enumerate(steps) if step.get('operation') == 'tool_call')
            extra = {'operation': 'tool_call', 'tool_name': first['tool_name'],
                     'arguments': copy.deepcopy(first['arguments']), 'expected_status': 200}
            steps.insert(position, extra)
            pairs.append({'id': 'platform-repeat-read:' + reference['scenario_id'],
                          'reference_id': reference['scenario_id'], 'kind': 'readonly_recovery',
                          'steps': steps, 'provenance': 'platform_static_readonly_repeat'})
    return pairs


def run_reward_pairs(pairs, references, execute):
    """Execute fresh baselines and variants; execute returns reward and final state."""
    baseline = {}
    results = []
    by_id = {case['scenario_id']: case for case in references}
    for pair in pairs:
        try:
            identity = pair['reference_id']
            if identity not in baseline:
                baseline[identity] = execute(by_id[identity]['steps'])
            original = baseline[identity]
            alternate = execute(pair['steps'])
            if any(isinstance(item['reward'], bool) or not isinstance(item['reward'], (int, float))
                   or not math.isfinite(item['reward']) for item in (original, alternate)):
                raise ValueError('reward pair execution returned a non-finite or nonnumeric score')
            same_state = original['state'] == alternate['state']
            valid_reference = original['reward'] >= 1 - 1e-9
            passed = valid_reference and same_state and alternate['reward'] >= 1 - 1e-9
            results.append({k: pair[k] for k in ('id', 'reference_id', 'kind', 'provenance')} | {
                'status': 'measured' if valid_reference and same_state else 'inconclusive',
                'passed': passed, 'reference_reward': original['reward'],
                'variant_reward': alternate['reward'], 'same_business_state': same_state})
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            results.append({'id': pair['id'], 'kind': pair['kind'], 'status': 'inconclusive',
                            'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    measured = [r for r in results if r['status'] == 'measured']
    errors = sum(not r['passed'] for r in measured)
    return {'version': '1.0', 'scope': 'declared_and_value_preserving_positive_variants',
            'status': 'not_applicable' if not results else 'passed' if all(r['passed'] for r in results) else 'failed',
            'passed': bool(results) and all(r['passed'] for r in results),
            'measured': len(measured), 'inconclusive': len(results) - len(measured),
            'false_negatives': errors, 'false_negative_rate': errors / len(measured) if measured else None,
            'results': results}
