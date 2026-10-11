"""Reward false-positive / false-negative rates of V3 bundles by answer counterfactuals.

  .venv/bin/python experiments/v3/reward_counterfactuals.py output/v3_fresh20_p1_<ts>

On every served world of every training-ready bundle, starting from the reference run:
- false negative: the reference itself, an `order: any` array shuffled, or an ungraded
  (`graded: false`) string changed must still succeed;
- false positive: one graded leaf changed (number +1, boolean flipped, enum to another value,
  id-like string altered, list shortened) must not succeed; nor may the correct answer with a
  written cell reverted or an untouched cell changed.
Writes reward_counterfactuals.json into the batch folder.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory_v3 import bundle, reward  # noqa: E402
from env_factory_v3.runtime import run_reference  # noqa: E402
from env_factory_v3.worlds import variants  # noqa: E402


def mutations(value, schema, path=()):
    """(path, mutated value, should_still_succeed) for each leaf of an answer value."""
    schema = schema or {}
    if schema.get('graded') is False:
        if isinstance(value, str):
            yield path, value + ' (reworded)', True
        return
    if isinstance(value, dict):
        props = schema.get('properties') or schema.get('fields') or {}
        for key, item in value.items():
            for sub, mutated, ok in mutations(item, props.get(key), path + (key,)):
                yield sub, mutated, ok
        return
    if isinstance(value, list):
        if len(value) > 1 and schema.get('order') == 'any':
            yield path, list(reversed(value)), True
        if value:
            yield path, value[:-1], False
        for index, item in enumerate(value):
            for sub, mutated, ok in mutations(item, schema.get('items'), path + (index,)):
                yield sub, mutated, ok
        return
    if isinstance(value, bool):
        yield path, not value, False
    elif isinstance(value, (int, float)):
        yield path, value + 1, False
    elif isinstance(value, str):
        others = [e for e in schema.get('enum') or [] if e != value]
        yield path, (others[0] if others else value + 'X'), False


def apply(answer, path, mutated):
    answer = copy.deepcopy(answer)
    if not path:
        return mutated
    target = answer
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = mutated
    return answer


def evaluate(folder):
    # Measures the current reward function on the bundle's logic; the platform signature is
    # irrelevant here (qualification is checked elsewhere).
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    request, schema = loaded['request'], loaded['schema']
    served = set(loaded['valid_worlds'])
    start = request['time']['start']
    counts = {'fn_checks': 0, 'fn': 0, 'fp_checks': 0, 'fp': 0, 'state_checks': 0}
    examples = []
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=20):
        if name not in served:
            continue
        expected = loaded['oracle'].expected(copy.deepcopy(data), copy.deepcopy(facts), start)
        result = run_reference(loaded['env'], data, facts, start=start)
        grade = lambda answer: reward.grade(dict(result, answer=answer), expected, schema, initial=data,
                                            reference_calls=result['tool_calls'])
        checks = [((), result['answer'], True)] + list(mutations(result['answer'], schema))
        for path, mutated, ok in checks:
            outcome = grade(apply(result['answer'], path, mutated))
            key = 'fn' if ok else 'fp'
            counts[key + '_checks'] += 1
            if outcome['success'] != ok:
                counts[key] += 1
                if len(examples) < 5:
                    examples.append({'world': name, 'path': list(path), 'expected_success': ok,
                                     'reward': outcome['reward']})
        # State counterfactuals with the correct answer: a missed write or a stray write must fail.
        for state in state_mutations(data, result['final_state']):
            outcome = reward.grade(dict(result, final_state=state), expected, schema, initial=data,
                                   reference_calls=result['tool_calls'])
            counts['fp_checks'] += 1
            counts['state_checks'] += 1
            if outcome['success']:
                counts['fp'] += 1
                if len(examples) < 5:
                    examples.append({'world': name, 'state_mutation': True, 'reward': outcome['reward']})
    return {**counts, 'examples': examples}


def state_mutations(initial, final):
    """The reference's final state with one written cell reverted, or one untouched cell changed."""
    for table, rows in final.items():
        before = {row.get('id'): row for row in initial.get(table, []) if isinstance(row, dict)}
        stray_done = False
        for index, row in enumerate(rows):
            original = before.get(row.get('id'))
            for field, value in row.items():
                if field == 'id':
                    continue
                if original is not None and original.get(field) != value:
                    reverted = copy.deepcopy(final)
                    reverted[table][index][field] = original.get(field)
                    yield reverted
                elif not stray_done and isinstance(value, (int, float)) and not isinstance(value, bool):
                    stray = copy.deepcopy(final)
                    stray[table][index][field] = value + 1
                    stray_done = True
                    yield stray


def main(batch):
    batch = Path(batch)
    rows, total = {}, {'fn_checks': 0, 'fn': 0, 'fp_checks': 0, 'fp': 0, 'state_checks': 0}
    for build in sorted(batch.glob('*/bundle')):
        try:
            row = evaluate(build)
        except ValueError as error:
            row = {'error': str(error)[:200]}
        rows[build.parent.name] = row
        for key in total:
            total[key] += row.get(key, 0)
    summary = {**total, 'fp_rate': total['fp'] / max(1, total['fp_checks']),
               'fn_rate': total['fn'] / max(1, total['fn_checks']), 'bundles': rows}
    (batch / 'reward_counterfactuals.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'bundles'}))
    for name, row in rows.items():
        if row.get('fp') or row.get('fn') or row.get('error'):
            print(name, json.dumps(row, ensure_ascii=False)[:400])


if __name__ == '__main__':
    main(sys.argv[1])
