"""Tool fuzzing: the authored tools off the reference path.

Differential qualification only exercises the calls the reference makes. A learner (and the policy
gate's strong model) makes many others, and a tool that crashes, behaves nondeterministically or
corrupts the tables there turns exploration into noise (in the world pilot most such failures were
off-path). This stage needs no model: every tool is called many times with arguments drawn from the
database (ids and values of matching columns), from other tables, and edge values, singly on fresh
copies and in random sequences on one copy. A call may refuse (BusinessError); it must not:
- crash (any other exception) on arguments of the declared types,
- give a different result or state for the same call on the same state (each run with fresh tools),
- keep state outside the tables (the same call on the same tables answers differently the second time
  with the same tool objects, e.g. a call history kept in a closure, which the oracle cannot see),
- return something that is not JSON,
- leave a table with rows whose columns changed or ids that are no longer unique.

    python -m env_factory.certify.fuzz BUNDLE [--calls 40] [--walks 20]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import traceback
from pathlib import Path

from env_factory.runtime import BusinessError, _parameters, canonical, is_write, json_type, jsonable

RESULT = 'fuzz.json'


def value_pool(tables):
    """{column name: [values]} over every table, plus all strings and numbers seen."""
    columns, strings, numbers = {}, set(), set()
    for table, rows in tables.items():
        for row in rows if isinstance(rows, list) else []:
            for key, value in row.items() if isinstance(row, dict) else []:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    columns.setdefault(key, []).append(value)
                    columns.setdefault(f'{table.rstrip("s")}_{key}', []).append(value)
                if isinstance(value, str):
                    strings.add(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers.add(value)
    return columns, sorted(strings), sorted(numbers)


def argument(rng, name, schema, pool):
    columns, strings, numbers = pool
    kind = (schema or {}).get('type')
    kind = next((k for k in kind if k != 'null'), None) if isinstance(kind, list) else kind
    matching = [v for v in columns.get(name, []) + columns.get(name.removesuffix('_id'), []) if v is not None]
    if kind == 'string':
        candidates = [v for v in matching if isinstance(v, str)] or strings or ['x']
        return rng.choice(candidates) if rng.random() < 0.85 else rng.choice(['', 'UNKNOWN-0', rng.choice(strings or ['x'])])
    if kind in ('integer', 'number'):
        candidates = [v for v in matching if isinstance(v, (int, float)) and not isinstance(v, bool)] or numbers or [1]
        value = rng.choice(candidates) if rng.random() < 0.8 else rng.choice([0, -1, 10 ** 6])
        return int(value) if kind == 'integer' else value
    if kind == 'boolean':
        return rng.random() < 0.5
    if kind == 'array':
        return [argument(rng, name.removesuffix('s'), schema.get('items') or {'type': 'string'}, pool)
                for _ in range(rng.randint(0, 3))]
    if kind == 'object':
        return {}
    return rng.choice(strings or ['x'])


def calls_for(rng, fn, pool):
    """Keyword arguments for one call: every required parameter, optional ones half the time."""
    args = {}
    for parameter in _parameters(fn):
        if parameter.default is not parameter.empty and rng.random() < 0.5:
            continue
        schema = json_type(parameter.annotation) if parameter.annotation is not parameter.empty else None
        args[parameter.name] = argument(rng, parameter.name, schema, pool)
    return args


def run(fn, db, now, args):
    """('ok', result) | ('refused', message) | ('crash', traceback)."""
    try:
        return 'ok', jsonable(copy.deepcopy(fn(db, now, **copy.deepcopy(args))))
    except BusinessError as exc:
        return 'refused', str(exc)[:200]
    except Exception:  # noqa: BLE001 - the authored tool failed
        return 'crash', traceback.format_exc(limit=3)[-600:]


def shape(tables):
    return {t: sorted({canonical(sorted(r)) for r in rows if isinstance(r, dict)}) for t, rows in tables.items()
            if isinstance(rows, list)}


def invariant_problems(before_shape, tables):
    problems = []
    for table, rows in tables.items():
        if not isinstance(rows, list):
            continue
        keys = {canonical(sorted(r)) for r in rows if isinstance(r, dict)}
        if before_shape.get(table) and not keys <= set(before_shape[table]):
            problems.append(f'{table}: rows with new or missing columns')
        ids = [r.get('id') for r in rows if isinstance(r, dict) and 'id' in r]
        if len(ids) != len(set(map(canonical, ids))):
            problems.append(f'{table}: duplicate ids')
    return problems


def fuzz(env, tables, now, *, calls=40, walks=20, walk_length=8, seed=0):
    tools = env.make_tools()
    pool, rng = value_pool(tables), random.Random(seed)
    before = shape(tables)
    report = {name: {'calls': 0, 'ok': 0, 'refused': 0, 'crash': 0, 'nondeterministic': 0, 'hidden_state': 0,
                     'not_json': 0, 'examples': []} for name in tools}

    def note(name, kind, args, detail):
        entry = report[name]
        entry[kind] += 1
        if len(entry['examples']) < 3:
            entry['examples'].append({'kind': kind, 'args': args, 'detail': detail})

    for name, fn in sorted(tools.items()):
        for _ in range(calls):
            args = calls_for(rng, fn, pool)
            report[name]['calls'] += 1
            first_db, second_db, again_db = (copy.deepcopy(tables) for _ in range(3))
            first = run(env.make_tools()[name], first_db, now, args)
            second = run(env.make_tools()[name], second_db, now, args)
            if first[0] == 'crash':
                note(name, 'crash', args, first[1])
                continue
            if canonical([first, first_db]) != canonical([second, second_db]):
                note(name, 'nondeterministic', args, None)
                continue
            run(fn, copy.deepcopy(tables), now, args)  # the same tool objects, called once before
            if canonical([run(fn, again_db, now, args), again_db]) != canonical([first, first_db]):
                note(name, 'hidden_state', args, None)
                continue
            try:
                json.dumps(first[1], allow_nan=False)
            except (TypeError, ValueError):
                note(name, 'not_json', args, None)
                continue
            report[name][first[0]] += 1
            for problem in invariant_problems(before, first_db) if is_write(fn) else []:
                note(name, 'crash', args, problem)
    sequences = []
    names = sorted(tools)
    for _ in range(walks):
        tools = env.make_tools()  # one episode's tools
        db, steps = copy.deepcopy(tables), []
        for _ in range(walk_length):
            name = rng.choice(names)
            args = calls_for(rng, tools[name], pool)
            kind, detail = run(tools[name], db, now, args)
            steps.append([name, kind])
            problems = invariant_problems(before, db)
            if kind == 'crash' or problems:
                sequences.append({'steps': steps, 'detail': detail if kind == 'crash' else problems})
                break
    crashes = sum(r['crash'] + r['nondeterministic'] + r['hidden_state'] + r['not_json'] for r in report.values())
    return {'passed': crashes == 0 and not sequences, 'defects': crashes + len(sequences),
            'tools': report, 'sequence_failures': sequences[:5]}


def check(folder, **options):
    from env_factory.certify import bundle
    folder = Path(folder)
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    request = loaded['request']
    report = fuzz(loaded['env'], request['tables'], request['time']['start'], **options)
    (folder / RESULT).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bundles', nargs='+')
    parser.add_argument('--calls', type=int, default=40)
    parser.add_argument('--walks', type=int, default=20)
    args = parser.parse_args()
    for folder in args.bundles:
        try:
            report = check(folder, calls=args.calls, walks=args.walks)
            kinds = ('crash', 'nondeterministic', 'hidden_state', 'not_json')
            bad = {n: {k: r[k] for k in kinds if r[k]} for n, r in report['tools'].items() if any(r[k] for k in kinds)}
            print(json.dumps({'bundle': folder, 'passed': report['passed'], 'defects': report['defects'], 'tools': bad},
                             ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - keep going over many bundles
            print(json.dumps({'bundle': folder, 'error': f'{type(exc).__name__}: {exc}'[:300]}))


if __name__ == '__main__':
    main()
