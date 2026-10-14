"""Grader test suite: does the answer comparison judge equivalent answers right and wrong ones wrong?

For every bundle, the oracle's expected answer on the original world is mutated one graded leaf at a
time and scored with the platform's own comparison (answers.score, with what the user knows, as when
serving free dialogue):
  equivalent variants must score 1   int <-> float, numeric strings with "$" or another number of decimals,
                                     case and spacing of free strings, ISO dates in words,
                                     reversed arrays (reported apart: equivalent only when order means nothing)
  wrong variants must score below 1  a number off by one, another enum value, another row id of the same
                                     table (or one character changed), an array missing an item
  silence must score 0               the field's structure with every leaf null (an agent that said nothing)
A miss is a comparison gap: a false negative teaches the student nothing (or an odd format), a false
positive pays for a wrong answer.

    python experiments/v3/grader_suite.py output/v3_bench_pool/build/*/bundle [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from env_factory_v3.core import answers  # noqa: E402

NUMERIC = re.compile(r'^\$?-?\d+(\.\d+)?$')
ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def leaves(spec, value, path=()):
    """(path, leaf spec, value) for every graded scalar or array leaf of a value."""
    if (spec or {}).get('graded', True) is False:
        return
    kind = (spec or {}).get('type')
    if kind == 'object' and isinstance(value, dict) and spec.get('properties'):
        for name, child in spec['properties'].items():
            if name in value:
                yield from leaves(child, value[name], path + (name,))
        return
    if kind == 'array' and isinstance(value, list):
        yield path, spec, value
        for index, item in enumerate(value):
            if (spec.get('items') or {}).get('type') == 'object':
                yield from leaves(spec['items'], item, path + (index,))
        return
    yield path, spec, value


def put(value, path, new):
    value = copy.deepcopy(value)
    if not path:
        return new
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new
    return value


def row_ids(tables):
    out = {}
    for name, rows in tables.items():
        ids = [r.get('id') for r in rows if isinstance(r, dict) and isinstance(r.get('id'), str)]
        for i in ids:
            out[i] = [j for j in ids if j != i]
    return out


def equivalents(spec, value):
    kind = spec.get('type')
    out = []
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, int) and kind in ('integer', 'number'):
        out.append(('int as float', float(value)))
    if isinstance(value, str):
        if NUMERIC.match(value):
            number = value.lstrip('$')
            out.append(('numeric string with $', '$' + number))
            out.append(('numeric string, other decimals', number + '0' if '.' in number else number + '.00'))
        elif 'enum' not in spec:
            if value.upper() != value:
                out.append(('free string upper case', value.upper()))
            out.append(('free string extra spaces', '  ' + value.replace(' ', '  ') + ' '))
            if ISO_DATE.match(value):
                day = datetime.date.fromisoformat(value)
                out.append(('ISO date in words', day.strftime('%b %d, %Y')))
    if isinstance(value, list) and len(value) >= 2 and spec.get('order', 'exact') == 'exact':
        out.append(('reversed array (order exact)', list(reversed(value))))
    return out


def wrongs(spec, value, ids):
    kind = spec.get('type')
    out = []
    if isinstance(value, bool):
        out.append(('boolean flipped', not value))
    elif isinstance(value, (int, float)):
        out.append(('number off by one', value + 1))
    elif isinstance(value, str):
        if spec.get('enum') and len(spec['enum']) > 1:
            out.append(('another enum value', next(e for e in spec['enum'] if e != value)))
        elif value in ids and ids[value]:
            out.append(('another row id', ids[value][0]))
        elif NUMERIC.match(value):
            number = float(value.lstrip('$'))
            out.append(('numeric string off by one', str(number + 1)))
        elif value:
            out.append(('one character changed', value[:-1] + ('x' if value[-1] != 'x' else 'y')))
    elif isinstance(value, list) and value:
        out.append(('array missing an item', value[:-1]))
    return out


def check(folder):
    from env_factory_v3.certify import bundle
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    if loaded.get('oracle') is None:
        return None
    request, schema = loaded['request'], loaded['schema']
    expected = loaded['oracle'].expected(copy.deepcopy(request['tables']), copy.deepcopy(request['user_facts']),
                                         request['time']['start'])['answer']
    known = answers.known_strings(request)
    ids = row_ids(request['tables'])
    found = []
    for name, spec in answers.graded_fields(schema).items():
        if name not in expected:
            continue
        for path, leaf_spec, value in leaves(spec, expected[name]):
            for label, variant in equivalents(leaf_spec, value):
                got = answers.score(spec, expected[name], put(expected[name], path, variant), known)
                if got < 1:
                    found.append({'kind': 'false_negative', 'case': label, 'field': '.'.join(map(str, (name,) + path)),
                                  'expected': value, 'variant': variant, 'score': round(got, 3)})
            for label, variant in wrongs(leaf_spec, value, ids):
                got = answers.score(spec, expected[name], put(expected[name], path, variant), known)
                if got >= 1:
                    internal = answers.internal(leaf_spec, value, known)
                    found.append({'kind': 'false_positive', 'case': label + (' (internal-id exemption)' if internal else ''),
                                  'field': '.'.join(map(str, (name,) + path)), 'expected': value, 'variant': variant})
        silent = silence(spec, expected[name])
        if expected[name] is not None and silent != expected[name]:
            got = answers.score(spec, expected[name], silent, known)
            if got > 0:
                found.append({'kind': 'false_positive', 'case': 'silence (every leaf null) earns credit',
                              'field': name, 'score': round(got, 3)})
    return found


def silence(spec, value):
    """The same structure with every leaf null: an agent that said nothing about this field."""
    if isinstance(value, dict) and (spec or {}).get('properties'):
        return {key: silence(spec['properties'].get(key), item) for key, item in value.items()}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bundles', nargs='+')
    parser.add_argument('--json', help='write every finding here')
    args = parser.parse_args()
    totals, bundles_hit, rows, broken = collections.Counter(), collections.Counter(), {}, []
    checked = 0
    for folder in args.bundles:
        try:
            found = check(folder)
        except Exception as exc:  # noqa: BLE001 - a bundle that will not load is listed, not fatal
            broken.append(f'{folder}: {type(exc).__name__}: {exc}'[:200])
            continue
        if found is None:
            continue
        checked += 1
        rows[folder] = found
        for kind_case in {(f['kind'], f['case']) for f in found}:
            bundles_hit[kind_case] += 1
        for f in found:
            totals[(f['kind'], f['case'])] += 1
    print(f'{checked} bundles checked, {len(broken)} not loadable')
    for (kind, case), count in sorted(totals.items()):
        print(f'  {kind:15s} {case:45s} leaves {count:5d}  bundles {bundles_hit[(kind, case)]:4d}')
    if args.json:
        Path(args.json).write_text(json.dumps({'findings': rows, 'broken': broken}, ensure_ascii=False, indent=1,
                                              default=str))


if __name__ == '__main__':
    main()
