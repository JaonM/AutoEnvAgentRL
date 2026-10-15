"""Mine comparison gaps from re-graded teacher episodes (regrade_teacher.py rows carry the expected and the
extracted answer): every graded leaf that did not match is classified as

  dropped_said     extracted null although the expected value was said in some form (a verify or extractor gap)
  format           extracted a value equal to the expected one up to typography (case, punctuation, rounding,
                   time notation): a comparison gap
  silent           extracted null and the value was never said (the teacher did not deliver it)
  wrong            extracted a different value (the teacher said something else, or the extractor misread)
  expected_null    the expected value is null (nothing to report) but a value was extracted (0, false, "none")

The matcher here is deliberately liberal (English number words, percentages, case and punctuation, 12-hour
times); its job is to find candidates, which are then read by hand.

    python experiments/grading_gaps.py output/v3_I2/teacher_pool3_regraded.jsonl [--examples 5]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from env_factory.core.extract import date_said  # noqa: E402

WORDS = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
         'seventeen eighteen nineteen twenty').split()


def loose(value):
    return re.sub(r'[^0-9a-z一-鿿]', '', str(value).casefold())


def said_loosely(text, value):
    if isinstance(value, bool) or value is None:
        return False
    lowered = text.casefold()
    if isinstance(value, (int, float)):
        forms = {f'{value:g}', f'{value:,}' if isinstance(value, int) else f'{value:g}'}
        if isinstance(value, int):
            forms |= {f'{value / 100:g}', f'{value / 100:.2f}', f'{value / 100:,.2f}'}
            if 0 <= value < len(WORDS):
                forms.add(WORDS[value])
        if isinstance(value, float) and value < 1:
            forms.add(f'{value * 100:g}%')
        forms |= {f'{round(float(value), 2):g}', f'{round(float(value), 1):g}'}
        return any(re.search(rf'(?<![\d.]){re.escape(f)}(?![\d])' if f[0].isdigit() else rf'\b{re.escape(f)}\b',
                             lowered) for f in forms)
    if isinstance(value, str):
        if date_said(text, value):
            return True
        clock = re.fullmatch(r'(\d{1,2}):(\d{2})', value.strip())
        if clock:
            hour, minute = int(clock[1]), clock[2]
            twelve = hour % 12 or 12
            suffix = 'pm' if hour >= 12 else 'am'
            pattern = rf'\b{twelve}(?::{minute})?\s*{suffix}\b' if minute == '00' else rf'\b{twelve}:{minute}\s*{suffix}\b'
            if re.search(pattern, lowered.replace('.', '')):
                return True
        return bool(loose(value)) and loose(value) in loose(text)
    return False


def same_loosely(expected, actual):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return abs(float(expected) - float(actual)) <= max(0.006, 0.005 * abs(float(expected)))
    if isinstance(expected, str) and isinstance(actual, str):
        return loose(expected) == loose(actual)
    return False


def leaves(expected, actual, path):
    if isinstance(expected, dict):
        for key, value in expected.items():
            yield from leaves(value, (actual or {}).get(key) if isinstance(actual, dict) else None, path + (key,))
    elif isinstance(expected, list) and expected and isinstance(expected[0], (dict, list)):
        for index, value in enumerate(expected):
            other = actual[index] if isinstance(actual, list) and index < len(actual) else None
            yield from leaves(value, other, path + (index,))
    else:
        yield path, expected, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('rows')
    parser.add_argument('--examples', type=int, default=4)
    args = parser.parse_args()
    counts, examples, by_type = collections.Counter(), collections.defaultdict(list), collections.Counter()
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    rows = [r for r in rows if r.get('expected_answer') is not None]
    for row in rows:
        speech = '\n'.join(m.get('content') or '' for m in row['messages'] if m.get('role') == 'assistant')
        extracted = row.get('extracted_answer') or {}
        for field, score in (row.get('field_scores') or {}).items():
            if score == 1.0:
                continue
            for path, value, got in leaves(row['expected_answer'].get(field), extracted.get(field), (field,)):
                if value == got:
                    continue
                if value is None:
                    kind = 'expected_null'  # nothing to report, yet a value was extracted (0, false, "none"...)
                elif isinstance(value, list) and isinstance(got, list) and sorted(map(str, value)) == sorted(map(str, got)):
                    kind = 'format'
                elif got is None:
                    kind = 'dropped_said' if said_loosely(speech, value) else 'silent'
                else:
                    kind = 'format' if same_loosely(value, got) else 'wrong'
                type_name = type(value if value is not None else got).__name__
                counts[kind] += 1
                by_type[(kind, type_name)] += 1
                if len(examples[(kind, type_name)]) < args.examples:
                    examples[(kind, type_name)].append(f"{row['id']} s{row['seed']} {'.'.join(map(str, path))}: "
                                                       f"expected {value!r} got {got!r}")
    print(f"{len(rows)} episodes with answers, {sum(r['success'] for r in rows)} successes")
    for kind, count in counts.most_common():
        print(f'{kind:13s} {count:5d}  ' + ', '.join(f'{t} {n}' for (k, t), n in by_type.most_common() if k == kind))
    for (kind, type_name), lines in sorted(examples.items()):
        if kind in ('dropped_said', 'format', 'expected_null'):
            print(f'\n[{kind} / {type_name}]')
            for line in lines:
                print('  ' + line[:220])


if __name__ == '__main__':
    main()
