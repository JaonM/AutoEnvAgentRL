"""Self-checks run by model sessions inside their working folder.

    python -m env_factory_v3.checks schema   # answer_schema.json is valid
    python -m env_factory_v3.checks env      # env.py loads, reference runs on all worlds, answer fits schema
    python -m env_factory_v3.checks oracle   # oracle.py loads and returns schema-valid results on all worlds
    python -m env_factory_v3.checks diff     # env.py and oracle.py agree everywhere (arbiter)
    python -m env_factory_v3.checks request  # a generated request.json is valid and reads naturally

Prints PASS/FAIL and exits 0 only on PASS. Results also go to check-<kind>.json.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from env_factory_v3 import answers, gates
from env_factory_v3.runtime import InvalidWorld, canonical, run_reference, tool_spec_problems
from env_factory_v3.sandbox import SandboxError, load_module
from env_factory_v3.spec import load
from env_factory_v3.worlds import variants

PERTURBED = 10


def shape_problems(initial, final):
    problems = []
    if set(final) != set(initial):
        problems.append(f'tables changed: {sorted(set(final) ^ set(initial))}')
    for table, rows in initial.items():
        columns = {tuple(sorted(r)) for r in rows if isinstance(r, dict)}
        for row in final.get(table, []):
            if not isinstance(row, dict):
                problems.append(f'{table}: non-object row')
            elif columns and tuple(sorted(row)) not in columns:
                problems.append(f'{table}: row columns {sorted(row)} differ from existing rows')
    return problems


def check_schema(folder):
    path = folder / 'answer_schema.json'
    if not path.exists():
        return ['answer_schema.json is missing'], []
    try:
        schema = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f'answer_schema.json is not JSON: {exc}'], []
    return answers.check_schema(schema), [f'fields: {sorted(schema.get("fields", {}))}']


def check_env(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    env = load_module(folder / 'env.py', role='env')
    tools = env.make_tools()
    problems, notes = [], []
    if not isinstance(tools, dict) or not tools:
        return ['make_tools() must return a non-empty dict'], notes
    problems += tool_spec_problems(tools)
    problems += gates.approval_problems(env, request)
    start = request['time']['start']
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=PERTURBED):
        try:
            first = run_reference(env, data, facts, start=start)
            second = run_reference(env, data, facts, start=start)
        except Exception:
            problems.append(f'{name}: reference raised\n{traceback.format_exc(limit=4)}')
            continue
        if canonical([first['answer'], first['final_state']]) != canonical([second['answer'], second['final_state']]):
            problems.append(f'{name}: reference is not deterministic')
        if first['safety_violations']:
            problems.append(f'{name}: reference attempted a write without valid consent')
        problems += [f'{name}: {p}' for p in answers.answer_problems(schema, first['answer'])]
        problems += [f'{name}: {p}' for p in shape_problems(data, first['final_state'])]
        writes = sum(1 for e in first['trace'] if e.get('write'))
        notes.append(f'{name}: calls={first["tool_calls"]} writes={writes} answer={canonical(first["answer"])[:160]}')
    structure = gates.category(request, env)
    if not structure['passed']:
        problems.append(f'category {request["category"]} does not match the reference trace {structure["structure"]}')
    return problems, notes


def check_oracle(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    oracle = load_module(folder / 'oracle.py', role='oracle')
    problems, notes, valid = [], [], 0
    start = request['time']['start']
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=PERTURBED):
        try:
            first = oracle.expected(json.loads(json.dumps(data)), json.loads(json.dumps(facts)), start)
            second = oracle.expected(json.loads(json.dumps(data)), json.loads(json.dumps(facts)), start)
            json.dumps(first, allow_nan=False)
        except InvalidWorld as exc:
            notes.append(f'{name}: invalid world ({exc})')
            if not name.startswith('perturb'):
                problems.append(f'{name}: the original world and approval flips must be valid')
            continue
        except Exception:
            problems.append(f'{name}: expected() raised\n{traceback.format_exc(limit=4)}')
            continue
        if not (isinstance(first, dict) and {'answer', 'final_state'} <= set(first)):
            problems.append(f'{name}: must return {{"answer": ..., "final_state": ...}}')
            continue
        valid += 1
        if canonical(first) != canonical(second):
            problems.append(f'{name}: not deterministic')
        problems += [f'{name}: {p}' for p in answers.answer_problems(schema, first['answer'])]
        problems += [f'{name}: {p}' for p in shape_problems(data, first['final_state'])]
        notes.append(f'{name}: answer={canonical(first["answer"])[:160]}')
    if valid < 5:
        problems.append(f'only {valid} valid worlds; raise InvalidWorld only for data that violates the business invariants')
    return problems, notes


def excluded_worlds(folder):
    path = Path(folder) / 'excluded_worlds.json'
    return json.loads(path.read_text()) if path.exists() else []


def check_diff(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    env = load_module(folder / 'env.py', role='env')
    oracle = load_module(folder / 'oracle.py', role='oracle')
    worlds = gates.compare_worlds(env, oracle, schema, request, perturbed=20, excluded=excluded_worlds(folder))
    result = gates.differential(worlds)
    (folder / 'mismatches.json').write_text(json.dumps(result['mismatches'], ensure_ascii=False, indent=1, default=str))
    notes = [f'counts: {result["counts"]}']
    problems = [f'{m["world"]}: {m["status"]} fields={m.get("fields")} state_diff={json.dumps(m.get("state_diff"), ensure_ascii=False)[:300]}'
                for m in result['mismatches']]
    if not result['passed'] and not problems:
        problems.append('not enough agreeing worlds (need the original, every flip and at least 5 perturbed)')
    return problems, notes


def check_request(folder):
    from env_factory_v3.generate import check_request as generated
    return generated(folder)


KINDS = {'schema': check_schema, 'env': check_env, 'oracle': check_oracle, 'diff': check_diff,
         'request': check_request}


def run(kind, folder):
    folder = Path(folder)
    try:
        problems, notes = KINDS[kind](folder)
    except (SandboxError, answers.SchemaError, FileNotFoundError, ValueError) as exc:
        problems, notes = [f'{type(exc).__name__}: {exc}'], []
    except Exception:
        problems, notes = [traceback.format_exc(limit=5)], []
    report = {'kind': kind, 'passed': not problems, 'problems': problems[:30], 'notes': notes[:40]}
    (folder / f'check-{kind}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in KINDS:
        print(f'usage: python -m env_factory_v3.checks {{{"|".join(KINDS)}}}')
        return 2
    report = run(argv[0], Path.cwd())
    print('\n'.join(report['notes']))
    if report['passed']:
        print('\nPASS')
        return 0
    print('\nFAIL')
    print('\n'.join(report['problems'][:12]))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
