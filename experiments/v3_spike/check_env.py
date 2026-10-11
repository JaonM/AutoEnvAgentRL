"""Self-check for env.py. Run in the sample folder: python check_env.py"""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))

from v3_runtime import NOW, canonical, run_reference  # noqa: E402
from worlds import variants  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schema_problems(initial, final):
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


def main():
    data = json.loads((HERE / 'data.json').read_text())
    facts = json.loads((HERE / 'facts.json').read_text())
    failures = []
    try:
        env = load(HERE / 'env.py', 'v3_env_under_check')
        tools = env.make_tools()
        if not tools or not any(getattr(fn, '__v3_write__', False) for fn in tools.values()):
            failures.append('make_tools() must return read tools and at least one @write tool')
    except Exception:
        print('FAIL: env.py does not load\n' + traceback.format_exc())
        return 1
    summary = []
    for name, world, world_facts in variants(data, facts):
        try:
            first = run_reference(env, world, world_facts, NOW)
            second = run_reference(env, world, world_facts, NOW)
        except Exception:
            failures.append(f'{name}: reference raised\n{traceback.format_exc(limit=4)}')
            continue
        if canonical([first['answer'], first['final_state']]) != canonical([second['answer'], second['final_state']]):
            failures.append(f'{name}: reference is not deterministic')
        failures += [f'{name}: {p}' for p in schema_problems(world, first['final_state'])]
        writes = sum(1 for e in first['trace'] if e.get('write'))
        summary.append(f'{name}: writes={writes} answer={canonical(first["answer"])[:160]}')
    print('\n'.join(summary))
    if failures:
        print('\nFAIL')
        print('\n'.join(failures[:12]))
        return 1
    print('\nPASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
