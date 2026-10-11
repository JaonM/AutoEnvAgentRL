"""Self-check for oracle.py. Run in the sample folder: python check_oracle.py"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_runtime import NOW, canonical  # noqa: E402
from worlds import variants  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    data = json.loads((HERE / 'data.json').read_text())
    facts = json.loads((HERE / 'facts.json').read_text())
    try:
        oracle = load(HERE / 'oracle.py', 'v3_oracle_under_check')
    except Exception:
        print('FAIL: oracle.py does not load\n' + traceback.format_exc())
        return 1
    failures, summary = [], []
    for name, world, world_facts in variants(data, facts):
        try:
            first = oracle.expected(copy.deepcopy(world), copy.deepcopy(world_facts), NOW)
            second = oracle.expected(copy.deepcopy(world), copy.deepcopy(world_facts), NOW)
            json.dumps(first, allow_nan=False)
        except Exception:
            failures.append(f'{name}: expected() raised\n{traceback.format_exc(limit=4)}')
            continue
        if not (isinstance(first, dict) and set(first) >= {'answer', 'final_state'}):
            failures.append(f'{name}: must return {{"answer": ..., "final_state": ...}}')
            continue
        if canonical(first) != canonical(second):
            failures.append(f'{name}: not deterministic')
        if set(first['final_state']) != set(world):
            failures.append(f'{name}: final_state must contain exactly the input tables')
        summary.append(f'{name}: answer={canonical(first["answer"])[:160]}')
    print('\n'.join(summary))
    if failures:
        print('\nFAIL')
        print('\n'.join(failures[:12]))
        return 1
    print('\nPASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
