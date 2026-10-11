"""Differential check between env.py (reference run) and oracle.py on shared worlds.

Run in a folder holding env.py, oracle.py, data.json, facts.json:  python diffcheck.py
Writes mismatches.json and prints a summary. Exit 0 when the two agree everywhere.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_runtime import NOW, run_reference  # noqa: E402
from worlds import state_diff, variants  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(folder, perturbed=20):
    folder = Path(folder)
    data = json.loads((folder / 'data.json').read_text())
    facts = json.loads((folder / 'facts.json').read_text())
    env = load(folder / 'env.py', f'v3_env_{abs(hash(str(folder)))}')
    oracle = load(folder / 'oracle.py', f'v3_oracle_{abs(hash(str(folder)))}')
    mismatches, compared, errors = [], 0, []
    for name, world, world_facts in variants(data, facts, perturbed=perturbed):
        try:
            mine = run_reference(env, world, world_facts, NOW)
        except Exception:
            errors.append({'world': name, 'side': 'env', 'error': traceback.format_exc(limit=3)})
            continue
        try:
            theirs = oracle.expected(copy.deepcopy(world), copy.deepcopy(world_facts), NOW)
        except Exception:
            errors.append({'world': name, 'side': 'oracle', 'error': traceback.format_exc(limit=3)})
            continue
        compared += 1
        diff = state_diff(theirs['final_state'], mine['final_state'], sorted(world))
        if diff:
            mismatches.append({'world': name, 'facts_changed': world_facts != facts,
                               'data': world if name.startswith('perturb') else None,
                               'facts': world_facts, 'state_diff_oracle_vs_env': diff,
                               'env_answer': mine['answer'], 'oracle_answer': theirs['answer'],
                               'env_trace': mine['trace'][-12:]})
    return {'compared': compared, 'mismatches': mismatches, 'errors': errors}


if __name__ == '__main__':
    result = compare(Path.cwd())
    Path('mismatches.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print(f"compared={result['compared']} mismatches={len(result['mismatches'])} errors={len(result['errors'])}")
    for m in result['mismatches'][:5]:
        print(m['world'], json.dumps(m['state_diff_oracle_vs_env'], ensure_ascii=False)[:400])
    raise SystemExit(0 if not result['mismatches'] and not result['errors'] else 1)
