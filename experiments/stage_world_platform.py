"""Stage the next authoring platform without modifying a frozen running experiment.

The source patch is deliberately small and domain-independent. Re-run callers
with PYTHONPATH=<output>/src. The original platform and its checksums stay intact.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def stage(destination):
    source = Path(__file__).resolve().parents[1] / 'src/env_factory'
    target = destination / 'src/env_factory'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = target / 'production/world.py'
    original = path.read_text()
    before = 'isinstance(schema, dict) and 6 <= len(schema) <= 10 and all('
    after = 'isinstance(schema, dict) and len(schema) >= 1 and all('
    if original.count(before) != 1:
        raise ValueError('unexpected world validator; review the migration')
    changed = original.replace(before, after).replace('goal_schema must describe 6 to 10 processes:',
                                                    'goal_schema must describe one or more processes:')
    changed=changed.replace('from env_factory.core import spec',
                            'from env_factory.core import spec\nfrom env_factory.task_contract import approval_problems as task_approval_problems')
    changed=changed.replace("problems.append(f'goal {goal!r}: process must be one of",
                            "problems.append(f'goal_schema: goal {goal!r}: process must be one of")
    for needle in ("        problems += goal_problems(facts.get('goals'), schema)",
                   "    problems += goal_problems(task['user_facts'].get('goals'), world['goal_schema'])"):
        if changed.count(needle)!=1:raise ValueError('unexpected task validation entry point')
        indent=needle[:len(needle)-len(needle.lstrip())]
        changed=changed.replace(needle,needle+'\n'+indent+'problems += task_approval_problems(request)')
    path.write_text(changed)
    identity = {'change': 'Remove authoring-only 6..10 catalogue limit; reject approval facts unrelated to declared goals; preserve existing checks.',
                'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'staged_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                'frozen_source_modified': False}
    (destination / 'migration.json').write_text(json.dumps(identity, indent=2)+'\n')
    return identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().output)))
