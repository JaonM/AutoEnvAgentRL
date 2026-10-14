"""Replay only the branch audit on finished builds (real codex auditor, no rebuild).

  .venv/bin/python experiments/replay_branch_audit.py output/v3_m1_<ts>/v2-cross-09 [...]

For each build: detect refusal cascades on its delivered (or last) env.py, run the auditor in
<build>/audit_replay/ and print the validated verdict.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory.authoring import branches, prompts
from env_factory.authoring.codex import CodexRunner  # noqa: E402
from env_factory.core import spec  # noqa: E402
from env_factory.core.sandbox import load_module  # noqa: E402


def replay(build, runner):
    request = json.loads((build / 'request.json').read_text())
    env_path = next(p for p in (build / 'bundle' / 'env.py', build / 'arbiter' / 'env.py', build / 'author' / 'env.py')
                    if p.exists())
    schema = json.loads((build / 'schema' / 'answer_schema.json').read_text())
    found = branches.cascades(load_module(env_path, role='env'), request)
    if not found:
        return {'build': build.name, 'cascades': []}
    folder = build / 'audit_replay'
    shutil.rmtree(folder, ignore_errors=True)
    spec.materialize(request, folder, schema)
    (folder / 'branches.json').write_text(json.dumps(found, ensure_ascii=False, indent=1))
    record = runner(folder, prompts.AUDITOR.format(python=sys.executable), timeout=300, role='audit')
    return {'build': build.name, 'cascades': [c['world'] for c in found], 'seconds': record.get('seconds'),
            **branches.verdict(folder, found, request)}


def main(builds):
    runner = CodexRunner()
    for build in builds:
        print(json.dumps(replay(Path(build), runner), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
