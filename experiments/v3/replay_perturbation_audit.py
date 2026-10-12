"""Replay only the perturbation audit on finished V3 builds (real codex auditor, no rebuild).

  .venv/bin/python experiments/v3/replay_perturbation_audit.py output/v3_m1_<ts>/v2-cross-15 [...]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from env_factory_v3.authoring import perturbations, prompts
from env_factory_v3.authoring.codex import CodexRunner  # noqa: E402
from env_factory_v3.core import spec  # noqa: E402
from env_factory_v3.core.sandbox import load_module  # noqa: E402


def replay(build, runner):
    request = json.loads((build / 'request.json').read_text())
    env_path = next(p for p in (build / 'bundle' / 'env.py', build / 'arbiter' / 'env.py') if p.exists())
    schema = json.loads((build / 'schema' / 'answer_schema.json').read_text())
    gates = build / 'bundle' / 'gates.json'
    excluded = json.loads(gates.read_text()).get('excluded_worlds') or [] if gates.exists() else []
    found = perturbations.insensitive(load_module(env_path, role='env'), request, excluded=excluded)
    if not found:
        return {'build': build.name, 'insensitive': []}
    folder = build / 'perturbation_audit_replay'
    shutil.rmtree(folder, ignore_errors=True)
    spec.materialize(request, folder, schema)
    (folder / 'perturbations.json').write_text(json.dumps(found, ensure_ascii=False, indent=1, default=str))
    record = runner(folder, prompts.PERTURBATION_AUDITOR.format(python=sys.executable), timeout=300,
                    role='perturbation_audit')
    verdict = perturbations.verdict(folder, found, request)
    cells = {f['world']: [f"{c['table']}.{c['id']}.{c['field']}: {c['original']}->{c['perturbed']}"
                          for c in f['changed_cells']] for f in found}
    return {'build': f'{build.parent.name}/{build.name}', 'seconds': record.get('seconds'),
            'valid': verdict['valid'], 'reason': verdict.get('reason'),
            'decisions': [{'world': d['world'], 'verdict': d['verdict'], 'cells': cells[d['world']],
                           'quote': d.get('brief_quote'), 'why': d.get('reason', '')[:240]}
                          for d in verdict.get('decisions', [])]}


def main(builds):
    runner = CodexRunner()
    for build in builds:
        print(json.dumps(replay(Path(build), runner), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
