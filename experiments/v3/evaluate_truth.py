"""Evaluate a V3 batch against V2's handwritten oracles (evaluation only; never shown to models).

  .venv/bin/python experiments/v3/evaluate_truth.py output/v3_m1_<timestamp>

For every build: pipeline verdict (training_ready / failure class) and whether the delivered
(or, for failed builds, the last) env.py reaches the truth's final state on the original world
and every approval flip (`truth_correct`, the M1 metric).

Read-only tasks never change state, so the final state alone says nothing about them. The answer
check runs on the original world and every approval flip (the worlds the truth adapter was
validated on): every number and id-like string in the truth answer must occur in the V3 answer. V3 answer schemas name fields differently and phrase enums and booleans
differently, so those are not compared automatically; `answer_samples` keeps both answers side by
side for review, because a different but equivalent answer contract (e.g. a total instead of a
difference) fails this check. `truth_correct_full` = state and answer checks both pass; a
training_ready build that fails it is a false-accept candidate that needs review.

`perturbed_answer_mismatch` is a diagnostic only: the truth adapter was not built for perturbed
worlds (some oracles read fixed task parameters instead of the perturbed rows), so a mismatch there
may be the truth's fault.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments' / 'v3_spike'))

from truth import truth  # noqa: E402  (V2 handwritten oracle adapter)
from env_factory_v3.runtime import run_reference  # noqa: E402
from env_factory_v3.sandbox import load_module  # noqa: E402
from env_factory_v3.runtime import InvalidWorld  # noqa: E402
from env_factory_v3.worlds import state_diff, variants  # noqa: E402

PERTURBED = 10

V2_ROUNDS = ['env_factory_v2_cross_business_baseline_01', 'env_factory_v2_cross_business_regression_02',
             'env_factory_v2_cross_business_regression_03']


def v2_results(sample_id):
    out = []
    for name in V2_ROUNDS:
        path = ROOT / 'output' / name / 'results.json'
        if path.exists():
            for sample in json.loads(path.read_text())['samples']:
                if sample['id'] == sample_id:
                    out.append(sample)
    return out


def leaves(value):
    """Numbers (as floats) and id-like strings of an answer; booleans and prose are skipped."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in leaves(v)]
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str) and value and not any(c.isspace() for c in value) and len(value) <= 40:
        return [value]
    return []


def missing_leaves(want, got):
    pool = leaves(got)
    missing = []
    for leaf in leaves(want):
        if leaf in pool:
            pool.remove(leaf)
        else:
            missing.append(leaf)
    return missing


def check_answers(env, spec, request, perturbed=0):
    worlds, wrong, samples = 0, [], []
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=perturbed):
        if perturbed and not name.startswith('perturb'):
            continue
        try:
            want = truth(spec, data, facts, request['time']['start'])['answer']
        except Exception:
            continue
        try:
            got = run_reference(env, data, facts, start=request['time']['start'])['answer']
        except InvalidWorld:
            continue
        except Exception as exc:
            got = f'{type(exc).__name__}: {exc}'[:200]
        worlds += 1
        missing = missing_leaves(want, got)
        if missing:
            wrong.append({'world': name, 'missing': missing})
        if name == 'original' or missing:
            samples.append({'world': name, 'truth': want, 'v3': got})
    return worlds, wrong, samples[:4]


def evaluate(build):
    status = json.loads((build / 'status.json').read_text())
    request = json.loads((build / 'request.json').read_text())
    sample = request['source']['sample']
    frozen = Path(request['source']['frozen'])
    spec = json.loads((frozen / f'{sample:02d}' / 'acceptance.json').read_text())
    env_path = next((p for p in (build / 'bundle' / 'env.py', build / 'arbiter' / 'env.py', build / 'author' / 'env.py')
                     if p.exists()), None)
    row = {'sample': sample, 'category': request['category'], 'title': request['title'], 'training_ready': status['training_ready'],
           'failure_class': status.get('failure_class'), 'reason': (status.get('reason') or '')[:120],
           'seconds': status.get('elapsed_seconds'), 'calls': len(status.get('calls', [])),
           'repairs_used': status.get('repairs_used')}
    if env_path is None:
        row['truth_correct'] = row['truth_correct_full'] = None
    else:
        try:
            env = load_module(env_path, role='env')
            wrong = []
            for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=0):
                want = truth(spec, data, facts, request['time']['start'])
                got = run_reference(env, data, facts, start=request['time']['start'])
                if state_diff(want['final_state'], got['final_state'], sorted(want['final_state'])):
                    wrong.append(name)
            row['truth_correct'] = not wrong
            row['truth_wrong_worlds'] = wrong
            worlds, answer_wrong, samples = check_answers(env, spec, request)
            row.update(answer_worlds=worlds, answer_correct=bool(worlds) and not answer_wrong,
                       answer_wrong=answer_wrong[:5], answer_samples=samples)
            row['truth_correct_full'] = row['truth_correct'] and row['answer_correct']
            p_worlds, p_wrong, p_samples = check_answers(env, spec, request, perturbed=PERTURBED)
            row.update(perturbed_answer_mismatch=f'{len(p_wrong)}/{p_worlds}', perturbed_wrong=p_wrong[:5],
                       perturbed_samples=[x for x in p_samples if x['world'] != 'original'][:2])
        except Exception as exc:
            row['truth_correct'] = row['truth_correct_full'] = False
            row['truth_error'] = f'{type(exc).__name__}: {exc}'[:200]
    v2 = v2_results(sample)
    row['v2_correct'] = f"{sum(bool(s.get('external_business_qualified')) for s in v2)}/{len(v2)}"
    row['v2_mean_seconds'] = round(sum(s.get('seconds') or 0 for s in v2) / len(v2)) if v2 else None
    return row


def main(batch):
    batch = Path(batch)
    rows = [evaluate(b) for b in sorted(p for p in batch.iterdir() if (p / 'status.json').exists())]
    rows.sort(key=lambda r: r['sample'])
    ready = [r for r in rows if r['training_ready']]
    summary = {'builds': len(rows), 'training_ready': len(ready),
               'training_ready_and_truth_correct': sum(1 for r in ready if r['truth_correct_full']),
               'false_accepts': sum(1 for r in ready if not r['truth_correct_full']),
               'truth_correct_any_stage': sum(1 for r in rows if r['truth_correct_full']),
               'by_category': {}, 'rows': rows}
    for r in rows:
        bucket = summary['by_category'].setdefault(r['category'], {'builds': 0, 'good': 0})
        bucket['builds'] += 1
        bucket['good'] += bool(r['training_ready'] and r['truth_correct_full'])
    (batch / 'truth_evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    lines = ['| # | category | title | training_ready | failure | state ok | answer ok (worlds) | perturbed answer mismatch | s | calls | V2 correct |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    lines += [f"| {r['sample']} | {r['category']} | {r['title']} | {r['training_ready']} | {r['failure_class'] or ''} | "
              f"{r['truth_correct']} | {r.get('answer_correct')} ({r.get('answer_worlds')}) | "
              f"{r.get('perturbed_answer_mismatch', '')} | {r['seconds']} | "
              f"{r['calls']} | {r['v2_correct']} |" for r in rows]
    lines.append(f"\ntraining_ready {summary['training_ready']}/{summary['builds']}, of which truth-correct "
                 f"{summary['training_ready_and_truth_correct']} (false accepts {summary['false_accepts']}).")
    lines.append('good (training_ready and truth-correct) by category: ' + ', '.join(
        f"{c} {v['good']}/{v['builds']}" for c, v in summary['by_category'].items()))
    (batch / 'truth_evaluation.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
