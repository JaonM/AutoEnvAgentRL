"""V3 spike: same frozen complex samples, same model, Python representation + differential check.

Usage (repository root, project venv):
  .venv/bin/python experiments/v3_spike/run.py --output output/v3_spike_01
  .venv/bin/python experiments/v3_spike/run.py --output output/v3_spike_dry --dry-run

Per sample: author A writes env.py and oracle B writes oracle.py in parallel, isolated folders
(codex exec, GPT-6.1-Sol, may run its own self-check). Then A-vs-B differential check on
original data, flipped approvals and perturbed worlds; on disagreement one arbiter repair.
Finally evaluation against V2's handwritten oracles (never shown to any model).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src'))

import prompts  # noqa: E402
from diffcheck import compare  # noqa: E402
from truth import expand, truth  # noqa: E402
from v3_runtime import NOW, run_reference  # noqa: E402
from worlds import answer_recall, state_diff, variants  # noqa: E402

MODEL = 'gpt-6.1-sol'
FROZEN = ROOT / 'output' / 'env_factory_v2_cross_business_frozen_01'
COMPLEX = [4, 5, 9, 10, 14, 15, 19, 20]
V2_ROUNDS = ['env_factory_v2_cross_business_baseline_01', 'env_factory_v2_cross_business_regression_02',
             'env_factory_v2_cross_business_regression_03']
RUNTIME_FILES = ['v3_runtime.py', 'worlds.py']


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def codex(folder, prompt, timeout, dry_run=False):
    """One isolated codex exec session; returns timing and token usage."""
    folder = Path(folder).resolve()
    try:
        from env_factory.generation.codex_transport import transport_args
        extra = transport_args()
    except Exception:
        extra = []
    args = ['codex', 'exec', '--ephemeral', '--sandbox', 'workspace-write', '--skip-git-repo-check',
            '--model', MODEL, '--json', '--ignore-user-config', '-c', 'approval_policy="never"',
            '-c', 'web_search="disabled"', '-C', str(folder),
            '--output-last-message', str(folder / 'last_message.txt'), *extra, '-']
    (folder / 'prompt.txt').write_text(prompt)
    if dry_run:
        return {'skipped': True, 'command': args}
    started = time.monotonic()
    with open(folder / 'events.jsonl', 'w') as events, open(folder / 'stderr.txt', 'w') as errors:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                   cwd=folder, text=True, start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
            status = 'exited'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = 'timeout'
    usage = {'input_tokens': 0, 'output_tokens': 0}
    for line in (folder / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = event.get('usage') or (event.get('msg') or {}).get('usage') or {}
        for key in usage:
            if isinstance(found.get(key), int):
                usage[key] += found[key]
    seconds = round(time.monotonic() - started, 1)
    if status == 'exited' and process.returncode != 0 and seconds < 5:
        status = 'infrastructure_error'
    return {'status': status, 'returncode': process.returncode, 'seconds': seconds, 'usage': usage,
            'stderr_tail': (folder / 'stderr.txt').read_text()[-400:]}


def self_check(folder, script):
    run = subprocess.run([sys.executable, script], cwd=folder, capture_output=True, text=True, timeout=300)
    return {'passed': run.returncode == 0, 'tail': (run.stdout + run.stderr)[-1500:]}


def prepare(sample_id, root):
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    sample = next(s for s in manifest['samples'] if s['id'] == sample_id)
    folder = FROZEN / f'{sample_id:02d}'
    business = json.loads((folder / 'business.json').read_text())
    data = expand(business)
    facts = json.loads((folder / 'user.json').read_text())['facts']
    spec = json.loads((folder / 'acceptance.json').read_text())
    text = prompts.brief(sample, business['policy'], data, facts, NOW)
    dirs = {}
    for role, checker in (('author', 'check_env.py'), ('oracle', 'check_oracle.py')):
        target = root / role
        target.mkdir(parents=True)
        for name in RUNTIME_FILES + [checker]:
            shutil.copy(HERE / name, target / name)
        (target / 'brief.md').write_text(text)
        save(target / 'data.json', data)
        save(target / 'facts.json', facts)
        dirs[role] = target
    return sample, data, facts, spec, dirs


def evaluate(env_folder, data, facts, spec):
    """Compare env.py reference runs with V2's handwritten oracle (evaluation only)."""
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(f'eval_env_{abs(hash(str(env_folder)))}',
                                                         env_folder / 'env.py')
    env = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(env)
    worlds, primary_ok, recall = [], True, None
    for name, world, world_facts in variants(data, facts, perturbed=20):
        try:
            want = truth(spec, world, world_facts, NOW)
        except Exception as exc:
            worlds.append({'world': name, 'truth': 'not_applicable', 'reason': str(exc)[:200]})
            continue
        try:
            got = run_reference(env, world, world_facts, NOW)
        except Exception:
            worlds.append({'world': name, 'match': False, 'env_error': traceback.format_exc(limit=2)})
            primary_ok = primary_ok and name.startswith('perturb')
            continue
        diff = state_diff(want['final_state'], got['final_state'], sorted(want['final_state']))
        entry = {'world': name, 'match': not diff, 'diff': diff}
        if name == 'original':
            recall = answer_recall(want['answer'], got['answer'])
            entry.update(truth_answer=want['answer'], env_answer=got['answer'], answer_recall=recall)
        worlds.append(entry)
        if not name.startswith('perturb') and diff:
            primary_ok = False
    perturbed = [w for w in worlds if w['world'].startswith('perturb') and 'match' in w]
    return {'state_correct': primary_ok, 'answer_recall': recall,
            'perturbed_agreement': f"{sum(w['match'] for w in perturbed)}/{len(perturbed)}",
            'worlds': worlds}


def run_sample(sample_id, output, *, repairs, dry_run, timeouts):
    root = output / f'sample-{sample_id:02d}'
    started = time.monotonic()
    result = {'id': sample_id, 'calls': {}}
    try:
        sample, data, facts, spec, dirs = prepare(sample_id, root)
        result.update(title=sample['title'], kind=spec['kind'])
        python = sys.executable
        jobs = {'author': (dirs['author'], prompts.AUTHOR.format(now=NOW, python=python), timeouts['author']),
                'oracle': (dirs['oracle'], prompts.ORACLE.format(now=NOW, python=python), timeouts['oracle'])}
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = {role: pool.submit(codex, *job, dry_run) for role, job in jobs.items()}
            for role, future in futures.items():
                result['calls'][role] = future.result()
        if dry_run:
            result['status'] = 'dry_run'
            return result
        for role, script, filename in (('author', 'check_env.py', 'env.py'), ('oracle', 'check_oracle.py', 'oracle.py')):
            exists = (dirs[role] / filename).exists()
            result[f'{role}_check'] = self_check(dirs[role], script) if exists else {'passed': False, 'tail': 'missing file'}
        if not result['author_check']['passed']:
            result['status'] = 'author_failed'
            return result
        arbiter = root / 'arbiter'
        arbiter.mkdir()
        for name in RUNTIME_FILES + ['check_env.py', 'check_oracle.py', 'diffcheck.py']:
            shutil.copy(HERE / name, arbiter / name)
        for name in ('brief.md', 'data.json', 'facts.json', 'env.py'):
            shutil.copy(dirs['author'] / name, arbiter / name)
        final_env = dirs['author']
        if result['oracle_check']['passed']:
            shutil.copy(dirs['oracle'] / 'oracle.py', arbiter / 'oracle.py')
            diff = compare(arbiter)
            save(arbiter / 'mismatches.json', diff)
            result['diff_before_repair'] = {'compared': diff['compared'], 'mismatches': len(diff['mismatches']),
                                            'errors': len(diff['errors'])}
            if (diff['mismatches'] or diff['errors']) and repairs > 0:
                result['calls']['arbiter'] = codex(arbiter, prompts.ARBITER.format(python=python),
                                                   timeouts['arbiter'])
                after = compare(arbiter)
                result['diff_after_repair'] = {'compared': after['compared'], 'mismatches': len(after['mismatches']),
                                               'errors': len(after['errors'])}
                if self_check(arbiter, 'check_env.py')['passed']:
                    final_env = arbiter
            result['diff_clean'] = not (result.get('diff_after_repair') or result['diff_before_repair'])['mismatches']
        else:
            result['diff_clean'] = None
        result['final_env'] = str(final_env.relative_to(output))
        # Pipeline verdict uses only A/B agreement; truth evaluation is separate.
        result['pipeline_accepts'] = bool(result['diff_clean'])
        evaluation = evaluate(final_env, data, facts, spec)
        save(root / 'evaluation.json', evaluation)
        result['state_correct'] = evaluation['state_correct']
        result['answer_recall'] = evaluation['answer_recall']
        result['perturbed_agreement'] = evaluation['perturbed_agreement']
        if result['oracle_check']['passed']:
            oracle_folder = arbiter if result.get('calls', {}).get('arbiter') else dirs['oracle']
            result['oracle_state_correct'] = oracle_correct(oracle_folder, data, facts, spec)
        result['status'] = 'evaluated'
    except Exception:
        result['status'] = 'harness_error'
        result['error'] = traceback.format_exc(limit=5)
    finally:
        result['seconds'] = round(time.monotonic() - started, 1)
        save(root / 'result.json', result)
    return result


def oracle_correct(folder, data, facts, spec):
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(f'eval_oracle_{abs(hash(str(folder)))}', folder / 'oracle.py')
    oracle = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(oracle)
        for name, world, world_facts in variants(data, facts, perturbed=0):
            want = truth(spec, world, world_facts, NOW)
            got = oracle.expected(copy.deepcopy(world), copy.deepcopy(world_facts), NOW)
            if state_diff(want['final_state'], got['final_state'], sorted(want['final_state'])):
                return False
        return True
    except Exception:
        return False


def v2_reference():
    rows = {}
    for name in V2_ROUNDS:
        path = ROOT / 'output' / name / 'results.json'
        if not path.exists():
            continue
        for sample in json.loads(path.read_text())['samples']:
            if sample['id'] in COMPLEX:
                rows.setdefault(sample['id'], []).append({
                    'round': name.replace('env_factory_v2_cross_business_', ''),
                    'qualified': sample.get('qualified'),
                    'external_business_qualified': sample.get('external_business_qualified'),
                    'failed_phase': sample.get('failed_phase'), 'seconds': round(sample.get('seconds') or 0),
                    'model_calls': sample.get('model_calls')})
    return rows


def summarize(results, output):
    v2 = v2_reference()
    lines = ['# V3 spike vs V2 on the 8 frozen complex cross-business samples', '',
             '| # | title | V3 self-check | A/B diff clean | V3 state correct | answer recall | perturbed | V3 s | V2 correct (3 rounds) | V2 s (mean) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(results, key=lambda r: r['id']):
        rounds = v2.get(r['id'], [])
        v2_ok = sum(bool(x['external_business_qualified']) for x in rounds)
        v2_seconds = round(sum(x['seconds'] for x in rounds) / len(rounds)) if rounds else '-'
        lines.append(f"| {r['id']} | {r.get('title', '')} | {r.get('author_check', {}).get('passed')} | "
                     f"{r.get('diff_clean')} | {r.get('state_correct')} | {r.get('answer_recall')} | "
                     f"{r.get('perturbed_agreement')} | {r.get('seconds')} | {v2_ok}/{len(rounds)} | {v2_seconds} |")
    correct = sum(bool(r.get('state_correct')) for r in results)
    accepted = [r for r in results if r.get('pipeline_accepts')]
    false_accept = sum(1 for r in accepted if not r.get('state_correct'))
    lines += ['', f'V3 state-correct: {correct}/{len(results)}; pipeline accepted (A/B agree): {len(accepted)}, '
              f'of which wrong vs truth: {false_accept}.',
              'Scope: business correctness only (no user simulator, Docker, reward gates); time frozen at '
              f'{NOW}s. V2 numbers are external business acceptance from its full pipeline.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    save(output / 'summary.json', {'results': results, 'v2': v2, 'v3_state_correct': correct,
                                   'pipeline_accepted': len(accepted), 'false_accepts': false_accept})
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', default=','.join(map(str, COMPLEX)))
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--repairs', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--author-timeout', type=int, default=600)
    parser.add_argument('--oracle-timeout', type=int, default=480)
    parser.add_argument('--arbiter-timeout', type=int, default=600)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit('output directory must be fresh')
    output.mkdir(parents=True)
    if not args.dry_run and shutil.which('codex') is None:
        raise SystemExit('codex CLI not found on PATH')
    ids = [int(x) for x in args.samples.split(',')]
    timeouts = {'author': args.author_timeout, 'oracle': args.oracle_timeout, 'arbiter': args.arbiter_timeout}
    save(output / 'protocol.json', {'model': MODEL, 'samples': ids, 'repairs': args.repairs, 'now': NOW,
                                    'parallel': args.parallel, 'timeouts': timeouts, 'frozen': str(FROZEN)})
    if not args.dry_run:
        probe = output / 'preflight'
        probe.mkdir()
        (probe / 'note.txt').write_text('preflight\n')
        check = codex(probe, 'Read note.txt and reply with its single word. Do not modify files.', 180)
        save(output / 'preflight.json', check)
        if check['status'] != 'exited' or check['returncode'] != 0:
            raise SystemExit(f'codex preflight failed: {check}')
        print(f"codex preflight ok in {check['seconds']}s")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda i: run_sample(i, output, repairs=args.repairs, dry_run=args.dry_run,
                                                     timeouts=timeouts), ids))
    print(f'wall seconds: {time.monotonic() - started:.0f}')
    summarize(results, output)


if __name__ == '__main__':
    main()
