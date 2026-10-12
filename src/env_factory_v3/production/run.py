"""End-to-end run: domains -> requests -> sandboxes -> (images) -> (policy gate) -> training manifest.

    python -m env_factory_v3 run --output DIR --count 30 (--split holdout-4 --domains 12 | --domains-file F)
                                 [--rubric-share 0.2] [--parallel 2] [--repairs 2] [--docker]
                                 [--policy-k 4] [--policy-user-mode structured|natural] [--eval-share 0.0]

Every stage keeps its output under DIR and is skipped when already done, so running the same
command again resumes. Builds that failed for infrastructure reasons are retried on resume, and
the earlier attempt is kept as <id>.attempt-N.

DIR/tasks.json tells trainable tasks from the rest:
  "tasks"     only the trainable tasks, in the exact format rl reads
              ({"id", "sandbox", "split", "weight"}): train with `--tasks DIR/tasks.json`;
              with --eval-share, whole domains are marked "eval" (never part of a domain);
  "excluded"  every task that must not be used for training, with the reason;
  "details"   everything known about every task (category, reward type, domain, user modes,
              image, policy gate...);
  "summary"   counts.
A task is trainable when its build passed every gate, its image was built and verified (with
--docker), and the policy gate's model solved it at least once (with --policy-k > 0).
DIR/report.json and DIR/report.md summarise the run.

Exit code: 0 when at least one task is trainable, 1 when none is, 2 on configuration or
infrastructure errors (no Neo4j for --split, no codex, every build failed on infrastructure).
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import time
from pathlib import Path

from env_factory_v3.authoring import pipeline
from env_factory_v3.certify import bundle
from env_factory_v3.core import spec
from env_factory_v3.production import openings

STAGES = ('domains', 'generate', 'build', 'docker', 'policy', 'manifest')


class RunError(Exception):
    """Configuration or infrastructure problem: exit code 2."""


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str))


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stage_domains(out, options, log):
    path = out / 'domains.json'
    if path.exists():
        log('domains', 'reused', path)
        return _read(path)
    if options.get('domains_file'):
        chosen = json.loads(Path(options['domains_file']).read_text())
    else:
        from env_factory_v3.production import domains
        try:
            graph = options['graph']() if callable(options.get('graph')) else options['graph']
        except Exception as exc:
            raise RunError(f'scene graph unavailable for --split (is Neo4j running?): {exc}') from exc
        report = domains.propose(graph, options['domain_client'], options['domains'], split=options['split'],
                                 seed=options.get('seed', 0), hops=options.get('hops', 2))
        _write(out / 'domains.report.json', report)
        chosen = report['accepted']
    if not chosen:
        raise RunError('no domains to generate from')
    _write(path, chosen)
    log('domains', 'done', f'{len(chosen)} domains')
    return chosen


def stage_generate(out, domains_, options, log):
    from env_factory_v3.production import generate
    folder = out / 'generate'
    summary = _read(folder / 'generation.json')
    if summary is None:
        if folder.exists():
            shutil.move(str(folder), str(out / f'generate.incomplete-{int(time.time())}'))
        summary = generate.generate(domains_, options['count'], folder, runner=options['runner'],
                                    parallel=options['parallel'], python=options['python'],
                                    rubric_share=options['rubric_share'], opening_client=options['opening_client'],
                                    english_share=options.get('english_share', 0.0))
        log('generate', 'done', f"{summary['kept']}/{summary['requested']} requests")
    else:
        log('generate', 'reused', f"{summary['kept']}/{summary['requested']} requests")
    return summary


def _finished(status):
    return status is not None and 'elapsed_seconds' in status and (status.get('training_ready')
                                                                   or status.get('failure_class'))


def stage_build(out, options, log):
    requests = sorted((out / 'generate' / 'requests').glob('*.json'))
    builds = out / 'build'
    builds.mkdir(exist_ok=True)
    todo = []
    for path in requests:
        request = spec.load(path)
        folder = builds / request['id']
        status = _read(folder / 'status.json')
        if _finished(status) and status.get('failure_class') != 'infrastructure':
            continue
        if folder.exists():  # unfinished or infrastructure failure: keep it, build again
            attempt = 1 + sum(1 for p in builds.glob(f"{request['id']}.attempt-*"))
            shutil.move(str(folder), str(builds / f"{request['id']}.attempt-{attempt}"))
        todo.append(request)

    def one(request):
        return pipeline.build(request, builds / request['id'], runner=options['runner'], repairs=options['repairs'],
                              judge=options.get('judge'))

    with concurrent.futures.ThreadPoolExecutor(options['parallel']) as pool:
        list(pool.map(one, todo))
    statuses = {path.stem: _read(builds / path.stem / 'status.json') for path in requests}
    if statuses and all((s or {}).get('failure_class') == 'infrastructure' for s in statuses.values()):
        raise RunError('every build failed on infrastructure (codex or network); run again to retry')
    log('build', 'done', f"{len(todo)} built, {sum(bool((s or {}).get('training_ready')) for s in statuses.values())}"
                         f"/{len(statuses)} training ready")
    return statuses


def stage_difficulty(out, statuses, options, log):
    """Difficulty floor (generate.MIN_REFERENCE_CALLS): a training-ready bundle whose reference solution
    needs fewer public tool calls than its shape's floor is too easy to teach anything; it gets no image
    and no policy gate."""
    from env_factory_v3.production import generate
    from env_factory_v3.runtime import run_reference
    floors = options.get('min_reference_calls', generate.MIN_REFERENCE_CALLS)
    results = {}
    for task, status in statuses.items():
        if not (status or {}).get('training_ready'):
            continue
        try:
            loaded = bundle.load_bundle(Path(status['bundle']))
            request = loaded['request']
            floor = floors.get((request.get('source') or {}).get('shape'), 0)
            calls = run_reference(loaded['env'], request['tables'], request['user_facts'],
                                  start=request['time']['start'])['tool_calls']
        except Exception as exc:  # later stages report the broken bundle
            results[task] = {'passed': True, 'error': f'{type(exc).__name__}: {exc}'[:300]}
            continue
        results[task] = {'passed': calls >= floor, 'reference_calls': calls, 'floor': floor}
    log('difficulty', 'done', f"{sum(r['passed'] for r in results.values())}/{len(results)} above the floor")
    return results


def stage_docker(out, statuses, options, log):
    from env_factory_v3.certify import docker
    results = {}
    for task, status in statuses.items():
        if not (status or {}).get('training_ready'):
            continue
        folder = Path(status['bundle'])
        try:
            results[task] = {'image': docker.image_for(folder), 'reused': True}
        except ValueError:
            try:
                metadata = docker.build(folder)
                results[task] = {'image': metadata['image_id'], 'reused': False}
            except Exception as exc:
                results[task] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
    log('docker', 'done', f"{sum('image' in r for r in results.values())}/{len(results)} images verified")
    return results


def stage_policy(out, statuses, options, log):
    from env_factory_v3.certify import policy_gate as policy
    k, mode = options['policy_k'], options['policy_user_mode']
    name = policy.RESULT if mode == 'structured' else f'policy-{mode}.json'
    results = {}
    for task, status in statuses.items():
        if not (status or {}).get('training_ready'):
            continue
        folder = Path(status['bundle'])
        saved = _read(folder / name)
        if saved and saved.get('k') == k and saved.get('status') in ('passed', 'unsolved') and saved.get(
                'platform_identity') == bundle.platform_identity():
            results[task] = saved
            continue
        try:
            results[task] = policy.gate(folder, options['policy_client'], k=k, user_mode=mode,
                                        user_client=options.get('user_client'))
        except Exception as exc:  # one bad bundle must not stop the batch; it is simply not trainable
            results[task] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:300]}
    log('policy', 'done', f"{sum(r.get('status') == 'passed' for r in results.values())}/{len(results)} solved")
    return results


def classify(task, status, docker_result, policy_result, options, difficulty=None):
    """(trainable, reason) for one task."""
    if status is None:
        return False, 'not built'
    if not status.get('training_ready'):
        return False, f"build failed: {status.get('failure_class')} - {(status.get('reason') or '')[:160]}"
    if difficulty and not difficulty['passed']:
        return False, (f"too easy: the reference solution needs {difficulty['reference_calls']} tool calls, "
                       f"below the floor of {difficulty['floor']}")
    if options['docker'] and 'image' not in (docker_result or {}):
        return False, f"image not verified: {(docker_result or {}).get('error', 'not built')}"
    if options['policy_k']:
        state = (policy_result or {}).get('status')
        if state == 'unsolved':
            return False, f"policy gate: {policy_result.get('model')} solved 0/{policy_result.get('played')}"
        if state == 'unextractable':
            return False, 'policy gate: the answer cannot be read back from what an agent tells the user'
        if state != 'passed':
            detail = (policy_result or {}).get('error', '')
            return False, f'policy gate not passed: {state or "not run"} {detail}'.strip()
    return True, None


def assign_eval(trainable, details, share):
    """Whole domains go to eval, never part of one: near-duplicate tasks must not straddle train/eval
    (rl rejects that). At least one domain always stays in train."""
    if not share or not trainable:
        return
    family = lambda task: details[task['id']].get('family') or details[task['id']].get('domain') or task['id']
    groups = {}
    for task in trainable:
        groups.setdefault(family(task), []).append(task)
    names = sorted(groups)
    wanted, chosen = share * len(trainable), []
    for name in names[::-1]:  # deterministic: last domains first
        if len(chosen) + 1 >= len(names) or sum(len(groups[n]) for n in chosen) >= wanted:
            break
        chosen.append(name)
    for name in chosen:
        for task in groups[name]:
            task['split'] = 'eval'


def manifest(out, statuses, docker_results, policy_results, generation, options, difficulty=None):
    difficulty = difficulty or {}
    trainable, excluded, details = [], [], {}
    requests = {path.stem: json.loads(path.read_text()) for path in (out / 'generate' / 'requests').glob('*.json')}
    for index, task in enumerate(sorted(statuses)):
        status, request = statuses[task], requests.get(task, {})
        ok, reason = classify(task, status, docker_results.get(task), policy_results.get(task), options,
                              difficulty.get(task))
        gates = _read(Path(status['bundle']) / 'gates.json', {}) if ok else {}
        modes = ['structured'] + (['natural'] if gates.get('natural', {}).get('passed') else [])
        details[task] = {
            'trainable': ok, 'reason': reason, 'category': request.get('category'),
            'reward_type': request.get('reward_type', 'business'), 'title': request.get('title'),
            'opening': request.get('user_request'), 'domain': (request.get('source') or {}).get('generated'),
            'family': (request.get('source') or {}).get('family'),
            'bundle': (status or {}).get('bundle'), 'failure_class': (status or {}).get('failure_class'),
            'reference_calls': (difficulty.get(task) or {}).get('reference_calls'),
            'build_seconds': (status or {}).get('elapsed_seconds'), 'repairs_used': (status or {}).get('repairs_used'),
            'user_modes': modes if ok else [], 'docker': docker_results.get(task),
            'policy': {k: (policy_results.get(task) or {}).get(k) for k in ('model', 'user_mode', 'k', 'successes',
                                                                           'success_rate', 'status')}
            if task in policy_results else None,
            'spec_findings': (status or {}).get('spec_findings')}
        if ok:
            trainable.append({'id': task, 'sandbox': str(Path(status['bundle']).resolve()), 'split': 'train',
                              'weight': 1.0})
        else:
            easy = not (difficulty.get(task) or {'passed': True})['passed']
            excluded.append({'id': task, 'reason': reason,
                             'failure_class': 'too_easy' if easy else (status or {}).get('failure_class')})
    for row in generation.get('rows', []):  # requests that never became a build
        if not row['passed']:
            task = f"gen-{row['index']:03d}"
            excluded.append({'id': task, 'reason': f"generation rejected: {'; '.join(row['problems'])[:200]}",
                             'failure_class': 'request'})
            details.setdefault(task, {'trainable': False, 'reason': excluded[-1]['reason'],
                                      'domain': row.get('domain'), 'category': row.get('category')})
    assign_eval(trainable, details, options['eval_share'])
    summary = {'requested': generation.get('requested'), 'requests': len(requests), 'built': len(statuses),
               'trainable': len(trainable), 'excluded': len(excluded),
               'train': sum(t['split'] == 'train' for t in trainable),
               'eval': sum(t['split'] == 'eval' for t in trainable)}
    value = {'tasks': trainable, 'excluded': excluded, 'summary': summary, 'details': details,
             'note': 'rl reads only "tasks" (trainable tasks); everything in "excluded" must not be trained on'}
    _write(out / 'tasks.json', value)
    if trainable and summary['eval']:
        # rl also rejects text-level near-duplicates across train/eval; keep the manifest loadable.
        try:
            from rl.data.tasks import load_tasks
            load_tasks(manifest=str(out / 'tasks.json'))
        except ImportError:
            pass
        except ValueError as exc:
            for task in trainable:
                task['split'] = 'train'
            summary.update(train=len(trainable), eval=0)
            value['eval_split_dropped'] = f'rl rejected the eval split ({exc}); every trainable task is train'
            _write(out / 'tasks.json', value)
    return value


def report(out, value, timings, options):
    details = value['details'].values()
    group = lambda key: {name: {'total': sum(1 for d in details if d.get(key) == name),
                                'trainable': sum(1 for d in details if d.get(key) == name and d['trainable'])}
                         for name in sorted({str(d.get(key)) for d in details if d.get(key) is not None})}
    failures = {}
    for row in value['excluded']:
        failures[row.get('failure_class') or 'gated'] = failures.get(row.get('failure_class') or 'gated', 0) + 1
    built = [d for d in details if d.get('build_seconds') is not None]
    data = {'summary': value['summary'], 'failure_classes': failures, 'by_category': group('category'),
            'by_reward_type': group('reward_type'), 'by_domain': group('domain'),
            'opening_diversity': openings.diversity([d['opening'] for d in details if d.get('opening')]),
            'mean_build_seconds': round(sum(d['build_seconds'] for d in built) / len(built), 1) if built else None,
            'stage_seconds': timings, 'options': {k: v for k, v in options.items() if isinstance(v, (str, int, float,
                                                                                                      bool, type(None)))}}
    _write(out / 'report.json', data)
    lines = [f"# V3 run: {out.name}", '',
             f"- trainable **{value['summary']['trainable']}** (train {value['summary']['train']}, eval "
             f"{value['summary']['eval']}) / built {value['summary']['built']} / requested "
             f"{value['summary']['requested']}",
             f"- excluded {value['summary']['excluded']}: " + ', '.join(f'{k} {v}' for k, v in failures.items()),
             f"- mean build {data['mean_build_seconds']} s; stages: " + ', '.join(f'{k} {v}s' for k, v in timings.items()),
             '', '| task | category | reward | trainable | reason / domain |', '| --- | --- | --- | --- | --- |']
    for task, d in sorted(value['details'].items()):
        lines.append(f"| {task} | {d.get('category')} | {d.get('reward_type')} | {'yes' if d['trainable'] else 'no'} "
                     f"| {d.get('reason') or d.get('domain') or ''} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return data


def run(out, **options):
    """Run or resume every stage; returns (exit_code, tasks.json value)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    options.setdefault('python', 'python')
    timings, events = {}, []

    def log(stage, state, info=''):
        events.append({'stage': stage, 'state': state, 'info': str(info), 'time': time.time()})
        _write(out / 'run_events.json', events)
        print(f'[{stage}] {state} {info}', flush=True)

    def timed(name, fn, *args):
        started = time.monotonic()
        result = fn(*args)
        timings[name] = round(time.monotonic() - started, 1)
        return result

    _write(out / 'run_options.json', {k: v for k, v in options.items() if isinstance(v, (str, int, float, bool,
                                                                                        type(None)))})
    domains_ = timed('domains', stage_domains, out, options, log)
    generation = timed('generate', stage_generate, out, domains_, options, log)
    statuses = timed('build', stage_build, out, options, log)
    difficulty = timed('difficulty', stage_difficulty, out, statuses, options, log)
    hard_enough = {task: status for task, status in statuses.items()
                   if (difficulty.get(task) or {'passed': True})['passed']}
    docker_results = timed('docker', stage_docker, out, hard_enough, options, log) if options['docker'] else {}
    policy_results = timed('policy', stage_policy, out, hard_enough, options, log) if options['policy_k'] else {}
    value = manifest(out, statuses, docker_results, policy_results, generation, options, difficulty)
    report(out, value, timings, options)
    log('manifest', 'done', f"{value['summary']['trainable']} trainable -> {out / 'tasks.json'}")
    return (0 if value['tasks'] else 1), value
