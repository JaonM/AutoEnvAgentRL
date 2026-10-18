"""Wait for an admitted world dataset, then run frozen two-seed SFT and tau2 controls.

No partial-run resume: a failed phase leaves its evidence and requires a fresh output.
Benchmark task IDs are used only by evaluation, never by the task author.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from sft_train import prepare_episodes, render_episode
from pass_k import report, summarize, task_rows


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'frozen input changed: {path}')


def validate_release(dataset):
    release = json.loads((dataset / 'release.json').read_text())
    if not release.get('training_ready') or (release['train_tasks'], release['dev_tasks']) != (1000, 200):
        raise ValueError('dataset has not released 1000 train and 200 dev tasks')
    for name in ('teacher.jsonl', 'valid_ids.json', 'tasks.json'):
        if digest(dataset / name) != release['source_hashes'].get(name):
            raise ValueError(f'release checksum mismatch: {name}')
    episodes = [json.loads(line) for line in (dataset / 'teacher.jsonl').read_text().splitlines()]
    ids = [e['id'] for e in episodes]
    held = json.loads((dataset / 'valid_ids.json').read_text())
    dev = {e['id'] for e in episodes if e['split'] == 'dev'}
    if len(ids) != 1200 or len(set(ids)) != 1200 or len(held) != 200 or set(held) != dev:
        raise ValueError('release task IDs are duplicated, missing or mispartitioned')
    counts = collections.Counter((e['domain'], e['split']) for e in episodes)
    if counts != {('retail', 'train'): 500, ('airline', 'train'): 500,
                  ('retail', 'dev'): 100, ('airline', 'dev'): 100}:
        raise ValueError('release domain quota differs from plan')
    if any(not e.get('success') or not e.get('policy_aware_success') or not e.get('quality_audit_passed')
           or not e.get('messages') for e in episodes):
        raise ValueError('release contains an unqualified demonstration')
    families = {s: {e['scenario_family'] for e in episodes if e['split'] == s} for s in ('train', 'dev')}
    if families['train'] & families['dev']:
        raise ValueError('scenario-family assignments overlap')
    return release


def data_preflight(teacher, replay, valid_ids, model, public_replay=None):
    # Reuse the trainer's actual filtering, deterministic augmentation, renderer and mask.
    from transformers import AutoTokenizer
    args = argparse.Namespace(teacher=str(teacher), replay=list(map(str, replay)), seed=0,
                              split_seed=0, augmentation_seed=0, rename_tools=.7, vary_results=.6,
                              greet=.5, relayout=.5, example_values=.5, inject_errors=.5,
                              context_only_announcements=True, context_only_reports=0,
                              valid_ids=str(valid_ids), public_replay=list(map(str, public_replay or [])))
    episodes = prepare_episodes(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    held = set(json.loads(Path(valid_ids).read_text()))
    parts = {s: {'samples': 0, 'tokens': 0, 'assistant_tokens': 0} for s in ('train', 'dev')}
    hashes = {s: hashlib.sha256() for s in parts}
    dropped = []
    sources = collections.Counter()
    for episode in episodes:
        split = 'dev' if episode['id'] in held else 'train'
        rendered = render_episode(tokenizer, episode, skip_failed=True)
        if len(rendered) != 1 or len(rendered[0][0]) > 32768 or not sum(rendered[0][1]):
            dropped.append(episode['id'])
            continue
        ids, mask = rendered[0]
        parts[split]['samples'] += 1
        sources[f'{split}:{"public" if episode.get("public") else "world"}'] += 1
        parts[split]['tokens'] += len(ids)
        parts[split]['assistant_tokens'] += sum(mask)
        hashes[split].update(json.dumps([ids, mask], separators=(',', ':')).encode() + b'\n')
    return {'parts': parts, 'sources': dict(sources), 'dropped': dropped, 'dedup': args.dedup_report,
            'train_sha256': hashes['train'].hexdigest(), 'valid_sha256': hashes['dev'].hexdigest()}


def training_command(teacher, replay, valid_ids, model, output, seed, public_replay=None):
    command = [sys.executable, '-u', 'experiments/sft_train.py', '--teacher', str(teacher),
               '--model', str(model), '--output', str(output), '--valid-ids', str(valid_ids),
               '--tuning', 'full', '--learning-rate', '2e-5', '--grad-accumulation', '16',
               '--epochs', '10', '--evals-per-epoch', '8', '--early-stopping-patience', '8',
               '--early-stopping-min-delta', '.001', '--early-stopping-min-epochs', '1',
               '--seed', str(seed), '--split-seed', '0', '--augmentation-seed', '0',
               '--rename-tools', '.7', '--vary-results', '.6', '--greet', '.5', '--relayout', '.5',
               '--no-error-turns', '--max-tokens', '32768', '--render', 'episode',
               '--example-values', '.5', '--inject-errors', '.5', '--context-only-announcements']
    if replay:
        command += ['--replay', *map(str, replay)]
    if public_replay:
        command += ['--public-replay', *map(str, public_replay)]
    return command


def evaluate(arm, model, trials, selected, ids_path, output, port, env):
    # quick_eval removes its own result directory; never hand it an existing arm.
    for domain in ('airline', 'retail'):
        if (Path(f'output/tau2/q{arm}-{domain}.log').exists()
                or Path(f'.bench/tau2-repo/data/simulations/v3bench/q{arm}-{domain}').exists()):
            raise ValueError(f'evaluation arm already exists: {arm}')
    options = {**env, 'BENCH_PORT': str(port), 'BATCH': '0', 'THINK': '0', 'SERVERS': '3',
               'BENCH_MAX_EVALS': '1', 'MAX_STEPS': '100', 'AGENT_TEMPERATURE': '1.0' if trials == 8 else '0.0'}
    options.pop('TASK_IDS_FILE', None)
    if trials == 8:
        options['TASK_IDS_FILE'] = str(ids_path.resolve())
    with (output / f'{arm}.log').open('x') as log:
        subprocess.run(['bash', 'experiments/bench/quick_eval.sh', arm, str(model), str(trials)],
                       env=options, stdout=log, stderr=subprocess.STDOUT, check=True)
    if trials == 8:
        result = report('q' + arm, 8, selected)
    else:
        rows = []
        protocol = {}
        for domain, count in (('airline', 50), ('retail', 114)):
            data = json.loads(Path(f'.bench/tau2-repo/data/simulations/v3bench/q{arm}-{domain}/results.json').read_text())
            expected = [task['id'] for task in data['tasks']]
            if len(set(expected)) != count:
                raise ValueError('single-trial task population differs from 164-task protocol')
            rows.extend(task_rows(data, domain, expected, 1))
            info = data['info']
            protocol[domain] = {'num_trials': info['num_trials'], 'seed': info['seed'], 'max_steps': info['max_steps'],
                                'agent_temperature': info['agent_info']['llm_args'].get('temperature')}
        result = {'arm': arm, 'protocol': protocol, 'overall': summarize(rows, 1),
                  'write': summarize([r for r in rows if r['write']], 1),
                  'no_write': summarize([r for r in rows if not r['write']], 1), 'tasks': rows}
    for protocol in result['protocol'].values():
        if (protocol['num_trials'], protocol['seed'], protocol['max_steps'], protocol['agent_temperature']) != (
                trials, 300, 100, 1.0 if trials == 8 else 0.0):
            raise ValueError('evaluation protocol changed')
    save(output / f'{arm}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen3-0.6B'))
    parser.add_argument('--task-ids', type=Path, default=Path('output/I2nfix/pass8/task_ids.json'))
    parser.add_argument('--old-control', action='store_true', help='repeat the historical data under the same early-stop schedule')
    parser.add_argument('--public-mix', type=Path, help='1000 qualified public episodes for an additional 1:1 sample mixture')
    parser.add_argument('--port', type=int, default=9310)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out.mkdir(parents=True, exist_ok=False)
    selected = json.loads(args.task_ids.read_text())
    if set(selected) != {'airline', 'retail'} or sum(map(len, selected.values())) != 32 \
            or any(len(ids) != len(set(ids)) for ids in selected.values()):
        raise ValueError('expected the existing fixed 32-task subset')
    ids_path = args.out / 'task_ids.json'
    save(ids_path, selected)
    code = [Path(__file__), *Path('experiments').glob('*.py'), *Path('experiments/bench').glob('*.py'),
            Path('experiments/bench/quick_eval.sh'), Path('experiments/bench/run_tau2.sh'), Path('.env')]
    frozen = {str(p.resolve()): digest(p) for p in code}
    frozen.update({str(p.resolve()): digest(p) for p in args.model.iterdir() if p.is_file()})
    frozen[str(ids_path.resolve())] = digest(ids_path)
    groups = [('world', args.dataset / 'teacher.jsonl', [], args.dataset / 'valid_ids.json', [])]
    if args.public_mix:
        rows = [json.loads(line) for line in args.public_mix.read_text().splitlines()]
        if len(rows) != 1000 or len({e['id'] for e in rows}) != 1000 or any(not e.get('public') for e in rows):
            raise ValueError('1:1 mixture requires 1000 distinct public episodes')
        frozen[str(args.public_mix.resolve())] = digest(args.public_mix)
        public_manifest = args.public_mix.parent / 'manifest.json'
        frozen[str(public_manifest.resolve())] = digest(public_manifest)
        if json.loads(public_manifest.read_text())['public_sha256'] != digest(args.public_mix):
            raise ValueError('public mixture checksum mismatch')
        groups.append(('mixed', args.dataset / 'teacher.jsonl', [], args.dataset / 'valid_ids.json', [args.public_mix]))
    if args.old_control:
        groups.append(('old', Path('output/I2n/selected.jsonl'), [Path('output/v3_I2/probe/replay.jsonl')],
                       Path('output/I2nfix/valid_ids.json'), []))
        frozen.update({str(p.resolve()): digest(p) for p in [groups[-1][1], *groups[-1][2], groups[-1][3]]})
    save(args.out / 'protocol.json', {'dataset': str(args.dataset), 'producer_pid': args.producer_pid,
         'frozen_inputs': frozen, 'early_stopping': {'max_epochs': 10, 'evals_per_epoch': 8, 'patience': 8,
         'min_delta': .001, 'min_epochs': 1}, 'old_control': args.old_control,
         'public_mix': {'path': str(args.public_mix), 'world_train': 1000, 'public_train': 1000,
                        'basis': 'full dialogue samples', 'validation': 'same 200 world tasks',
                        'public_augmentation': 'none; append after deterministic world augmentation'} if args.public_mix else None,
         'evaluation': {'single_tasks': 164, 'subset_tasks': selected, 'trials': [1, 8], 'seed': 300},
         'limits': ['Different datasets have different validation tasks and update/token budgets.',
                    'This does not measure learning transfer to unaligned domains.']})
    phase = 'waiting_for_dataset_release'
    try:
        save(args.out / 'status.json', {'phase': phase, 'training_started': False})
        deadline = time.monotonic() + 14 * 86400
        while not (args.dataset / 'release.json').exists():
            producer = subprocess.run(['ps', '-p', str(args.producer_pid), '-o', 'command='], capture_output=True, text=True)
            if producer.returncode or 'generate_world_dataset.py' not in producer.stdout:
                raise RuntimeError('dataset producer exited without release; no training started')
            if time.monotonic() >= deadline:
                raise TimeoutError('dataset did not release within 14 days')
            time.sleep(30)
        verify_hashes(frozen)
        release = validate_release(args.dataset)
        frozen.update({str((args.dataset / name).resolve()): sha for name, sha in release['source_hashes'].items()})
        save(args.out / 'dataset_release.json', release)
        env = {**os.environ, 'PYTHONPATH': 'src', 'TOKENIZERS_PARALLELISM': 'false'}
        models = [('base', args.model)]
        world_valid_hash = None
        for group, teacher, replay, held, public_replay in groups:
            phase = f'preflight_{group}'
            save(args.out / 'status.json', {'phase': phase})
            verify_hashes(frozen)
            preflight = data_preflight(teacher, replay, held, args.model, public_replay)
            save(args.out / f'{group}_data_preflight.json', preflight)
            if preflight['dropped'] or (group == 'world' and tuple(preflight['parts'][s]['samples'] for s in ('train', 'dev')) != (1000, 200)):
                raise ValueError('training rendering drops tasks or violates 1000/200 quota; inspect preflight')
            if group == 'world':
                world_valid_hash = preflight['valid_sha256']
            if group == 'mixed' and (preflight['sources'] != {'train:world': 1000, 'train:public': 1000, 'dev:world': 200}
                                     or preflight['valid_sha256'] != world_valid_hash):
                raise ValueError('mixture ratio or world validation fingerprint changed')
            first_manifest = None
            for seed in (0, 1):
                phase = f'training_{group}_seed{seed}'
                save(args.out / 'status.json', {'phase': phase, 'training_started': True})
                verify_hashes(frozen)
                run = args.out / f'{group}_seed{seed}'
                run.mkdir()
                cmd = training_command(teacher, replay, held, args.model, run / 'sft', seed, public_replay)
                save(run / 'command.json', cmd)
                with (run / 'sft.log').open('x') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                manifest = json.loads((run / 'sft/data_manifest.json').read_text())
                if any(manifest[key] != preflight[key] for key in ('train_sha256', 'valid_sha256')):
                    raise ValueError('training data differs from rendered preflight')
                if first_manifest is not None and manifest != first_manifest:
                    raise ValueError('data/augmentation differs across seeds')
                first_manifest = manifest
                models.append((f'{group}-s{seed}', run / 'sft/model'))
        results = {}
        for trials in (1, 8):
            for label, model in models:
                phase = f'evaluation_{label}_{trials}_trials'
                save(args.out / 'status.json', {'phase': phase, 'training_started': True})
                verify_hashes(frozen)
                arm = f'{args.out.name}-{label}-t{trials}'
                results[arm] = evaluate(arm, model, trials, selected, ids_path, args.out, args.port, env)
                save(args.out / 'evaluation_report.json', results)
        save(args.out / 'status.json', {'phase': 'complete', 'training_started': True})
        (args.out / 'all.done').write_text('done\n')
    except Exception as exc:
        save(args.out / 'status.json', {'phase': 'failed', 'failed_phase': phase,
                                     'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
