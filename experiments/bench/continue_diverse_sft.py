"""Wait for the bounded diversity producer, audit its release, then run SFT controls."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from release_diverse_dataset import audit_and_release, read, save, sha


def producer_state(producer, pid):
    progress = read(producer / 'progress.json')
    if progress['phase'] == 'completed':
        return progress
    if progress['phase'] in ('failed', 'budget_exhausted'):
        raise RuntimeError('producer stopped without a complete dataset: ' + progress['phase'])
    process = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    if process.returncode or 'supplement_world_diversity.py' not in process.stdout or str(producer) not in process.stdout:
        raise RuntimeError('expected diversity producer exited; incomplete data cannot train')
    return progress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--producer-pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--public-mix', type=Path, required=True)
    p.add_argument('--old-control', action='store_true')
    p.add_argument('--port', type=int, default=9410)
    p.add_argument('--wait-hours', type=float, default=0,
                   help='Optional generation deadline; 0 keeps following a live bounded producer')
    args = p.parse_args()
    if args.wait_hours < 0:
        p.error('--wait-hours must be nonnegative')
    os.chdir(ROOT)
    args.producer = args.producer.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    wake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())]) if sys.platform == 'darwin' else None
    phase = 'waiting_for_generation'
    started = datetime.now().astimezone().isoformat()
    files = [*ROOT.glob('src/**/*.py'), *ROOT.glob('experiments/**/*.py'),
             *ROOT.glob('experiments/bench/*.sh'), ROOT / '.env', args.public_mix,
             args.public_mix.parent / 'manifest.json', ROOT / 'output/I2nfix/pass8/task_ids.json',
             *args.audit_root.glob('*/branches.json'), *args.audit_root.glob('*/matrix.json')]
    frozen = {str(p.resolve()): sha(p) for p in files}
    save(args.output / 'protocol.json', {'started': started, 'producer': str(args.producer),
         'producer_pid': args.producer_pid, 'runtime': str(ROOT), 'frozen_inputs': frozen,
         'public_mix': str(args.public_mix), 'old_control': args.old_control,
         'wait_budget_hours': args.wait_hours or None, 'offpeak_wait_disabled': True,
         'authorization': 'User requested automatic continuation through training and evaluation after generation.',
         'order': ['complete 1000/200', 'joint deterministic evidence/coverage review',
                   'world seed0 + 164 evaluation', 'base 164', 'mixed seed0 + 164',
                   'world/mixed seed1 + 164', 'historical controls if requested', 'fixed 32 x 8 for all arms']})
    def state(**extra):
        save(args.output / 'status.json', {'phase': phase, 'started': started,
             'updated': datetime.now().astimezone().isoformat(), **extra})
    def unchanged():
        for name, expected in frozen.items():
            if sha(name) != expected:
                raise ValueError('frozen experiment input changed: ' + name)
    try:
        deadline = time.monotonic() + args.wait_hours * 3600 if args.wait_hours else None
        while True:
            progress = producer_state(args.producer, args.producer_pid)
            state(training_started=False, counts=progress['dataset_counts'], remaining=progress['remaining'])
            if progress['phase'] == 'completed':
                break
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError('bounded generation wait expired')
            time.sleep(30)
        unchanged()
        phase = 'joint_dataset_review'
        state(training_started=False)
        dataset = args.output / 'dataset'
        audit_and_release(args.producer, dataset, args.audit_root)
        unchanged()
        phase = 'training_and_evaluation'
        # Use a unique control name, because evaluation arms share the output/tau2 namespace.
        control_output = args.output / (args.output.name + '-control')
        command = [sys.executable, '-u', str(ROOT / 'experiments/bench/world_sft_control.py'),
                   '--dataset', str(dataset), '--producer-pid', str(args.producer_pid),
                   '--out', str(control_output), '--public-mix', str(args.public_mix),
                   '--evaluate-as-ready', '--port', str(args.port)]
        if args.old_control:
            command.append('--old-control')
        save(args.output / 'control_command.json', command)
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'ENVFACTORY_OFFPEAK': '0',
               'TOKENIZERS_PARALLELISM': 'false'}
        with (args.output / 'control.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                detail = read(control_output / 'status.json') if (control_output / 'status.json').exists() else {}
                state(control=detail, control_pid=child.pid, results=str(control_output / 'results.md'))
                time.sleep(30)
            if child.returncode:
                raise RuntimeError(f'experiment control exited {child.returncode}; see control.log and control status')
        phase = 'complete'
        state(training_started=True, results=str(control_output / 'results.md'))
    except Exception as exc:
        failed_phase, phase = phase, 'failed'
        state(failed_phase=failed_phase, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if wake is not None and wake.poll() is None:
            wake.terminate()


if __name__ == '__main__':
    main()
