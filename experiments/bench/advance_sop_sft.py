"""Finish deterministic migration, freeze a runtime, then launch off-peak SFT production.

This controller never weakens an admission gate and never trains incomplete data.
It is intentionally independent of the interactive assistant session.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from env_factory.certify import bundle
from prepare_scaled_experiment import read,save


def freeze(destination):
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('src','experiments'):
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,destination/name)
    for name in ('.venv','.env','.bench','output','models','data'):
        if (ROOT/name).exists():(destination/name).symlink_to(ROOT/name,target_is_directory=(ROOT/name).is_dir())
    code={str(p.resolve()):bundle.sha256(p) for directory in ('src','experiments') for p in (destination/directory).rglob('*')
          if p.is_file() and p.suffix in ('.py','.sh')}
    return {'platform':bundle.platform_identity(),'code':code}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worlds',type=Path,required=True)
    p.add_argument('--replay-pid',type=int,required=True);p.add_argument('--port',type=int,default=9410)
    args=p.parse_args();out=args.output.resolve();worlds=args.worlds.resolve();replay=out/'replay'
    phase='waiting_for_replay';started=datetime.now().astimezone().isoformat()
    platform=bundle.platform_identity()
    def state(**extra):save(out/'advance_status.json',dict(phase=phase,started=started,
        updated=datetime.now().astimezone().isoformat(),training_started=False,**extra))
    wake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
    try:
        while True:
            progress=read(replay/'progress.json') if (replay/'progress.json').exists() else {}
            state(replayed=progress.get('completed',0),total=progress.get('total',556),reused=progress.get('reused',0))
            if progress.get('phase')=='completed':break
            proc=subprocess.run(['ps','-p',str(args.replay_pid),'-o','command='],capture_output=True,text=True)
            if proc.returncode or 'replay_world_semantics.py' not in proc.stdout or str(out.relative_to(ROOT)) not in proc.stdout:
                raise RuntimeError('expected replay process exited before complete migration')
            time.sleep(20)
        if bundle.platform_identity()!=platform:raise RuntimeError('platform changed during migration')
        if any(r.get('reason')=='migration_error' for r in progress['results']):
            raise RuntimeError('migration infrastructure/specification errors need diagnosis before sampling')
        phase='preparing_sop_deficits';state()
        with (out/'preparation.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'experiments/prepare_sop_experiment.py'),
                '--replay',str(replay),'--worlds',str(worlds),'--output',str(out/'preparation')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        phase='freezing_runtime';state()
        runtime=out/'runtime';identity=freeze(runtime);save(out/'dataset_identity.json',identity)
        plan=out/'preparation/fill_plan/plan.json';dataset=out/'dataset_run_v1'
        env={**os.environ,'PYTHONPATH':str(runtime/'src'),'ENVFACTORY_OFFPEAK':'1','TOKENIZERS_PARALLELISM':'false',
             'ENVFACTORY_CODEX_TRANSPORT':'auto',
             'PATH':str(ROOT/'output/world_sft_next/author_runtime_v1/bin')+os.pathsep+os.environ.get('PATH','')}
        python=str(runtime/'.venv/bin/python')
        producer=[python,'-u',str(runtime/'experiments/supplement_world_diversity.py'),'--plan',str(plan),
                  '--frozen',str(runtime),'--dataset-identity',str(out/'dataset_identity.json'),'--output',str(dataset)]
        with (out/'generation.log').open('x') as log:
            child=subprocess.Popen(producer,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch=dict(started=datetime.now().astimezone().isoformat(),producer_pid=child.pid,producer_command=producer,
                    runtime=str(runtime),offpeak=True,expected_baseline=read(plan)['historical_demonstrations_reused'])
        save(out/'launch.json',launch)
        for _ in range(30):
            if (dataset/'progress.json').exists():break
            if child.poll() is not None:raise RuntimeError('new SOP producer exited; see generation.log')
            time.sleep(1)
        if not (dataset/'progress.json').exists():raise RuntimeError('new SOP producer did not publish readiness')
        continuation=[python,'-u',str(runtime/'experiments/bench/continue_diverse_sft.py'),
            '--producer',str(dataset),'--producer-pid',str(child.pid),'--output',str(out/'sop_control_v1'),
            '--audit-root',str(out/'preparation/empty_audit'),
            '--public-mix',str(ROOT/'output/world_sft_next/public_mix_v3/public_train.jsonl'),
            '--old-control','--wait-hours','0','--port',str(args.port)]
        with (out/'continuation.log').open('x') as log:
            watcher=subprocess.Popen(continuation,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch.update(continuation_pid=watcher.pid,continuation_command=continuation);save(out/'launch.json',launch)
        phase='sampling_and_continuation_started';state(producer_pid=child.pid,continuation_pid=watcher.pid)
    except Exception as exc:
        failed_phase=phase;phase='failed';state(failed_phase=failed_phase,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if wake is not None and wake.poll() is None:wake.terminate()


if __name__=='__main__':main()
