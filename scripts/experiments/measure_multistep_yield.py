"""Measure fresh authoring on frozen requests through the production gates.

This is a fixed-input experiment, not fresh graph sampling. Failed authoring and
infrastructure errors stay in the planned denominator. Never imports old source
candidates or retries a terminal sample outside the production repair budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from env_factory.generation.artifacts import write_task_artifact
from env_factory.generation.code_agent import generate
from env_factory.graph.knowledge_graph import TaskType
from env_factory.task_pipeline import build_one, interrupt, write_json
from env_factory.tasks.task import Task


def platform_files():
    paths = [*(PROJECT / 'src').rglob('*.py'), *(PROJECT / 'scripts').rglob('*.py'),
             *(PROJECT / 'scripts').rglob('*.sh'), PROJECT / 'docs/code_agent_authoring.md',
             PROJECT / 'pyproject.toml', PROJECT / 'uv.lock']
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def summarize(samples):
    total = len(samples)
    terminal = sum(s['status'] in {'passed', 'failed'} for s in samples)
    good = sum(s['status'] == 'passed' and s.get('training_ready') is True for s in samples)
    return {'planned': total, 'terminal': terminal, 'training_ready': good,
            'pending': total - terminal,
            'end_to_end_yield': good / total if total and terminal == total else None,
            'target_met': bool(total and terminal == total and good / total >= .9),
            'scope': 'fresh_authoring_on_frozen_requests_not_population_yield'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--author-budget', type=float, default=600)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.author_budget < float('inf'):
        parser.error('repeats and author budget must be positive and finite')
    requests = [json.loads(p.read_text()) for p in args.request]
    if any(r.get('training_category') != 'multi_step_agentic' for r in requests):
        parser.error('every frozen request must be multi_step_agentic')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    fingerprints = platform_files()
    config = dict(code_agent='codex', code_agent_model='gpt-6.1-sol',
                  sandbox_model='gpt-6.1-sol', sandbox_runtime='docker',
                  max_attempts=3, max_total_repairs=3, infrastructure_retries=1,
                  score_threshold=8, build_timeout=3600, score_timeout=1800,
                  sample_build_budget=0)
    samples = []
    for repeat in range(args.repeats):
        for i, request in enumerate(requests):
            index = len(samples) + 1
            frozen = {**request, 'code_agent': 'codex', 'code_agent_model': 'gpt-6.1-sol'}
            request_path = root / f'input-{index}.json'
            write_json(request_path, frozen)
            samples.append(dict(id=index, input=str(request_path), original_input=str(args.request[i]),
                                repeat=repeat, status='pending', training_ready=False,
                                input_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()))
    write_json(root / 'protocol.json', dict(scope='fixed_input_fresh_authoring',
               target=.9, author_budget_seconds=args.author_budget, build_config=config,
               codex_transport=os.getenv('ENVFACTORY_CODEX_TRANSPORT', 'auto'),
               platform_files=fingerprints, samples=samples,
               limits=['Small fixed input set; not representative graph sampling.',
                       'Executable training qualification; no policy learning claim.',
                       'No retrospective source rescue counted as successful generation.']))
    def publish():
        write_json(root / 'results.json', dict(samples=samples, summary=summarize(samples)))
    publish()
    for sample in samples:
        if platform_files() != fingerprints:
            raise RuntimeError('Platform changed during experiment; remaining samples not run')
        directory = root / 'artifacts' / f"artifact-{sample['id']}"
        task_dir = directory / 'task'
        task_dir.mkdir(parents=True)
        started = time.monotonic()
        sample.update(status='running', phase='generation')
        publish()
        try:
            frozen = json.loads(Path(sample['input']).read_text())
            artifacts = generate(request=frozen, artifact_dir=task_dir,
                                 script_count=3, timeout=args.author_budget)
            sample['generation_passed'] = True
            sample['generation_seconds'] = time.monotonic() - started
            task = Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
                        task_type=TaskType(artifacts['task_type']), task_intent=artifacts['task_intent'],
                        complexity=artifacts['complexity'], artifacts=artifacts)
            path = write_task_artifact(task_dir, task, 'multi_step_agentic')
            sample['phase'] = 'build'
            publish()
            output = directory / 'sandbox'
            result = build_one(PROJECT, path, output, config)
            write_json(output / 'pipeline_result.json', result)
            sample.update(training_ready=result.get('training_ready') is True,
                          status='passed' if result.get('passed') is True and result.get('training_ready') is True else 'failed',
                          failure_code=result.get('failure_code'), failure_class=result.get('failure_class'))
        except Exception as exc:
            sample.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        finally:
            sample['seconds'] = time.monotonic() - started
            sample['platform_unchanged'] = platform_files() == fingerprints
            if not sample['platform_unchanged']:
                sample.update(status='failed', training_ready=False, error='Platform changed during sample')
            publish()
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    return 0 if summarize(samples)['target_met'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
