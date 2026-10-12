"""Learner difficulty calibration: k sampled episodes of a local MLX model on each V3 task.

GRPO only learns from tasks where the learner sometimes succeeds and sometimes fails, so this
measures the learner itself (not the policy-gate model): per task success rate, mean reward and
why episodes ended, in one user mode. Results are appended per task, so an interrupted run resumes.

    python experiments/v3/learner_pass_at_k.py --tasks output/v3_smoke/all_tasks.json \
        --model models/Qwen3-4B-Instruct-2507-4bit --k 8 --user-mode natural --output output/v3_pass_at_k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', required=True, help='tasks.json from `env_factory_v3 run` (its "tasks" list)')
    parser.add_argument('--model', default='models/Qwen3-4B-Instruct-2507-4bit')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--user-mode', choices=('natural', 'structured'), default='natural')
    parser.add_argument('--tool-names', choices=('random', 'fixed', 'heldout'), default='random',
                        help='platform tool naming; heldout is the probe for calls to tools that are not offered')
    parser.add_argument('--max-steps', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-context', type=int, default=16384)
    parser.add_argument('--thinking-mode', choices=('auto', 'thinking', 'no-thinking'), default='auto')
    parser.add_argument('--only', default='', help='comma-separated task ids')
    parser.add_argument('--output', required=True, help='JSONL, one row per (task, user mode); resumes')
    args = parser.parse_args()
    os.environ['ENVFACTORY_V3_USER_MODE'] = args.user_mode
    os.environ['ENVFACTORY_V3_TOOL_NAMES'] = args.tool_names

    import mlx.core as mx
    from rl.backends.mlx.model_options import policy_options
    from rl.backends.mlx.policy import Policy
    from rl.environments.v3 import V3Episode
    from rl.rollout.worker import rollout
    from rl.trainer.cli import parse_args

    config = parse_args(['--output', '/dev/null', '--tasks', args.tasks, '--model', args.model, '--max-steps', str(args.max_steps),
                         '--max-tokens', str(args.max_tokens), '--max-context', str(args.max_context),
                         '--thinking-mode', args.thinking_mode])
    config = asdict(config) if not isinstance(config, dict) else config
    mx.set_default_device(mx.gpu)
    mx.random.seed(config['seed'])
    policy = Policy(config['model'], config['tuning'], config['layers'], config['rank'], config['bits'],
                    temperature=config['temperature'], critic=False, **policy_options(config, inference=True))

    tasks = json.loads(Path(args.tasks).read_text())['tasks']
    wanted = {t for t in args.only.split(',') if t}
    out = Path(args.output)
    done = set()
    if out.exists():
        done = {(r['id'], r['user_mode']) for r in map(json.loads, out.read_text().splitlines())
                if r.get('k') == args.k and r.get('tool_names', 'random') == args.tool_names}
    for task in tasks:
        if (wanted and task['id'] not in wanted) or (task['id'], args.user_mode) in done:
            continue
        env = V3Episode(task['sandbox'])
        episodes = []
        try:
            for seed in range(args.k):
                started = time.monotonic()
                try:
                    result = rollout(policy, env, config, 7000 + seed)
                    score = result.get('score') or {}
                    episodes.append({'reward': result.get('final_reward'), 'success': bool(score.get('success')),
                                     'reason': score.get('reason'), 'finish': result.get('finish_reason'),
                                     'steps': len(result.get('actions') or []), 'world': result.get('world'),
                                     'tool_calls': score.get('tool_calls'),
                                     'unknown_tool_calls': result.get('unknown_tool_calls'),
                                     'platform_tool_hallucinations': result.get('platform_tool_hallucinations'),
                                     'seconds': round(time.monotonic() - started, 1)})
                except Exception as exc:  # one broken episode is recorded, not fatal
                    episodes.append({'error': f'{type(exc).__name__}: {exc}'[:300],
                                     'seconds': round(time.monotonic() - started, 1)})
        finally:
            env.close()
        rewards = [e.get('reward') or 0.0 for e in episodes]
        successes = sum(e.get('success', False) for e in episodes)
        row = {'id': task['id'], 'sandbox': task['sandbox'], 'user_mode': args.user_mode, 'model': args.model,
               'tool_names': args.tool_names,
               'k': args.k, 'successes': successes, 'success_rate': round(successes / len(episodes), 3),
               'mean_reward': round(sum(rewards) / len(rewards), 3),
               'reward_variance': round(sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards), 4),
               'episodes': episodes}
        with out.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"{task['id']} {args.user_mode}: {successes}/{len(episodes)} mean {row['mean_reward']} "
              f"({sum(e['seconds'] for e in episodes):.0f}s)", flush=True)


if __name__ == '__main__':
    main()
