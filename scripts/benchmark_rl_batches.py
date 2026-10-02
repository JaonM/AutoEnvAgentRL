#!/usr/bin/env python3
"""Local-only cached vs tensor-batch scoring/gradient benchmark (no simulator/API).

Example: .venv/bin/python scripts/benchmark_rl_batches.py --output output/rl_runs/batch-benchmark.json
Measures kernels on identical sampled trajectories, not end-to-end RL speedup.
"""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3-4B-Instruct-2507-4bit')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from rl.model import Policy
    from rl.actor import collate, actor_loss, ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    import mlx.optimizers as optim
    mx.random.seed(42)
    model = Policy(args.model)
    samples = []
    for index in range(args.samples):
        sample = model.sample([{'role': 'user', 'content':
            f'Invent a name and description for a fictional animal from planet {index}. Use one paragraph.'}], max_tokens=args.tokens)
        sample.update(advantage=(-1. if index % 2 else 1.), weight=1./args.samples,
                      _ref_logp=sample['old_logp'], loss_mask=[1]*len(sample['tokens']))
        samples.append(sample)
    batch = collate(samples)
    def cached_score():
        return [model.cached_token_stats(mx.array([s['prompt'] + s['tokens']]), len(s['prompt']))[0] for s in samples]
    def batch_score():
        return model.batch_token_stats(batch['ids'], batch['prompt_lengths'], batch['response_lengths'])[0]
    def loss(policy, packed):
        return actor_loss(policy, packed, algorithm='grpo', clip=.2, beta=.01, value_coefficient=0.)[0]
    grad = nn.value_and_grad(model, loss)
    cached_batch = {**batch, 'cached_fallback': True}
    def cached_gradient(): return grad(model, cached_batch)
    def batch_gradient(): return grad(model, batch)
    def timed(fn):
        mx.eval(fn())  # warm-up and graph/materialization
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = fn()
            mx.eval(result)
            durations.append(time.perf_counter() - start)
        return statistics.median(durations), result
    cached_seconds, old = timed(cached_score)
    batch_seconds, new = timed(batch_score)
    cached_grad_seconds, (cached_loss, cached_grads) = timed(cached_gradient)
    batch_grad_seconds, (batch_loss, batch_grads) = timed(batch_gradient)
    guarded = ActorTrainer(model, model, optim.Adam(learning_rate=1e-6), Config(), StageTimes())
    def guarded_gradient():
        guarded.cached_policy_scores.clear()  # include real validation cost on every measured call
        guarded.score(samples, verify=True)
        total, accumulated = mx.array(0.), None
        for rows in guarded.chunks(samples):
            packed = collate(rows)
            packed['cached_fallback'] = guarded.shape_key(rows) in guarded.fallback_shapes
            value, gradients = grad(model, packed)
            total = total + value
            accumulated = gradients if accumulated is None else tree_map(lambda a, b: a+b, accumulated, gradients)
        return total, accumulated
    guarded_seconds, (guarded_loss, guarded_grads) = timed(guarded_gradient)
    def cached_pipeline():
        mx.eval(cached_score())  # policy drift
        mx.eval(cached_score())  # reference
        mx.eval(cached_gradient())
        return cached_score()   # post-update KL scoring, without mutating weights in a benchmark
    def guarded_pipeline():
        guarded.cached_policy_scores.clear()
        guarded.score(samples)
        guarded.score(samples, verify=True)
        guarded.score(samples, reference=True)
        for rows in guarded.chunks(samples):
            packed = collate(rows)
            packed['cached_fallback'] = guarded.shape_key(rows) in guarded.fallback_shapes
            mx.eval(grad(model, packed))
        guarded.cached_policy_scores.clear()
        return guarded.score(samples)
    baseline_pipeline_seconds, _ = timed(cached_pipeline)
    guarded_pipeline_seconds, _ = timed(guarded_pipeline)
    difference = tree_map(lambda a, b: a-b, cached_grads, batch_grads)
    norm = lambda tree: sum(float(mx.sum(value*value)) for _, value in tree_flatten(tree)) ** .5
    errors = [float(mx.max(mx.abs(new[index, :len(sample['tokens'])] - old[index]))) for index, sample in enumerate(samples)]
    result = {'scope': 'local kernel microbenchmark; no sandbox/API and no end-to-end speedup claim',
              'model': args.model, 'samples': len(samples), 'response_tokens': [len(s['tokens']) for s in samples],
              'cached_scoring_seconds': cached_seconds, 'batch_scoring_seconds': batch_seconds,
              'scoring_speedup': cached_seconds/batch_seconds,
              'cached_forward_backward_seconds': cached_grad_seconds, 'batch_forward_backward_seconds': batch_grad_seconds,
              'forward_backward_speedup': cached_grad_seconds/batch_grad_seconds,
              'max_logp_error': max(errors), 'per_row_logp_error': errors,
              'cached_loss': float(cached_loss), 'batch_loss': float(batch_loss),
              'relative_gradient_l2_error': norm(difference) / max(norm(cached_grads), 1e-12),
              'cached_training_phases_seconds': baseline_pipeline_seconds,
              'guarded_training_phases_seconds': guarded_pipeline_seconds,
              'guarded_training_phases_speedup': baseline_pipeline_seconds/guarded_pipeline_seconds,
              'guarded_forward_backward_seconds': guarded_seconds,
              'guarded_speedup_vs_cached_forward_backward': cached_grad_seconds/guarded_seconds,
              'guarded_relative_gradient_l2_error': norm(tree_map(lambda a, b: a-b, cached_grads, guarded_grads))/max(norm(cached_grads), 1e-12),
              'guarded_physical_batch_sizes': [len(rows) for rows in guarded.chunks(samples)],
              'guarded_cached_fallback_rows': sum(len(rows) for rows in guarded.chunks(samples) if guarded.shape_key(rows) in guarded.fallback_shapes),
              'batch_logp_tolerance': guarded.config.batch_logp_tolerance,
              'source_sha256': {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path('src/rl').glob('*.py')},
              'peak_metal_gb': mx.get_peak_memory()/1e9}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
