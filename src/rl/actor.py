"""Policy (actor) training: physical tensor batches with weighted accumulation."""
import math
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .objectives import clipped_surrogate, sampled_kl, clipped_value_loss
from .metrics import WeightedMetrics


def physical_batches(samples, max_sequences, max_tokens):
    """Length-bucketed right padding; logical sandbox/rollout weights stay intact."""
    pending = []
    for sample in sorted(samples, key=lambda s: len(s['prompt']) + len(s['tokens'])):
        length = len(sample['prompt']) + len(sample['tokens'])
        if length > max_tokens:
            raise ValueError('one segment exceeds max_tokens_per_micro_batch; increase the explicit budget')
        if pending and (len(pending) >= max_sequences or (len(pending) + 1) * length > max_tokens):
            yield pending
            pending = []
        pending.append(sample)
    if pending:
        yield pending


def collate(samples, *, normalize=1.):
    prompts = [len(s['prompt']) for s in samples]
    responses = [len(s['tokens']) for s in samples]
    length, width = max(p + r for p, r in zip(prompts, responses)), max(responses)
    ids, old_logp, ref_logp, advantages, old_values, returns, weights = [], [], [], [], [], [], []
    def row(value, n):
        values = value if isinstance(value, list) else [value] * n
        if len(values) != n:
            raise ValueError('unaligned training tensor')
        return values + [0.] * (width - n)
    for sample, n in zip(samples, responses):
        full = sample['prompt'] + sample['tokens']
        ids.append(full + [0] * (length - len(full)))
        mask = sample.get('loss_mask', [1] * n)
        if len(mask) != n or not sum(mask):
            raise ValueError('empty or unaligned assistant loss mask')
        old_logp.append(row(sample['old_logp'], n))
        ref_logp.append(row(sample.get('_ref_logp', [0.] * n), n))
        advantages.append(row(sample.get('advantage', 0.), n))
        old_values.append(row(sample.get('old_values', [0.] * n), n))
        returns.append(row(sample.get('return', [0.] * n), n))
        weights.append(row([sample.get('weight', 1.) / normalize / sum(mask) * x for x in mask], n))
    return {'ids': mx.array(ids), 'prompt_lengths': prompts, 'response_lengths': responses,
            **{key: mx.array(value) for key, value in [('old_logp', old_logp), ('ref_logp', ref_logp),
               ('advantages', advantages), ('old_values', old_values), ('returns', returns), ('weights', weights)]}}


def actor_loss(model, batch, *, algorithm, clip, beta, value_coefficient):
    if batch.get('cached_fallback'):
        probabilities, predictions = [], []
        width = batch['old_logp'].shape[1]
        for index, (prompt, response) in enumerate(zip(batch['prompt_lengths'], batch['response_lengths'])):
            p, v = model.cached_token_stats(batch['ids'][index:index+1, :prompt+response], prompt)
            probabilities.append(mx.pad(p, (0, width-response)))
            if v is not None:
                predictions.append(mx.pad(v, (0, width-response)))
        logp = mx.stack(probabilities)
        values = mx.stack(predictions) if predictions else None
    else:
        logp, values = model.batch_token_stats(batch['ids'], batch['prompt_lengths'], batch['response_lengths'])
    weight = batch['weights']
    active = weight > 0
    logp = mx.where(active, logp, 0.)
    old_logp = mx.where(active, batch['old_logp'], 0.)
    ref_logp = mx.where(active, batch['ref_logp'], 0.)
    pg = mx.sum(clipped_surrogate(logp, old_logp, batch['advantages'], clip) * weight)
    kl = mx.sum(sampled_kl(logp, ref_logp) * weight)
    vf = (mx.sum(clipped_value_loss(mx.where(active, values, 0.), mx.where(active, batch['old_values'], 0.), mx.where(active, batch['returns'], 0.), clip) * weight)
          if algorithm == 'ppo' else mx.array(0.))
    total = pg + beta * kl + value_coefficient * vf
    delta = logp - old_logp
    return total, (pg, kl, vf, mx.sum((mx.abs(mx.exp(delta) - 1) > clip) * weight),
                   mx.sum((mx.exp(delta) - 1 - delta) * weight))


class ActorTrainer:
    def __init__(self, policy, reference, optimizer, config, times):
        self.policy, self.reference, self.optimizer = policy, reference, optimizer
        self.config, self.times = config, times
        self.max_numerical_error = 0.
        self.max_physical_batch = 0
        self.fallback_shapes = set()
        self.numeric_fallbacks = 0
        self.batched_gradient_microbatches = 0
        self.cached_gradient_microbatches = 0
        self.cached_policy_scores = {}
        self.value_grad = nn.value_and_grad(policy, lambda model, batch: actor_loss(
            model, batch, algorithm=config.algorithm, clip=config.clip, beta=config.beta,
            value_coefficient=config.value_coefficient))

    def state(self):
        return {'fallback_shapes': list(self.fallback_shapes), 'numeric_fallbacks': self.numeric_fallbacks,
                'max_numerical_error': self.max_numerical_error, 'max_physical_batch': self.max_physical_batch,
                'batched_gradient_microbatches': self.batched_gradient_microbatches,
                'cached_gradient_microbatches': self.cached_gradient_microbatches}

    def restore_state(self, state):
        self.fallback_shapes = {tuple(x) for x in state.get('fallback_shapes', [])}
        self.numeric_fallbacks = state.get('numeric_fallbacks', 0)
        self.max_numerical_error = state.get('max_numerical_error', 0.)
        self.max_physical_batch = state.get('max_physical_batch', 0)
        self.batched_gradient_microbatches = state.get('batched_gradient_microbatches', 0)
        self.cached_gradient_microbatches = state.get('cached_gradient_microbatches', 0)

    @staticmethod
    def shape_key(rows):
        return (len(rows), max(len(s['prompt']) + len(s['tokens']) for s in rows), max(len(s['tokens']) for s in rows))

    def chunks(self, samples):
        def split(rows):
            if self.shape_key(rows) in self.fallback_shapes and len(rows) > 1:
                middle = len(rows) // 2
                yield from split(rows[:middle])
                yield from split(rows[middle:])
            else:
                yield rows
        for rows in physical_batches(samples, self.config.micro_batch_size, self.config.max_tokens_per_micro_batch):
            yield from split(rows)

    def score(self, samples, *, reference=False, verify=False):
        model = self.reference if reference else self.policy
        output = {}
        def cached_score(row):
            key = (tuple(row['prompt']), tuple(row['tokens']))
            if not reference and key in self.cached_policy_scores:
                return mx.array(self.cached_policy_scores[key])
            value = model.cached_token_stats(mx.array([row['prompt'] + row['tokens']]), len(row['prompt']))[0]
            mx.eval(value)
            if not reference:
                self.cached_policy_scores[key] = value.tolist()
            return value
        cached_results = {}
        pending = list(self.chunks(samples))
        while pending:
            rows = pending.pop(0)
            self.max_physical_batch = max(self.max_physical_batch, len(rows))
            key = self.shape_key(rows)
            if key in self.fallback_shapes and len(rows) == 1:
                with self.times.measure('actor_cached_scoring_fallback'):
                    output[id(rows[0])] = cached_score(rows[0]).tolist()
                    if reference:
                        rows[0]['_ref_logp'] = output[id(rows[0])]
                continue
            with self.times.measure('actor_reference_scoring' if reference else 'actor_policy_scoring'):
                batch = collate(rows)
                logp, values = model.batch_token_stats(batch['ids'], batch['prompt_lengths'], batch['response_lengths'])
                mx.eval(logp, values)
                measured = logp.tolist()
            key = self.shape_key(rows)
            if verify and key not in self.fallback_shapes:
                # Check every row against the independent cached path; never overwrite behavior probabilities.
                with self.times.measure('actor_batch_numerical_check'):
                    errors = []
                    for index, row in enumerate(rows):
                        if id(row) not in cached_results:
                            cached_results[id(row)] = cached_score(row)
                        difference = mx.abs(logp[index, :len(row['tokens'])] - cached_results[id(row)])
                        errors.append(float(mx.max(mx.where(mx.array(row.get('loss_mask', [1]*len(row['tokens']))), difference, 0.))))
                    error = max(errors)
                self.max_numerical_error = max(self.max_numerical_error, error)
                if not math.isfinite(error):
                    raise FloatingPointError('non-finite batch scoring discrepancy')
                if error > self.config.batch_logp_tolerance:
                    self.fallback_shapes.add(key)
                    self.numeric_fallbacks += 1
            if key in self.fallback_shapes and len(rows) > 1:
                middle = len(rows) // 2
                pending[:0] = [rows[:middle], rows[middle:]]
                continue
            if key in self.fallback_shapes:
                with self.times.measure('actor_cached_scoring_fallback'):
                    measured = [cached_score(row).tolist() for row in rows]
            for row, probabilities in zip(rows, measured):
                output[id(row)] = probabilities[:len(row['tokens'])]
                if reference:
                    row['_ref_logp'] = output[id(row)]
        return output

    def step(self, samples, *, notify=lambda: None):
        self.score(samples, verify=True)
        self.score(samples, reference=True)
        accumulated, aggregate = None, WeightedMetrics()
        normalizer = sum(s['weight'] for s in samples)
        for rows in self.chunks(samples):
            notify()
            with self.times.measure('actor_forward_backward'):
                batch = collate(rows, normalize=normalizer)
                batch['cached_fallback'] = self.shape_key(rows) in self.fallback_shapes
                if batch['cached_fallback']:
                    self.cached_gradient_microbatches += 1
                else:
                    self.batched_gradient_microbatches += 1
                (total, details), grads = self.value_grad(self.policy, batch)
                mx.eval(total, details, grads)
                mass = sum(s['weight'] for s in rows) / normalizer
                aggregate.add(mass, [float(x) / mass for x in (total, *details)])
                accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
        measured = aggregate.result()
        if self.config.target_kl and measured['behavior_kl'] > self.config.target_kl:
            return {**measured, 'skipped': 'target_kl'}
        flat = [g for _, g in tree_flatten(accumulated)]
        if not all(bool(mx.all(mx.isfinite(g))) for g in flat):
            raise FloatingPointError('non-finite actor gradient')
        with self.times.measure('actor_optimizer_step'):
            clipped, norm = optim.clip_grad_norm(accumulated, self.config.max_grad_norm)
            mx.eval(clipped, norm)
            if float(norm) == 0:
                return {**measured, 'skipped': 'zero_gradient'}
            if not math.isfinite(float(norm)):
                raise FloatingPointError('non-finite gradient norm')
            self.optimizer.update(self.policy, clipped)
            mx.eval(self.policy.parameters(), self.optimizer.state)
            self.cached_policy_scores.clear()
        with self.times.measure('actor_post_update_kl'):
            current = self.score(samples)
            post_kl = sum(s['weight'] / normalizer * sum(
                (math.exp(new - old) - 1 - (new - old))
                for new, old, mask in zip(current[id(s)], s['old_logp'], s.get('loss_mask', [1]*len(s['tokens']))) if mask)
                / sum(s.get('loss_mask', [1]*len(s['tokens']))) for s in samples)
        if not math.isfinite(post_kl):
            raise FloatingPointError('non-finite post-update KL')
        return {**measured, 'gradient_norm': float(norm), 'post_behavior_kl': max(0., post_kl),
                'post_kl_exceeded': bool(self.config.target_kl and post_kl > self.config.target_kl)}
