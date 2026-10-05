"""Sandbox batches: preserve each sandbox's complete rollout group per step."""
import random


def combine_samples(groups, rollout_group):
    samples = []
    for sandbox_index, group in enumerate(groups):
        if {s['episode_index'] for s in group} != set(range(rollout_group)):
            raise ValueError('incomplete rollout group')
        samples.extend({**sample, 'sandbox_index': sandbox_index,
                        'episode_index': sandbox_index * rollout_group + sample['episode_index'],
                        'weight': sample['weight'] / len(groups)} for sample in group)
    return samples


def minibatches(batch, size, *, seed):
    """Shuffle sandboxes, keeping all their rollouts together; retain a short tail."""
    sandboxes = {}
    for item in batch:
        sandboxes.setdefault(item[0]['sandbox_index'], []).append(item)
    if size < 1 or not sandboxes:
        raise ValueError('positive mini-batch size and nonempty sandbox batch required')
    indices = list(sandboxes)
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), size):
        yield [item for index in indices[start:start + size] for item in sandboxes[index]]
