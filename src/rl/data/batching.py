"""Sandbox batches: preserve each sandbox's complete rollout group per step."""
import random


def combine_samples(groups, rollout_group):
    samples, offset = [], 0
    for sandbox_index, group in enumerate(groups):
        indices = {s['episode_index'] for s in group}
        if not indices or indices != set(range(len(indices))):
            raise ValueError('incomplete rollout group')
        metadata = {}
        for sample in group:
            index = sample['episode_index']
            tree = sample.get('tree')
            if index in metadata and metadata[index] != tree:
                raise ValueError('inconsistent tree metadata')
            metadata[index] = tree
        if any(metadata.values()):
            trunks = {i for i, tree in metadata.items() if tree and tree.get('role') == 'trunk'}
            if trunks != set(range(rollout_group)):
                raise ValueError('incomplete rollout trunks')
            for i, tree in metadata.items():
                if not tree or tree.get('role') not in ('trunk', 'branch'):
                    raise ValueError('invalid tree role')
                if tree['role'] == 'branch' and (tree.get('trunk') not in trunks
                        or not isinstance(tree.get('fork_step'), int) or tree['fork_step'] < 1):
                    raise ValueError('invalid branch lineage')
        elif len(indices) != rollout_group:
            raise ValueError('incomplete rollout group')
        samples.extend({**sample, 'sandbox_index': sandbox_index,
                        'episode_index': offset + sample['episode_index'],
                        'weight': sample['weight'] / len(groups)} for sample in group)
        offset += len(indices)
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
