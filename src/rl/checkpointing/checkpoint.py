"""Atomic, checksummed learner checkpoints committed at update boundaries."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(root, policy, optimizer, state):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['optimizer_step']:08d}-{uuid.uuid4().hex[:12]}"
    temporary = root / ('.' + name)
    temporary.mkdir()
    try:
        mx.save_safetensors(str(temporary / 'policy.safetensors'), dict(tree_flatten(policy.trainable_parameters())))
        mx.save_safetensors(str(temporary / 'optimizer.safetensors'),
                            {k: mx.array(v) for k, v in tree_flatten(optimizer.state)})
        learner_state = {**state, 'schema_version': 1,
                         'mlx_rng_state': [key.tolist() for key in mx.random.state]}
        atomic_json(temporary / 'state.json', learner_state)
        files = ('policy.safetensors', 'optimizer.safetensors', 'state.json')
        manifest = {name_: sha256(temporary / name_) for name_ in files}
        atomic_json(temporary / 'manifest.json', manifest)
        temporary.rename(root / name)
        atomic_json(root / 'latest.json', {'directory': name, 'manifest_sha256': sha256(root / name / 'manifest.json')})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root / name


def read_checkpoint(root):
    root = Path(root).resolve()
    pointer = json.loads((root / 'latest.json').read_text())
    directory = root / pointer['directory']
    if directory.resolve().parent != root or directory.name.startswith('.'):
        raise ValueError('invalid checkpoint pointer')
    if sha256(directory / 'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('checkpoint manifest checksum mismatch')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if set(manifest) != {'policy.safetensors', 'optimizer.safetensors', 'state.json'}:
        raise ValueError('invalid checkpoint manifest')
    for name, digest in manifest.items():
        if sha256(directory / name) != digest:
            raise ValueError(f'checkpoint checksum mismatch: {name}')
    state = json.loads((directory / 'state.json').read_text())
    if state.get('schema_version') != 1:
        raise ValueError('unsupported checkpoint version')
    return directory, state


def restore_checkpoint(root, policy, optimizer):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    directory, state = read_checkpoint(root)
    weights = mx.load(str(directory / 'policy.safetensors'))
    expected = dict(tree_flatten(policy.trainable_parameters()))
    if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in expected):
        raise ValueError('checkpoint parameters do not match policy')
    policy.update(tree_unflatten(list(weights.items())))
    optimizer.state = tree_unflatten(list(mx.load(str(directory / 'optimizer.safetensors')).items()))
    if len(state['mlx_rng_state']) != len(mx.random.state):
        raise ValueError('incompatible MLX RNG state')
    for current, saved in zip(mx.random.state, state['mlx_rng_state']):
        current[...] = mx.array(saved, dtype=mx.uint32)
    mx.eval(policy.parameters(), optimizer.state)
    return state
