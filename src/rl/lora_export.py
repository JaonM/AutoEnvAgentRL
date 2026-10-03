"""Native MLX adapter and optional merged/requantized model exports."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.utils import save_config, save_model


@contextmanager
def fused_model(model, *, requantize=False):
    original = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, LoRALinear)]
    rng = [x.tolist() for x in mx.random.state]
    try:
        model.update_modules(tree_unflatten([(name, layer.fuse(dequantize=not requantize)) for name, layer in original]))
        yield model
    finally:
        model.update_modules(tree_unflatten(original))
        for key, value in zip(mx.random.state, rng):
            key[...] = mx.array(value, dtype=mx.uint32)


def deployment_config(model, original):
    config = copy.deepcopy(original)
    config.pop('quantization', None)
    config.pop('quantization_config', None)
    quantized = {name: {'bits': layer.bits, 'group_size': layer.group_size, 'mode': layer.mode}
                 for name, layer in model.named_modules() if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding))}
    if quantized:
        quantization = {**next(iter(quantized.values())), **quantized}
        quantization.update({name: False for name, layer in model.named_modules() if isinstance(layer, (nn.Linear, nn.Embedding))})
        config['quantization'] = quantization
    config['torch_dtype'] = 'float32'
    return config


def validate(policy, paths, *, requantize=False):
    """Reload exports independently; quantization drift is measured, not hidden."""
    from mlx_lm import load
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    expected = policy.lm(ids)
    expected_logp = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
    mx.eval(expected, expected_logp)
    results = {}
    for kind, path in paths.items():
        model, _ = load(policy._base_path, adapter_path=path) if kind == 'adapter' else load(path)
        model.set_dtype(mx.float32)
        actual = model(ids)
        logp = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
        error = float(mx.max(mx.abs(actual - expected)))
        kl = float(mx.mean(mx.sum(mx.exp(expected_logp) * (expected_logp - logp), axis=-1)))
        import math
        if not math.isfinite(error) or not math.isfinite(kl):
            raise ValueError(f'non-finite {kind} export output')
        if kind == 'adapter' and error > .01:
            raise ValueError(f'native adapter reload mismatch: {error}')
        results[kind] = {'max_logit_error': error, 'mean_kl': kl,
                         'requantized': kind == 'merged' and requantize,
                         'scope': 'six fixed tokens; no task-quality claim'}
        del model
    return results


def export(policy, output, *, merge=False, requantize=False):
    if policy._tuning != 'lora':
        raise ValueError('LoRA export requires lora tuning')
    if requantize and not merge:
        raise ValueError('requantization requires a merged export')
    root = Path(output)
    adapter = root / 'lora_adapter'
    adapter.mkdir(parents=True, exist_ok=True)
    tensors = dict(tree_flatten(policy.lm.trainable_parameters()))
    if not tensors or any(not key.endswith(('.lora_a', '.lora_b')) for key in tensors):
        raise ValueError('unexpected trainable parameters in LoRA adapter')
    mx.save_safetensors(str(adapter / 'adapters.safetensors'), tensors)
    from .provenance import model_identity
    base_identity = model_identity(policy._base_path)[1]
    metadata = {'base_identity': base_identity, 'fine_tune_type': 'lora', 'num_layers': policy._lora_layers,
                'lora_parameters': policy._lora_config, 'base_model_name_or_path': policy._base_path,
                'thinking_mode': policy._thinking_mode, 'temperature': policy._temperature}
    (adapter / 'adapter_config.json').write_text(json.dumps(metadata, indent=2))
    paths = {'adapter': str(adapter.resolve())}
    if merge:
        destination = root / ('lora_requantized' if requantize else 'lora_merged')
        with fused_model(policy.lm, requantize=requantize):
            save_model(destination, policy.lm)
            save_config(deployment_config(policy.lm, policy._base_config), destination / 'config.json')
        policy.tokenizer.save_pretrained(destination)
        (destination / 'rl_generation.json').write_text(json.dumps({k: metadata[k] for k in ('thinking_mode', 'temperature')}, indent=2))
        paths['merged'] = str(destination.resolve())
    return paths
