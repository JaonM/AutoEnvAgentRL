"""Read weight headers before allocating a full-parameter MLX training run."""
import json
import math
from pathlib import Path
import struct


def full_training_memory(model, rollout_workers, *, qat=False, packed_inference=False, bits=4, group_size=64):
    root = Path(model)
    config = json.loads((root / 'config.json').read_text())
    quantization = config.get('quantization') or config.get('quantization_config') or {}
    if quantization and not qat:
        raise ValueError('full tuning requires floating-point base weights; use the original model, not a quantized checkpoint')
    parameters = 0
    matrix_parameters = 0
    seen = set()
    tensors = {}
    files = sorted(root.glob('model*.safetensors'))
    if not files:
        raise ValueError('full tuning requires model*.safetensors weights')
    index_path = root / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    if weight_map is not None and set(weight_map.values()) != {path.name for path in files}:
        raise ValueError('incomplete or inconsistent model weight shards')
    for path in files:
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 100 * 1024 * 1024:
                raise ValueError('invalid safetensors header size')
            header = json.loads(stream.read(size))
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if name in seen:
                raise ValueError(f'duplicate base tensor: {name}')
            seen.add(name)
            tensors[name] = tensor
    if weight_map is not None and seen != set(weight_map):
        raise ValueError('weight headers do not match model index')
    packed_modules = {name[:-7] for name, t in tensors.items() if name.endswith('.weight') and t['dtype'] == 'U32'}
    for name, tensor in tensors.items():
        if name == 'lm_head.weight' and config.get('tie_word_embeddings'):
            continue
        module, _, field = name.rpartition('.')
        if qat and module in packed_modules and field in {'scales', 'biases'}:
            continue  # Recomputed from FP32 masters, not independent trainable parameters.
        count = math.prod(tensor['shape'])
        if tensor['dtype'] not in {'F16', 'BF16', 'F32', 'F64'}:
            settings = quantization.get(module, quantization)
            if not (qat and field == 'weight' and tensor['dtype'] == 'U32' and
                    isinstance(settings, dict) and settings.get('bits') in {4, 8} and
                    settings.get('mode', 'affine') == 'affine' and
                    module + '.scales' in tensors and module + '.biases' in tensors):
                raise ValueError('full tuning requires floating-point tensors; full QAT also supports MLX affine 4/8-bit weights')
            count *= 32 // settings['bits']
        parameters += count
        if len(tensor['shape']) == 2 and field == 'weight':
            matrix_parameters += count
    # Actor + reference + workers + gradient + two Adam moments. Activations,
    # KV caches, critic, temporary gradients and allocator overhead are extra.
    lower_bound = parameters * 4 * (5 + rollout_workers)
    if qat and packed_inference:
        packed_bytes = matrix_parameters * (bits / 8 + 8 / group_size) + (parameters - matrix_parameters) * 4
        lower_bound = math.ceil(parameters * 16 + packed_bytes * (1 + rollout_workers))
    return {'parameters': parameters, 'training_dtype': 'float32', 'qat': qat, 'packed_inference': packed_inference,
            'rollout_workers': rollout_workers, 'persistent_lower_bound_bytes': lower_bound,
            'excludes': ['activations', 'KV caches', 'critic', 'temporary buffers']}


def check_full_training_memory(model, rollout_workers, *, available_bytes=None, qat=False, packed_inference=False, bits=4):
    if available_bytes is None:
        import psutil
        available_bytes = psutil.virtual_memory().available
    estimate = full_training_memory(model, rollout_workers, qat=qat, packed_inference=packed_inference, bits=bits)
    estimate['available_bytes'] = available_bytes
    if estimate['persistent_lower_bound_bytes'] > available_bytes:
        label = 'full QAT' if qat else 'full tuning'
        raise MemoryError(f"{label} needs at least {estimate['persistent_lower_bound_bytes']/1e9:.1f} GB "
                          f"before activations/KV/temporary buffers; available {available_bytes/1e9:.1f} GB. "
                          'Use a larger-memory machine, a smaller model, LoRA or projection-only QAT.')
    return estimate
