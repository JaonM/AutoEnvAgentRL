"""Portable MLX affine/LoRA -> Hugging Face/PEFT conversion (no MLX dependency)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def dequantize(weight, scales, biases, *, bits, group_size):
    import torch
    if bits not in (4, 8) or weight.dtype != torch.uint32 or group_size < 1:
        raise ValueError('only MLX affine uint32 4/8-bit packing is supported')
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    unpacked = ((weight.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten(-2).float()
    if unpacked.shape[-1] % group_size or scales.shape != biases.shape or scales.shape != (*unpacked.shape[:-1], unpacked.shape[-1] // group_size):
        raise ValueError('packed weight/scales/biases shape mismatch')
    return (unpacked.reshape(*scales.shape, group_size) * scales.float().unsqueeze(-1)
            + biases.float().unsqueeze(-1)).flatten(-2)


class Weights:
    def __init__(self, root, *, overlay=None):
        from safetensors import safe_open
        self.locations = {}
        files = sorted(Path(root).glob('model*.safetensors'))
        if not files:
            raise ValueError('model directory requires model*.safetensors')
        for path in files:
            with safe_open(path, framework='pt', device='cpu') as stream:
                for key in stream.keys():
                    if key in self.locations:
                        raise ValueError(f'duplicate tensor {key}')
                    self.locations[key] = path
        index = Path(root) / 'model.safetensors.index.json'
        if index.exists():
            declared = json.loads(index.read_text())['weight_map']
            if set(declared) != set(self.locations) or any(self.locations[k].name != v for k,v in declared.items()):
                raise ValueError('model index does not match available shards')
        if overlay:
            with safe_open(overlay, framework='pt', device='cpu') as stream:
                for key in stream.keys():
                    self.locations[key] = Path(overlay)

    def get(self, key):
        from safetensors import safe_open
        with safe_open(self.locations[key], framework='pt', device='cpu') as stream:
            return stream.get_tensor(key)


class Shards:
    def __init__(self, root, max_bytes):
        self.root, self.max_bytes = root, max_bytes
        root.mkdir(parents=True)
        self.pending, self.size, self.files, self.total = {}, 0, [], 0

    def add(self, name, tensor):
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.size + size > self.max_bytes:
            self.flush()
        self.pending[name] = tensor.contiguous()
        self.size += size
        self.total += size

    def flush(self):
        from safetensors.torch import save_file
        if self.pending:
            path = self.root / f'part-{len(self.files):05d}.safetensors'
            save_file(self.pending, path, metadata={'format': 'pt'})
            self.files.append((path, list(self.pending)))
            self.pending, self.size = {}, 0

    def finish(self):
        self.flush()
        mapping = {}
        for index, (path, keys) in enumerate(self.files, 1):
            name = 'model.safetensors' if len(self.files) == 1 else f'model-{index:05d}-of-{len(self.files):05d}.safetensors'
            path.rename(self.root / name)
            mapping.update({key: name for key in keys})
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': self.total}, 'weight_map': mapping}, indent=2))


def copy_tokenizer(source, destination):
    for pattern in ('*token*.json', 'vocab.json', 'merges.txt', '*.model', '*.tiktoken', '*.jinja', 'generation_config.json', 'rl_generation.json'):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copy2(path, destination / path.name)


def convert(model, output, *, adapter=None, qat_export=None, format='merged', dtype='bfloat16', max_shard_size_mb=1024):
    import torch
    from safetensors.torch import load_file, save_file
    from .provenance import model_identity
    from .checkpoint import sha256
    model, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('output must be a new directory')
    if output.is_relative_to(model):
        raise ValueError('output must be outside the source model directory')
    if format not in {'merged', 'peft'} or dtype not in {'float32', 'float16', 'bfloat16'} or max_shard_size_mb < 1:
        raise ValueError('invalid conversion options')
    if format == 'peft' and not adapter:
        raise ValueError('PEFT output requires an adapter')
    if adapter and qat_export:
        raise ValueError('adapter and QAT overlay cannot be combined')
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3':
        raise ValueError('this converter currently validates Qwen3 architecture only')
    source_identity = model_identity(str(model))[1]
    lora, adapter_config = {}, None
    if adapter:
        adapter = Path(adapter).resolve()
        adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
        if adapter_config.get('fine_tune_type') != 'lora' or adapter_config.get('base_identity') != source_identity:
            raise ValueError('adapter/base identity mismatch; export the adapter with this framework first')
        lora = load_file(adapter / 'adapters.safetensors')
        if not lora or any(not key.endswith(('.lora_a', '.lora_b')) for key in lora):
            raise ValueError('invalid native LoRA tensors')
    quant = config.get('quantization') or config.get('quantization_config') or {}
    overlay = None
    if qat_export:
        qat_export = Path(qat_export).resolve()
        identity = json.loads((qat_export / 'qat_base_identity.json').read_text())
        if identity['base_identity'] != source_identity:
            raise ValueError('QAT overlay/base identity mismatch')
        settings = json.loads((qat_export / 'qat_quantization.json').read_text())
        if not settings:
            raise ValueError('empty QAT overlay')
        quant = {**quant, **settings}
        overlay = qat_export / 'qat_quantized.safetensors'
        from safetensors import safe_open
        original = Weights(model)
        with safe_open(overlay, framework='pt', device='cpu') as stream:
            expected = {name + '.' + field for name in settings for field in ('weight','scales','biases')}
            expected.update(name + '.bias' for name in settings if name + '.bias' in original.locations)
            if set(stream.keys()) != expected or any(name + '.weight' not in original.locations for name in settings):
                raise ValueError('QAT overlay tensors do not match base/modules')
    weights = Weights(model, overlay=overlay)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    used = set()
    try:
        base = staging / 'base_model' if format == 'peft' else staging
        # Staging itself already exists; shard writer expects a new directory.
        if base == staging:
            base = staging / 'model'
        writer = Shards(base, max_shard_size_mb * 1024**2)
        for key in sorted(weights.locations):
            module, _, field = key.rpartition('.')
            if field in {'scales', 'biases'} and module + '.weight' in weights.locations:
                continue
            if config.get('tie_word_embeddings') and key == 'lm_head.weight':
                continue
            value = weights.get(key)
            if value.dtype == torch.uint32:
                spec = quant.get(module, quant)
                if field != 'weight' or not isinstance(spec, dict) or spec.get('mode', 'affine') != 'affine':
                    raise ValueError(f'unsupported packed tensor {key}')
                value = dequantize(value, weights.get(module + '.scales'), weights.get(module + '.biases'),
                                   bits=spec['bits'], group_size=spec['group_size'])
            if not value.is_floating_point():
                raise ValueError(f'unexpected non-floating tensor {key}')
            a, b = module + '.lora_a', module + '.lora_b'
            if field == 'weight' and a in lora:
                rank = adapter_config['lora_parameters']['rank']
                if b not in lora or tuple(lora[a].shape) != (value.shape[1], rank) or tuple(lora[b].shape) != (rank, value.shape[0]):
                    raise ValueError(f'LoRA shape mismatch: {module}')
                used.update((a,b))
                if format == 'merged':
                    value = value.float() + (adapter_config['lora_parameters']['scale'] * lora[b].float().T) @ lora[a].float().T
            writer.add(key, value.to(getattr(torch, dtype)))
        if used != set(lora):
            raise ValueError('unused or incomplete adapter tensors')
        writer.finish()
        config.pop('quantization', None)
        config.pop('quantization_config', None)
        config.pop('_name_or_path', None)
        config.update(torch_dtype=dtype, dtype=dtype, architectures=['Qwen3ForCausalLM'])
        (base / 'config.json').write_text(json.dumps(config, indent=2))
        copy_tokenizer(model, base)
        generation = adapter_config if adapter_config else identity if qat_export else None
        if generation:
            (base / 'rl_generation.json').write_text(json.dumps({k:generation[k] for k in ('thinking_mode','temperature')},indent=2))
        if format == 'peft':
            destination = staging / 'adapter'; destination.mkdir()
            tensors = {}
            modules = []
            for key in sorted(lora):
                name, _, field = key.rpartition('.')
                converted = 'lora_A' if field == 'lora_a' else 'lora_B'
                tensors[f'base_model.model.{name}.{converted}.weight'] = lora[key].float().T.contiguous()
                if field == 'lora_a': modules.append(name)
            parameters = adapter_config['lora_parameters']
            description = {'peft_type':'LORA', 'task_type':'CAUSAL_LM', 'inference_mode':True,
                'base_model_name_or_path':str(output / 'base_model'), 'r':parameters['rank'],
                'lora_alpha':parameters['scale'] * parameters['rank'], 'lora_dropout':parameters['dropout'],
                'bias':'none', 'target_modules':modules}
            save_file(tensors, destination / 'adapter_model.safetensors', metadata={'format':'pt'})
            (destination / 'adapter_config.json').write_text(json.dumps(description, indent=2))
        else:
            for path in base.iterdir():path.rename(staging / path.name)
            base.rmdir()
        manifest = {'format':format,'dtype':dtype,'source_identity':source_identity,
            'source_model':str(model),'source_adapter':str(adapter) if adapter else None,
            'source_qat_export':str(qat_export) if qat_export else None,
            'semantics':'MLX affine weights dequantized to standard floating weights; not NF4/AWQ/GPTQ conversion',
            'optimizer_state_converted':False,
            'files':{str(p.relative_to(staging)):sha256(p) for p in staging.rglob('*') if p.is_file()}}
        (staging / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--qat-export')
    parser.add_argument('--format', choices=['merged','peft'], default='merged')
    parser.add_argument('--dtype', choices=['float32','float16','bfloat16'], default='bfloat16')
    parser.add_argument('--max-shard-size-mb', type=int, default=1024)
    args=vars(parser.parse_args())
    print(convert(**args))


if __name__ == '__main__':
    main()
