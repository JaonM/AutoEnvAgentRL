"""Weight interpolation after SFT (WiSE-FT): weights = alpha * tuned + (1 - alpha) * base.

Fine-tuning a small model on narrow data erases general abilities; interpolating back toward the base
keeps most of the new skill and recovers much of what was lost. Both models must share the
architecture (full fine-tuning or fused LoRA). The tuned model's config, tokenizer and generation
config are copied.

    python experiments/v3/interpolate.py --base models/Qwen3-0.6B --tuned output/x/model --alpha 0.5 --output output/x/wise-0.5
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def load(folder):
    import mlx.core as mx
    weights = {}
    for path in sorted(Path(folder).glob('*.safetensors')):
        weights.update(mx.load(str(path)))
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base', required=True)
    parser.add_argument('--tuned', required=True)
    parser.add_argument('--alpha', type=float, required=True, help='weight of the tuned model (0..1)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import mlx.core as mx
    base, tuned = load(args.base), load(args.tuned)
    extra = set(tuned) - set(base)
    if extra:
        raise SystemExit(f'tuned weights missing from the base: {sorted(extra)[:5]}')
    # The base may carry lm_head.weight although embeddings are tied (mlx saves without it): ignore it.
    mixed = {}
    for name, value in tuned.items():
        reference = base[name]
        if value.shape != reference.shape:
            raise SystemExit(f'{name}: shape {value.shape} vs {reference.shape}')
        mixed[name] = (args.alpha * value.astype(mx.float32) + (1 - args.alpha) * reference.astype(mx.float32)
                       ).astype(value.dtype)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for path in Path(args.tuned).iterdir():
        if path.suffix != '.safetensors' and path.name != 'model.safetensors.index.json' and path.is_file():
            shutil.copy(path, out / path.name)
    mx.save_safetensors(str(out / 'model.safetensors'), mixed, metadata={'format': 'mlx'})
    print(f'alpha {args.alpha}: {len(mixed)} tensors -> {out}')


if __name__ == '__main__':
    main()
