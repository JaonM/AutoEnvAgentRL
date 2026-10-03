"""Weight-only affine QAT with FP32 masters and straight-through gradients."""
import mlx.core as mx
import mlx.nn as nn
from contextlib import contextmanager
from mlx.utils import tree_unflatten


def master_weight(base):
    if isinstance(base, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        if getattr(base, 'mode', 'affine') != 'affine':
            raise ValueError('QAT requires affine quantization')
        return mx.dequantize(base.weight, base.scales, base.biases,
                             group_size=base.group_size, bits=base.bits).astype(mx.float32)
    return base.weight.astype(mx.float32)


def fake_quantize(weight, *, bits=4, group_size=64):
    # Quantizer sees a detached master: backward is the identity STE, while
    # forward exactly uses the MLX deployment quantizer's reconstructed weights.
    packed, scales, biases = mx.quantize(mx.stop_gradient(weight), group_size=group_size, bits=bits)
    restored = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
    return weight + mx.stop_gradient(restored - weight)


class QATLinear(nn.Module):
    def __init__(self, base, *, bits=4, group_size=64):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.weight = master_weight(base)
        if 'bias' in base:
            self.bias = base.bias.astype(mx.float32)

    def __call__(self, x):
        weight = fake_quantize(self.weight, bits=self.bits, group_size=self.group_size)
        out = x @ weight.astype(x.dtype).T
        return out + self.bias.astype(x.dtype) if 'bias' in self else out

    def deployed(self):
        layer = nn.QuantizedLinear(self.weight.shape[1], self.weight.shape[0],
                                   bias='bias' in self, group_size=self.group_size, bits=self.bits)
        layer.weight, layer.scales, layer.biases = mx.quantize(
            self.weight, group_size=self.group_size, bits=self.bits)
        if 'bias' in self:
            layer.bias = self.bias
        return layer


class QATEmbedding(nn.Module):
    """Quantize the lookup table and its tied output projection identically."""
    def __init__(self, base, *, bits=4, group_size=64):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.weight = master_weight(base)

    def __call__(self, ids):
        return fake_quantize(self.weight, bits=self.bits, group_size=self.group_size)[ids]

    def as_linear(self, x):
        weight = fake_quantize(self.weight, bits=self.bits, group_size=self.group_size)
        return x @ weight.astype(x.dtype).T

    def deployed(self):
        layer = nn.QuantizedEmbedding(*self.weight.shape, group_size=self.group_size, bits=self.bits)
        layer.weight, layer.scales, layer.biases = mx.quantize(
            self.weight, group_size=self.group_size, bits=self.bits)
        return layer


QAT_MODULES = (QATLinear, QATEmbedding)


@contextmanager
def packed_model(model):
    """Temporarily use deployment operators without copying FP32 masters."""
    original = [(name, module) for name, module in model.named_modules() if isinstance(module, QAT_MODULES)]
    rng = [value.tolist() for value in mx.random.state]
    try:
        model.update_modules(tree_unflatten([(name, module.deployed()) for name, module in original]))
        yield model
    finally:
        model.update_modules(tree_unflatten(original))
        for key, value in zip(mx.random.state, rng):
            key[...] = mx.array(value, dtype=mx.uint32)


def install_qat(model, layers=1, bits=4, group_size=64, *, scope='projections'):
    if scope == 'full':
        replacements = []
        for name, module in model.named_modules():
            kind = (QATEmbedding if isinstance(module, (nn.Embedding, nn.QuantizedEmbedding))
                    else QATLinear if isinstance(module, (nn.Linear, nn.QuantizedLinear)) else None)
            if kind is not None:
                layer = kind(module, bits=bits, group_size=group_size)
                if layer.weight.shape[-1] % group_size:
                    raise ValueError(f'QAT group size {group_size} does not divide {name} weight width')
                replacements.append((name, layer))
        if not replacements:
            raise ValueError('full QAT requires linear or embedding modules')
        model.update_modules(tree_unflatten(replacements))
        from mlx.utils import tree_flatten
        if any(not mx.issubdtype(value.dtype, mx.floating) for _, value in tree_flatten(model.parameters())):
            raise ValueError('full QAT found unsupported packed/non-floating parameters')
        model.unfreeze()
        return [name for name, _ in replacements]
    if scope != 'projections':
        raise ValueError('unknown QAT scope')
    names = []
    if not 1 <= layers <= len(model.layers):
        raise ValueError('invalid QAT layer count')
    for index in range(len(model.layers) - layers, len(model.layers)):
        block = model.layers[index]
        replacements = []
        for name, module in block.named_modules():
            if name in {'self_attn.q_proj', 'self_attn.v_proj'}:
                replacements.append((name, QATLinear(module, bits=bits, group_size=group_size)))
                names.append(f'model.layers.{index}.{name}')
        block.update_modules(tree_unflatten(replacements))
    if not names:
        raise ValueError('QAT currently supports attention q_proj/v_proj models')
    return names


def install_packed_qat(model, layers=1, bits=4, group_size=64, *, scope='projections'):
    """Inference counterpart: materialize one master layer at a time, then discard it."""
    if scope == 'full':
        names = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.QuantizedEmbedding))]
    else:
        if not 1 <= layers <= len(model.layers):
            raise ValueError('invalid QAT layer count')
        selected = {id(m) for block in model.layers[-layers:] for n, m in block.named_modules()
                    if n in {'self_attn.q_proj', 'self_attn.v_proj'}}
        names = [(name, module) for name, module in model.named_modules() if id(module) in selected]
    rng = [key.tolist() for key in mx.random.state]
    try:
        for name, module in names:
            kind = QATEmbedding if isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)) else QATLinear
            packed = kind(module, bits=bits, group_size=group_size).deployed()
            mx.eval(packed.parameters())
            model.update_modules(tree_unflatten([(name, packed)]))
    finally:
        for key, value in zip(mx.random.state, rng):
            key[...] = mx.array(value, dtype=mx.uint32)
    model.freeze()
