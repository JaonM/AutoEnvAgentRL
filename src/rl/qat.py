"""Weight-only affine QAT with FP32 masters and straight-through gradients."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


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
        weight = (mx.dequantize(base.weight, base.scales, base.biases,
                               group_size=base.group_size, bits=base.bits)
                  if isinstance(base, nn.QuantizedLinear) else base.weight)
        self.weight = weight.astype(mx.float32)
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


def install_qat(model, layers=1, bits=4, group_size=64):
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
