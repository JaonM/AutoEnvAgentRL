"""MLX-LM policy/critic adapter; generation and scoring use identical policies."""
from pathlib import Path
import hashlib
import json
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from .qat import install_qat, QATLinear
from .errors import ContextBudgetExceeded


class Policy(nn.Module):
    def __init__(self, model, tuning='qat', layers=1, rank=8, bits=4, group_size=64, temperature=1.0, *, critic=False):
        super().__init__()
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature
        self.lm, tokenizer = load(model)
        self.lm.freeze()
        if tuning == 'qat':
            install_qat(self.lm, layers, bits, group_size)
        elif tuning == 'lora':
            linear_to_lora_layers(self.lm, layers, {'rank': rank, 'scale': 16., 'dropout': 0.,
                                                  'keys': ['self_attn.q_proj', 'self_attn.v_proj']})
        else:
            raise ValueError('unknown tuning mode')
        # Packed integer weights remain quantized. FP32 floating parameters
        # keep cached generation and full-sequence training numerically aligned;
        # BF16 activations produced >0.2 log-probability discrepancies on Qwen3.
        self.lm.set_dtype(mx.float32)
        self._has_critic = critic
        if critic:
            self.critic = nn.Linear(self.lm.args.hidden_size, 1)
            self.critic.weight = mx.zeros_like(self.critic.weight)
            self.critic.bias = mx.zeros_like(self.critic.bias)
        self._tokenizer = tokenizer
        self.eval()
        mx.eval(self.parameters())

    @property
    def tokenizer(self):
        return self._tokenizer

    def _prefill(self, prompt):
        cache = make_prompt_cache(self.lm)
        for start in range(0, prompt.shape[1] - 1, 512):
            self.lm.model(prompt[:, start:min(start + 512, prompt.shape[1] - 1)], cache=cache)
        return cache

    def _step(self, tokens, cache):
        logp, values = self._batch_step(tokens, cache)
        return logp[0], values[0] if values is not None else None

    def _batch_step(self, tokens, cache):
        hidden = self.lm.model(tokens, cache=cache)[:, -1:]
        logits = (self.lm.model.embed_tokens.as_linear(hidden)
                  if self.lm.args.tie_word_embeddings else self.lm.lm_head(hidden))
        logits = logits.astype(mx.float32)[:, 0, :] / self._temperature
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        value = self.critic(hidden.astype(mx.float32))[:, 0, 0] if self._has_critic else None
        return logp, value

    def batch_decoder(self, *, greedy=False):
        from .batch_decode import BatchDecoder
        return BatchDecoder(self, greedy=greedy)

    def cached_token_stats(self, tokens, prompt_length):
        if tokens.shape[0] != 1 or not 0 < prompt_length < tokens.shape[1]:
            raise ValueError('one nonempty prompt/completion per scoring call required')
        cache = self._prefill(tokens[:, :prompt_length])
        logps, values = [], []
        for position in range(prompt_length - 1, tokens.shape[1] - 1):
            logp, value = self._step(tokens[:, position:position+1], cache)
            logps.append(logp[tokens[0, position+1]].reshape(1))
            if value is not None:
                values.append(value.reshape(1))
        return mx.concatenate(logps), mx.concatenate(values) if values else None

    def batch_token_stats(self, tokens, prompt_lengths, response_lengths):
        """Right-padded causal batch; project only response prediction positions."""
        width = max(response_lengths)
        hidden = self.lm.model(tokens[:, :-1])
        positions = mx.array(prompt_lengths)[:, None] - 1 + mx.arange(width)[None, :]
        positions = mx.minimum(positions, hidden.shape[1] - 1)
        selected = mx.take_along_axis(hidden, positions[:, :, None], axis=1)
        logits = (self.lm.model.embed_tokens.as_linear(selected)
                  if self.lm.args.tie_word_embeddings else self.lm.lm_head(selected))
        logits = logits.astype(mx.float32) / self._temperature
        targets = mx.take_along_axis(tokens, positions + 1, axis=1)
        logp = mx.take_along_axis(logits, targets[:, :, None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)
        values = self.critic(selected.astype(mx.float32)).squeeze(-1) if self._has_critic else None
        return logp, values

    def token_stats(self, tokens, prompt_length):
        if tokens.shape[0] != 1 or not 0 < prompt_length < tokens.shape[1]:
            raise ValueError('one nonempty prompt/completion per scoring call required')
        logp, values = self.batch_token_stats(tokens, [prompt_length], [tokens.shape[1] - prompt_length])
        return logp[0], values[0] if values is not None else None

    def value(self, prompt, max_context=None):
        if not self._has_critic:
            raise ValueError("value estimation requires a PPO critic")
        if max_context is not None and len(prompt) > max_context:
            raise ContextBudgetExceeded('bootstrap context budget exceeded')
        ids = mx.array([prompt])
        _, value = self._step(ids[:, -1:], self._prefill(ids))
        return float(value)

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def sample(self, messages, max_tokens=256, max_context=4096, greedy=False):
        iterator = self.iter_sample(messages,max_tokens,max_context,greedy=greedy)
        while True:
            try:
                next(iterator)
            except StopIteration as result:
                return result.value

    def iter_sample(self, messages, max_tokens=256, max_context=4096, greedy=False):
        """Yield at token boundaries; each iterator owns an independent KV cache."""
        prompt = self.encode(messages)
        if len(prompt) + max_tokens > max_context:
            raise ContextBudgetExceeded('policy context budget exceeded; refusing silent context truncation')
        tokens, old_logp = [], []
        compute_seconds = 0.
        tick_started = time.perf_counter()
        started = time.monotonic()
        ids = mx.array([prompt])
        cache = self._prefill(ids)
        current = ids[:, -1:]
        # Logits are already temperature-scaled in the shared scoring path.
        sampler = make_sampler(temp=0.0 if greedy else 1.0)
        for _ in range(max_tokens):
            logprobs, _ = self._step(current, cache)
            token = int(sampler(logprobs[None])[0])
            tokens.append(token)
            old_logp.append(float(logprobs[token]))
            compute_seconds += time.perf_counter() - tick_started
            yield None
            if token in self.tokenizer.eos_token_ids:
                break
            tick_started = time.perf_counter()
            current = mx.array([[token]])
        verification_started = time.perf_counter()
        scored_logp, values = self.cached_token_stats(mx.array([prompt + tokens]), len(prompt))
        discrepancy = float(mx.max(mx.abs(scored_logp - mx.array(old_logp))))
        if not discrepancy <= .001:
            raise ValueError(f'cached sampling / training log-probability discrepancy: {discrepancy}')
        old_values = values.tolist() if values is not None else None
        return {'prompt': prompt, 'tokens': tokens, 'old_logp': old_logp,
                **({'old_values': old_values} if old_values is not None else {}),
                'generation_finish': 'eos' if tokens[-1] in self.tokenizer.eos_token_ids else 'token_limit',
                'behavior_scoring_max_logp_error': discrepancy,
                'text': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'generation_seconds': time.monotonic() - started,
                'generation_compute_seconds': compute_seconds,
                'verification_seconds': time.perf_counter() - verification_started}

    def save_snapshot(self, root, version):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'policy-{version:06d}.safetensors'
        temporary = root / f'.policy-{version:06d}.safetensors'
        mx.save_safetensors(str(temporary), dict(tree_flatten(self.trainable_parameters())))
        temporary.replace(path)
        return str(path)

    def restore(self, path, *, policy_only=False):
        weights = mx.load(str(path))
        if policy_only:
            weights = {name: value for name, value in weights.items() if not name.startswith('critic.')}
        expected = {name for name, _ in tree_flatten(self.trainable_parameters())}
        if set(weights) != expected:
            raise ValueError('snapshot parameter keys do not match policy')
        self.update(tree_unflatten(list(weights.items())))
        mx.eval(self.parameters())

    def digest(self, *, policy_only=False, effective=False):
        digest = hashlib.sha256()
        parameters = self.lm.trainable_parameters() if policy_only else self.trainable_parameters()
        if effective:
            parameters = {name: layer.deployed().parameters() for name, layer in self.lm.named_modules()
                          if isinstance(layer, QATLinear)}
            if not parameters:
                raise ValueError('effective QAT digest requires QAT layers')
        for name, value in tree_flatten(parameters):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode() + memoryview(value).tobytes())
        return digest.hexdigest()

    def export_qat(self, output):
        """Export selected layers in the actual packed deployment format."""
        weights, description = {}, {}
        for name, layer in self.lm.named_modules():
            if isinstance(layer, QATLinear):
                packed = layer.deployed()
                for field, value in tree_flatten(packed.parameters()):
                    weights[f'{name}.{field}'] = value
                description[name] = {'bits': layer.bits, 'group_size': layer.group_size}
        if weights:
            mx.save_safetensors(str(Path(output) / 'qat_quantized.safetensors'), weights)
            (Path(output) / 'qat_quantization.json').write_text(json.dumps(description, indent=2))
        return description

    def load_qat_export(self, output):
        """Replace fake quantization with the exported packed Metal operators.

        Load on the same base model/configuration; the export contains only
        tuned projection layers, not a standalone copy of the base model.
        """
        root = Path(output)
        description = json.loads((root / 'qat_quantization.json').read_text())
        weights = mx.load(str(root / 'qat_quantized.safetensors'))
        modules = dict(self.lm.named_modules())
        replacements = []
        expected = set()
        for name, settings in description.items():
            base = modules.get(name)
            if not isinstance(base, QATLinear):
                raise ValueError(f'export layer does not match QAT configuration: {name}')
            packed = nn.QuantizedLinear(base.weight.shape[1], base.weight.shape[0],
                bias='bias' in base, bits=settings['bits'], group_size=settings['group_size'])
            for field, value in tree_flatten(packed.parameters()):
                key = f'{name}.{field}'
                expected.add(key)
                if key not in weights or weights[key].shape != value.shape:
                    raise ValueError(f'invalid exported tensor: {key}')
            packed.update(tree_unflatten([(key[len(name)+1:], value)
                for key, value in weights.items() if key.startswith(name + '.')]))
            replacements.append((name, packed))
        if not expected or set(weights) != expected:
            raise ValueError('export tensor keys do not match layer metadata')
        self.lm.update_modules(tree_unflatten(replacements))
        mx.eval(self.parameters())
        return description
