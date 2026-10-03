"""MLX-LM policy/critic adapter; generation and scoring use identical policies."""
from pathlib import Path
import copy
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

from .qat import install_qat, QATLinear, QAT_MODULES, packed_model
from .errors import ContextBudgetExceeded
from .chat import completion


class Policy(nn.Module):
    def __init__(self, model, tuning='qat', layers=1, rank=8, bits=4, group_size=64, temperature=1.0, *, critic=False, thinking_mode='auto', qat_scope='projections', lora_targets='self_attn.q_proj,self_attn.v_proj', lora_scale=16., lora_dropout=0., gradient_checkpointing=False, logits_chunk_size=128, prefill_chunk_size=512, prefix_cache_tokens=0, packed_inference=False, inference_only=False, cache_inference=False):
        super().__init__()
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature
        if thinking_mode not in {'auto', 'thinking', 'no-thinking'}:
            raise ValueError('unknown thinking mode')
        self._thinking_mode = thinking_mode
        self._tuning = tuning
        if qat_scope not in {'projections', 'full'} or (qat_scope == 'full' and tuning != 'qat'):
            raise ValueError('qat_scope full requires tuning qat')
        self._qat_scope = qat_scope
        if lora_dropout != 0:
            raise ValueError('RL requires lora_dropout=0 for deterministic behavior/target probabilities')
        if lora_scale <= 0 or min(logits_chunk_size, prefill_chunk_size) < 1 or prefix_cache_tokens < 0:
            raise ValueError('invalid policy memory/LoRA options')
        self._logits_chunk_size = logits_chunk_size
        self._prefill_chunk_size = prefill_chunk_size
        self._prefix_cache_tokens = prefix_cache_tokens if cache_inference else 0
        from collections import OrderedDict
        self._prefix_cache = OrderedDict()
        self._prefill_stats = {'reused_tokens': 0, 'cache_hits': 0}
        self._inference_only = inference_only
        self._publish_packed = packed_inference and tuning == 'qat'
        self._base_path = str(model)
        self._lora_config = {'rank': rank, 'scale': lora_scale, 'dropout': lora_dropout,
                             'keys': [s.strip() for s in lora_targets.split(',') if s.strip()]}
        self._lora_layers = layers
        self.lm, tokenizer, self._base_config = load(model, return_config=True)
        if lora_targets in {'attention', 'all-linear'}:
            self._lora_config['keys'] = [name for name, module in self.lm.layers[-1].named_modules()
                if isinstance(module, (nn.Linear, nn.QuantizedLinear)) and
                (lora_targets == 'all-linear' or name.startswith('self_attn.'))]
        self.lm.freeze()
        if tuning == 'qat':
            if inference_only:
                from .qat import install_packed_qat
                install_packed_qat(self.lm, layers, bits, group_size, scope=qat_scope)
            else:
                install_qat(self.lm, layers, bits, group_size, scope=qat_scope)
        elif tuning == 'lora':
            if not 1 <= layers <= len(self.lm.layers) or not self._lora_config['keys']:
                raise ValueError('invalid LoRA layer count or targets')
            for block in self.lm.layers[-layers:]:
                available = dict(block.named_modules())
                if any(key not in available for key in self._lora_config['keys']):
                    raise ValueError('LoRA target not found in selected transformer layers')
            linear_to_lora_layers(self.lm, layers, self._lora_config)
        elif tuning == 'full':
            if any(not mx.issubdtype(value.dtype, mx.floating) for _, value in tree_flatten(self.lm.parameters())):
                raise ValueError('full tuning requires floating-point base weights; quantized tensors cannot be trained in full mode')
            self.lm.unfreeze()
        elif tuning == 'inference':
            pass
        else:
            raise ValueError('unknown tuning mode')
        # Packed integer weights remain quantized. FP32 floating parameters
        # keep cached generation and full-sequence training numerically aligned;
        # BF16 activations produced >0.2 log-probability discrepancies on Qwen3.
        self.lm.set_dtype(mx.float32)
        if gradient_checkpointing and not inference_only:
            from .activation_checkpoint import install
            install(self.lm)
        self._has_critic = critic
        if critic:
            self.critic = nn.Linear(self.lm.args.hidden_size, 1)
            self.critic.weight = mx.zeros_like(self.critic.weight)
            self.critic.bias = mx.zeros_like(self.critic.bias)
        self._tokenizer = tokenizer
        if thinking_mode != 'auto' and 'enable_thinking' not in (tokenizer.chat_template or ''):
            raise ValueError('explicit thinking mode requires a tokenizer template with enable_thinking')
        self.eval()
        mx.eval(self.parameters())

    @property
    def tokenizer(self):
        return self._tokenizer

    def _iter_prefill(self, prompt):
        if self._prefix_cache_tokens:
            from .prefix_cache import iter_prefill
            return (yield from iter_prefill(self, prompt))
        cache = make_prompt_cache(self.lm)
        for start in range(0, prompt.shape[1] - 1, self._prefill_chunk_size):
            stop = min(start + self._prefill_chunk_size, prompt.shape[1] - 1)
            self.lm.model(prompt[:, start:stop], cache=cache)
            mx.eval([c.state for c in cache])
            if stop < prompt.shape[1] - 1:
                yield None
        return cache

    def _prefill(self, prompt):
        iterator = self._iter_prefill(prompt)
        while True:
            try:
                next(iterator)
            except StopIteration as done:
                return done.value

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

    def cached_token_stats(self, tokens, prompt_length, *, return_entropy=False):
        if tokens.shape[0] != 1 or not 0 < prompt_length < tokens.shape[1]:
            raise ValueError('one nonempty prompt/completion per scoring call required')
        cache = self._prefill(tokens[:, :prompt_length])
        logps, values, entropies = [], [], []
        for position in range(prompt_length - 1, tokens.shape[1] - 1):
            logp, value = self._step(tokens[:, position:position+1], cache)
            logps.append(logp[tokens[0, position+1]].reshape(1))
            if return_entropy:
                entropies.append(self._entropy(logp).reshape(1))
            if value is not None:
                values.append(value.reshape(1))
        result = (mx.concatenate(logps), mx.concatenate(values) if values else None)
        return (*result, mx.concatenate(entropies)) if return_entropy else result

    def batch_token_stats(self, tokens, prompt_lengths, response_lengths, *, return_entropy=False):
        """Right-padded causal batch; project only response prediction positions."""
        width = max(response_lengths)
        hidden = self.lm.model(tokens[:, :-1])
        positions = mx.array(prompt_lengths)[:, None] - 1 + mx.arange(width)[None, :]
        positions = mx.minimum(positions, hidden.shape[1] - 1)
        selected = mx.take_along_axis(hidden, positions[:, :, None], axis=1)
        targets = mx.take_along_axis(tokens, positions + 1, axis=1)
        head = self.lm.model.embed_tokens if self.lm.args.tie_word_embeddings else self.lm.lm_head
        def project(parameters, hidden, target):
            head.update(parameters)
            target = mx.stop_gradient(target)
            logits = head.as_linear(hidden) if self.lm.args.tie_word_embeddings else head(hidden)
            distribution = logits.astype(mx.float32) / self._temperature
            distribution = distribution - mx.logsumexp(distribution, axis=-1, keepdims=True)
            return (mx.take_along_axis(distribution, target[:, :, None], axis=-1).squeeze(-1),
                    self._entropy(distribution) if return_entropy else mx.zeros(target.shape))
        scores, entropies = [], []
        # Recompute each vocabulary projection during backward, retaining only
        # token statistics rather than all response-by-vocabulary activations.
        for start in range(0, width, self._logits_chunk_size):
            stop = start + self._logits_chunk_size
            p, e = mx.checkpoint(project)(head.trainable_parameters(), selected[:, start:stop], targets[:, start:stop])
            scores.append(p)
            entropies.append(e)
        logp = mx.concatenate(scores, axis=1)
        values = self.critic(selected.astype(mx.float32)).squeeze(-1) if self._has_critic else None
        if return_entropy:
            return logp, values, mx.concatenate(entropies, axis=1)
        return logp, values

    @staticmethod
    def _entropy(logp):
        """Full-vocabulary entropy in nats; diagnostic only, with no gradient."""
        logp = mx.stop_gradient(logp)
        return -mx.sum(mx.exp(logp) * mx.where(mx.isfinite(logp), logp, 0.), axis=-1)

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

    def encode(self, messages, *, tools=None):
        if tools and (not self.tokenizer.has_tool_calling or not callable(self.tokenizer.tool_parser)):
            raise ValueError('policy tokenizer must support native tool calling')
        options = {'tools': tools} if tools is not None else {}
        if self._thinking_mode != 'auto':
            options['enable_thinking'] = self._thinking_mode == 'thinking'
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)

    def sample(self, messages, max_tokens=256, max_context=4096, greedy=False, *, tools=None):
        iterator = self.iter_sample(messages,max_tokens,max_context,greedy=greedy,tools=tools)
        while True:
            try:
                next(iterator)
            except StopIteration as result:
                return result.value

    def iter_sample(self, messages, max_tokens=256, max_context=4096, greedy=False, *, tools=None):
        """Yield at token boundaries; each iterator owns an independent KV cache."""
        prompt = self.encode(messages, tools=tools)
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
                **completion(self.tokenizer, tokens, tools),
                'generation_seconds': time.monotonic() - started,
                'generation_compute_seconds': compute_seconds,
                'verification_seconds': time.perf_counter() - verification_started}

    def save_snapshot(self, root, version):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'policy-{version:06d}.safetensors'
        temporary = root / f'.policy-{version:06d}.safetensors'
        mx.save_safetensors(str(temporary), dict(tree_flatten(self.trainable_parameters())))
        if self._publish_packed:
            packed_root = root / 'inference'
            packed_root.mkdir(exist_ok=True)
            packed_temporary = packed_root / temporary.name
            with packed_model(self.lm):
                mx.save_safetensors(str(packed_temporary), dict(tree_flatten(self.parameters())))
            packed_temporary.replace(packed_root / path.name)
        temporary.replace(path)
        return str(path)

    def restore(self, path, *, policy_only=False):
        self._prefix_cache.clear()
        if self._inference_only:
            path = Path(path).parent / 'inference' / Path(path).name
        weights = mx.load(str(path))
        if policy_only:
            weights = {name: value for name, value in weights.items() if not name.startswith('critic.')}
        expected = {name for name, _ in tree_flatten(self.parameters() if self._inference_only else self.trainable_parameters())}
        if set(weights) != expected:
            raise ValueError('snapshot parameter keys do not match policy')
        self.update(tree_unflatten(list(weights.items())))
        mx.eval(self.parameters())

    def digest(self, *, policy_only=False, effective=False):
        if effective and self._tuning == 'qat' and self._qat_scope == 'full':
            with packed_model(self.lm):
                digest = hashlib.sha256()
                for name, value in tree_flatten(self.lm.parameters()):
                    digest.update(name.encode())
                    digest.update(str(value.dtype).encode() + memoryview(value).tobytes())
                return digest.hexdigest()
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

    def export_full(self, output):
        """Export complete, unquantized language-model weights without the PPO critic."""
        if self._tuning != 'full':
            raise ValueError('complete-model export requires full tuning')
        from mlx_lm.utils import save_model, save_config
        root = Path(output)
        save_model(root, self.lm)
        config = copy.deepcopy(self._base_config)
        config['torch_dtype'] = 'float32'
        save_config(config, root / 'config.json')
        self.tokenizer.save_pretrained(root)
        (root / 'rl_generation.json').write_text(json.dumps({
            'thinking_mode': self._thinking_mode, 'temperature': self._temperature}, indent=2))
        return str(root.resolve())

    def export_qat(self, output):
        """Export packed projections, or a standalone model for full QAT."""
        if self._tuning == 'qat' and self._qat_scope == 'full':
            from mlx_lm.utils import save_model, save_config
            root = Path(output) / 'qat_model'
            description = {name: {'bits': layer.bits, 'group_size': layer.group_size}
                           for name, layer in self.lm.named_modules() if isinstance(layer, QAT_MODULES)}
            if not description:
                raise ValueError('full QAT export requires trainable QAT modules')
            with packed_model(self.lm):
                save_model(root, self.lm)
            config = copy.deepcopy(self._base_config)
            # Replace source quantization settings, including per-layer overrides.
            config['quantization'] = {'bits': next(iter(description.values()))['bits'],
                                      'group_size': next(iter(description.values()))['group_size'],
                                      'mode': 'affine', **description}
            config['torch_dtype'] = 'float32'
            save_config(config, root / 'config.json')
            self.tokenizer.save_pretrained(root)
            (root / 'rl_generation.json').write_text(json.dumps({
                'thinking_mode': self._thinking_mode, 'temperature': self._temperature}, indent=2))
            return description
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
            if hasattr(self, '_base_path'):
                from .provenance import model_identity
                (Path(output) / 'qat_base_identity.json').write_text(json.dumps({
                    'base_identity': model_identity(self._base_path)[1],
                    'thinking_mode': self._thinking_mode, 'temperature': self._temperature}, indent=2))
        return description

    def load_qat_export(self, output):
        """Replace fake quantization with the exported packed Metal operators.

        Projection QAT requires the same base; full QAT loads the standalone
        export, including all trained floating-point norms and biases.
        """
        root = Path(output)
        if self._qat_scope == 'full':
            if self._tuning != 'qat':
                raise ValueError('full QAT reload requires tuning qat')
            deployed, _, _ = load(root / 'qat_model', return_config=True)
            if deployed.args != self.lm.args:
                raise ValueError('full QAT export architecture does not match policy')
            self.lm = deployed
            self.lm.set_dtype(mx.float32)
            self.lm.freeze()
            self.eval()
            mx.eval(self.parameters())
            return {name: {'bits': layer.bits, 'group_size': layer.group_size}
                    for name, layer in self.lm.named_modules()
                    if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding))}
        description = json.loads((root / 'qat_quantization.json').read_text())
        weights = mx.load(str(root / 'qat_quantized.safetensors'))
        modules = dict(self.lm.named_modules())
        replacements = []
        expected = set()
        for name, settings in description.items():
            base = modules.get(name)
            if not isinstance(base, (QATLinear, nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f'export layer does not match QAT configuration: {name}')
            input_dims = base.weight.shape[1] * (32 // base.bits if isinstance(base, nn.QuantizedLinear) else 1)
            packed = nn.QuantizedLinear(input_dims, base.weight.shape[0],
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
