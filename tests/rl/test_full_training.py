"""Full-parameter optimizer, checkpoint and export tests on a tiny Qwen3 architecture."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
mx = pytest.importorskip('mlx.core', exc_type=ImportError)


@pytest.fixture
def tiny_qwen3(tmp_path, request):
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    root = tmp_path / 'tiny-qwen3'
    settings = getattr(request, 'param', {})
    hidden = settings.get('hidden_size', 32)
    args = ModelArgs(model_type='qwen3', hidden_size=hidden, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=hidden // 2, vocab_size=64, rms_norm_eps=1e-6,
                     max_position_embeddings=4096, rope_theta=1000000., tie_word_embeddings=settings.get('tied', True))
    mx.random.seed(19)
    model = Model(args)
    save_model(root, model)
    (root / 'config.json').write_text(json.dumps({**asdict(args), 'eos_token_id': 1, 'torch_dtype': 'float32'}))
    vocab = {'<unk>': 0, '<eos>': 1, **{f't{i}': i for i in range(2, 64)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', eos_token='<eos>')
    wrapped.chat_template = "{% for message in messages %}{{ message.content + ' ' }}{% endfor %}{% if add_generation_prompt %}t2 {% endif %}"
    wrapped.save_pretrained(root)
    return root


def test_full_unfreezes_embedding_attention_mlp_norm_and_exports(tiny_qwen3, tmp_path):
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from rl.model import Policy
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    policy = Policy(str(tiny_qwen3), tuning='full')
    reference = Policy(str(tiny_qwen3), tuning='full')
    reference.freeze()
    before = {k: mx.array(v) for k, v in tree_flatten(policy.lm.parameters())}
    assert before.keys() == dict(tree_flatten(policy.lm.trainable_parameters())).keys()
    config = Config(tuning='full', learning_rate=1e-3, target_kl=0.)
    trainer = ActorTrainer(policy, reference, optim.Adam(config.learning_rate), config, StageTimes())
    rows = []
    for prompt, response, advantage in [([2, 3], [4, 5], 1.), ([6, 7], [8, 9], -1.)]:
        logp, _ = policy.token_stats(mx.array([prompt + response]), len(prompt))
        rows.append({'prompt': prompt, 'tokens': response, 'old_logp': logp.tolist(),
                     'advantage': advantage, 'weight': .5})
    result = trainer.step(rows)
    assert result['gradient_norm'] > 0
    after = dict(tree_flatten(policy.lm.parameters()))
    changed = {name for name in before if bool(mx.any(before[name] != after[name]))}
    for suffix in ('embed_tokens.weight', 'self_attn.k_proj.weight', 'self_attn.o_proj.weight',
                   'mlp.gate_proj.weight', 'input_layernorm.weight', 'norm.weight'):
        assert any(name.endswith(suffix) for name in changed), suffix
    export = policy.export_full(tmp_path / 'full-export')
    reloaded = Policy(export, tuning='full')
    ids = mx.array([[2, 3, 4, 5]])
    assert reloaded.token_stats(ids, 2)[0].tolist() == policy.token_stats(ids, 2)[0].tolist()
    assert not any(k.startswith('critic') for k in mx.load(str(Path(export) / 'model.safetensors')))


@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_full_pipeline_resume_and_tensorboard(tiny_qwen3, tmp_path, monkeypatch, algorithm):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path, monkeypatch, algorithm, tuning='full', model_path=str(tiny_qwen3))
    exported = tmp_path / algorithm / 'full_model'
    assert (exported / 'model.safetensors').exists()
    assert (exported / 'tokenizer_config.json').exists()


def test_full_memory_preflight_checks_headers_and_rejects_quantization(tiny_qwen3):
    from rl.model_memory import full_training_memory, check_full_training_memory
    estimate = full_training_memory(tiny_qwen3, 1)
    assert estimate['persistent_lower_bound_bytes'] == estimate['parameters'] * 24
    with pytest.raises(MemoryError, match='before activations'):
        check_full_training_memory(tiny_qwen3, 1, available_bytes=1)
    config = json.loads((tiny_qwen3 / 'config.json').read_text())
    config['quantization'] = {'bits': 4, 'group_size': 64}
    (tiny_qwen3 / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='floating-point base weights'):
        full_training_memory(tiny_qwen3, 1)


def test_full_rejects_packed_model_before_unfreezing(tiny_qwen3, tmp_path):
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import save_model
    from rl.model import Policy
    model, _ = load(tiny_qwen3)
    nn.quantize(model, group_size=32, bits=4)
    quantized = tmp_path / 'quantized'
    save_model(quantized, model)
    config = json.loads((tiny_qwen3 / 'config.json').read_text())
    config['quantization'] = {'bits': 4, 'group_size': 32}
    (quantized / 'config.json').write_text(json.dumps(config))
    for source in tiny_qwen3.glob('*token*'):
        if source.is_file():
            (quantized / source.name).write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='floating-point base weights'):
        Policy(str(quantized), tuning='full')


@pytest.mark.parametrize('mode,expected', [('auto', None), ('thinking', True), ('no-thinking', False)])
def test_thinking_mode_is_forwarded_without_changing_messages(mode, expected):
    import mlx.nn as nn
    from types import SimpleNamespace
    from rl.model import Policy
    calls = []
    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        return [1, 2]
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy._thinking_mode = mode
    policy._tokenizer = SimpleNamespace(apply_chat_template=template, has_tool_calling=True, tool_parser=lambda: None)
    messages = [{'role': 'user', 'content': 'test'}]
    tools = [{'type': 'function', 'function': {'name': 'test'}}]
    assert policy.encode(messages, tools=tools) == [1, 2]
    assert calls[0][0] is messages and calls[0][1]['tools'] is tools
    assert calls[0][1].get('enable_thinking') is expected
    if mode == 'auto':
        assert 'enable_thinking' not in calls[0][1]


def test_partial_full_model_shards_are_not_treated_as_small_model(tiny_qwen3):
    from rl.model_memory import full_training_memory
    index = tiny_qwen3 / 'model.safetensors.index.json'
    data = json.loads(index.read_text())
    data['weight_map']['missing.weight'] = 'model-missing.safetensors'
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='incomplete'):
        full_training_memory(tiny_qwen3, 1)
