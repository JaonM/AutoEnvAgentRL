"""All-parameter QAT on real tiny Qwen3 modules, including packed deployment."""
import json
from pathlib import Path

import pytest
mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from test_full_training import tiny_qwen3


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size': 64, 'tied': True},
                                        {'hidden_size': 64, 'tied': False}], indirect=True)
@pytest.mark.parametrize('packed_base', [False, True])
def test_full_qat_updates_and_standalone_export(tiny_qwen3, tmp_path, packed_base):
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import save_model
    from rl.model import Policy
    from rl.qat import QATLinear, QATEmbedding
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    from rl.model_memory import full_training_memory
    expected_count = full_training_memory(tiny_qwen3, 1)['parameters']
    config = json.loads((tiny_qwen3 / 'config.json').read_text())
    bits = 4 if config['tie_word_embeddings'] else 8
    if packed_base:
        base, _ = load(tiny_qwen3)
        nn.quantize(base, group_size=64, bits=bits)
        save_model(tiny_qwen3, base)
        config['quantization'] = {'group_size': 64, 'bits': bits}
        (tiny_qwen3 / 'config.json').write_text(json.dumps(config))
    assert full_training_memory(tiny_qwen3, 1, qat=True)['parameters'] == expected_count
    policy = Policy(str(tiny_qwen3), qat_scope='full', bits=bits)
    reference = Policy(str(tiny_qwen3), qat_scope='full', bits=bits)
    reference.freeze()
    modules = dict(policy.lm.named_modules())
    assert isinstance(modules['model.embed_tokens'], QATEmbedding)
    for index in range(2):
        for projection in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                           'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            assert isinstance(modules[f'model.layers.{index}.{projection}'], QATLinear)
    if not config['tie_word_embeddings']:
        assert isinstance(modules['lm_head'], QATLinear)
    before = {k: mx.array(v) for k, v in tree_flatten(policy.lm.parameters())}
    assert before.keys() == dict(tree_flatten(policy.lm.trainable_parameters())).keys()
    trainer = ActorTrainer(policy, reference, optim.Adam(1e-3),
                           Config(qat_scope='full', target_kl=0.), StageTimes())
    rows = []
    for prompt, tokens, advantage in [([2, 3], [4, 5], 1.), ([6, 7], [8, 9], -1.)]:
        logp, _ = policy.token_stats(mx.array([prompt + tokens]), len(prompt))
        rows.append({'prompt': prompt, 'tokens': tokens, 'old_logp': logp.tolist(),
                     'advantage': advantage, 'weight': .5})
    result = trainer.step(rows)
    assert result['gradient_norm'] > 0
    after = dict(tree_flatten(policy.lm.parameters()))
    changed = {k for k in before if bool(mx.any(before[k] != after[k]))}
    for suffix in ['embed_tokens.weight', 'self_attn.k_proj.weight', 'self_attn.o_proj.weight',
                   'mlp.gate_proj.weight', 'input_layernorm.weight', 'norm.weight']:
        assert any(k.endswith(suffix) for k in changed), suffix
    if not config['tie_word_embeddings']:
        assert 'lm_head.weight' in changed
    rng = [x.tolist() for x in mx.random.state]
    digest = policy.digest()
    effective = policy.digest(effective=True)
    description = policy.export_qat(tmp_path)
    assert len(description) == 15 + (not config['tie_word_embeddings'])
    assert policy.digest() == digest and policy.digest(effective=True) == effective
    assert rng == [x.tolist() for x in mx.random.state]
    # FP32 norms are also part of the effective deployed policy.
    original_norm = policy.lm.model.norm.weight
    policy.lm.model.norm.weight = original_norm + .1
    assert policy.digest(effective=True) != effective
    policy.lm.model.norm.weight = original_norm
    ids = mx.array([[2, 3, 4, 5]])
    expected = policy.lm(ids)
    deployed, tokenizer = load(tmp_path / 'qat_model')
    assert float(mx.max(mx.abs(expected - deployed(ids)))) < .01
    assert dict(tree_flatten(deployed.parameters()))['model.norm.weight'].tolist() == after['model.norm.weight'].tolist()
    exported = dict(tree_flatten(deployed.parameters()))
    assert exported['model.embed_tokens.weight'].dtype == mx.uint32
    assert not any(name.startswith('critic') for name in exported)
    reference.load_qat_export(tmp_path)
    assert float(mx.max(mx.abs(expected - reference.lm(ids)))) < .01


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size': 64}], indirect=True)
@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_full_qat_training_resume(tiny_qwen3, tmp_path, monkeypatch, algorithm):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path, monkeypatch, algorithm, tuning='qat', model_path=str(tiny_qwen3), qat_scope='full')
    export = tmp_path / algorithm / 'qat_model'
    assert (export / 'model.safetensors').exists()
    assert (export / 'tokenizer_config.json').exists()
    report = json.loads((tmp_path / algorithm / 'training_report.json').read_text())
    assert report['qat_scope'] == 'full' and report['effective_qat_weights_changed']


def test_full_qat_config_rejects_non_qat_tuning():
    from rl.train import Config
    for tuning in ['full', 'lora']:
        with pytest.raises(ValueError, match='requires tuning qat'):
            Config(sandbox='x', output='y', tuning=tuning, qat_scope='full').validate()
