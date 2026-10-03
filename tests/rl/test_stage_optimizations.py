import json
from pathlib import Path
import pytest
mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from test_full_training import tiny_qwen3


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size':64}], indirect=True)
@pytest.mark.parametrize('tuning', ['full','lora','qat'])
def test_chunking_and_checkpointing_preserve_gradients(tiny_qwen3, tuning):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from rl.model import Policy
    options = dict(tuning=tuning, qat_scope='full' if tuning == 'qat' else 'projections',
                   lora_targets='all-linear', layers=2)
    mx.random.seed(2)
    baseline=Policy(str(tiny_qwen3), logits_chunk_size=128, **options)
    mx.random.seed(2)
    optimized=Policy(str(tiny_qwen3), logits_chunk_size=2, gradient_checkpointing=True, **options)
    ids=mx.array([[2,3,4,5,6,7],[3,4,5,6,7,8]])
    def loss(policy):
        logp, _, entropy=policy.batch_token_stats(ids,[2,2],[4,4],return_entropy=True)
        return logp.sum(), entropy
    (a,ea), ga=nn.value_and_grad(baseline,loss)(baseline)
    (b,eb), gb=nn.value_and_grad(optimized,loss)(optimized)
    assert float(mx.max(mx.abs(a-b))) < 1e-4
    assert float(mx.max(mx.abs(ea-eb))) < 1e-5
    ga,gb=dict(tree_flatten(ga)),dict(tree_flatten(gb))
    assert ga.keys()==gb.keys()
    assert max(float(mx.max(mx.abs(ga[k]-gb[k]))) for k in ga)<1e-4
    assert not getattr(type(baseline.lm.layers[0]),'_rl_checkpointed',False)


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size':64}], indirect=True)
def test_packed_reference_snapshot_and_prefix_cache(tiny_qwen3,tmp_path):
    from rl.model import Policy
    from rl.qat import QAT_MODULES
    actor=Policy(str(tiny_qwen3),qat_scope='full',packed_inference=True,critic=True)
    actor.lm.model.norm.weight += .1
    actor.critic.bias += .2
    path=actor.save_snapshot(tmp_path/'snapshots',1)
    reference=Policy(str(tiny_qwen3),qat_scope='full',inference_only=True,
                     cache_inference=True,prefix_cache_tokens=64,prefill_chunk_size=2)
    reference.restore(path,policy_only=True)
    assert not any(isinstance(m,QAT_MODULES) for _,m in reference.lm.named_modules())
    ids=mx.array([[2,3,4,5]])
    assert float(mx.max(mx.abs(actor.token_stats(ids,2)[0]-reference.token_stats(ids,2)[0])))<.01
    one=reference._prefill(ids)
    first=reference._step(ids[:,-1:],one)[0]
    two=reference._prefill(ids)
    second=reference._step(ids[:,-1:],two)[0]
    assert reference._prefill_stats['reused_tokens']>0
    assert float(mx.max(mx.abs(first-second)))<1e-6
    reference.restore(path,policy_only=True)
    assert not reference._prefix_cache


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size':64}], indirect=True)
def test_lora_exports_load_and_preserve_policy(tiny_qwen3,tmp_path):
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import save_model
    from rl.model import Policy
    from rl.lora_export import export
    base,_=load(tiny_qwen3)
    nn.quantize(base,group_size=64,bits=4)
    save_model(tiny_qwen3,base)
    cfg=json.loads((tiny_qwen3/'config.json').read_text());cfg['quantization']={'group_size':64,'bits':4}
    (tiny_qwen3/'config.json').write_text(json.dumps(cfg))
    p=Policy(str(tiny_qwen3),tuning='lora',layers=2,lora_targets='all-linear',rank=3,lora_scale=2.5)
    for _,module in p.lm.named_modules():
        if hasattr(module,'lora_b'):module.lora_b += .02
    ids=mx.array([[2,3,4]])
    expected=p.lm(ids);mx.eval(expected)
    digest=p.digest();rng=[k.tolist() for k in mx.random.state]
    paths=export(p,tmp_path/'merged',merge=True)
    assert p.digest()==digest and rng==[k.tolist() for k in mx.random.state]
    native,_=load(tiny_qwen3,adapter_path=paths['adapter']); native.set_dtype(mx.float32)
    merged,_=load(paths['merged']);merged.set_dtype(mx.float32)
    assert float(mx.max(mx.abs(native(ids)-expected)))<1e-5
    merged_modules = dict(merged.named_modules())
    for name,module in p.lm.named_modules():
        if hasattr(module,'lora_a'):
            base_weight = mx.dequantize(module.linear.weight,module.linear.scales,module.linear.biases,group_size=64,bits=4)
            correct = base_weight + (module.scale * module.lora_b.T) @ module.lora_a.T
            assert float(mx.max(mx.abs(merged_modules[name].weight-correct))) < 1e-6
    assert float(mx.max(mx.abs(merged(ids)-expected)))<.01
    quantized=export(p,tmp_path/'quantized',merge=True,requantize=True)
    deployed,_=load(quantized['merged'])
    assert mx.all(mx.isfinite(deployed(ids))).item()
    assert len(json.loads((Path(paths['adapter'])/'adapter_config.json').read_text())['lora_parameters']['keys'])==7


def test_future_policy_versions_are_quarantined(tmp_path):
    from rl.recovery_versions import quarantine
    for name,version in [('result-00000001-attempt-0.json',1),('result-00000002-attempt-0.json',2)]:
        p=tmp_path/'task_queue'/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps({'policy_version':version}))
    for version in [1,2]:
        p=tmp_path/'snapshots'/f'policy-{version:06d}.safetensors';p.parent.mkdir(exist_ok=True);p.write_bytes(b'x')
    quarantine(tmp_path,1,tmp_path/'recovery')
    assert (tmp_path/'snapshots/policy-000001.safetensors').exists()
    assert not (tmp_path/'snapshots/policy-000002.safetensors').exists()
    assert (tmp_path/'recovery/task_queue/result-00000002-attempt-0.json').exists()


def test_long_prefill_does_not_block_ready_decode(tiny_qwen3):
    from rl.model import Policy
    policy = Policy(str(tiny_qwen3), tuning='lora', prefill_chunk_size=2)
    decoder = policy.batch_decoder()
    # Fix tokens to isolate scheduling from random EOS on this tiny model.
    decoder.sampler = lambda logp: mx.full((logp.shape[0],), 2, dtype=mx.int32)
    decoder.add('short', [{'role': 'user', 'content': 't3'}], 3, 64)
    assert not decoder.tick()
    assert len(decoder.requests['short']['tokens']) == 1
    decoder.add('long', [{'role': 'user', 'content': ' '.join(['t4'] * 12)}], 2, 64)
    assert not decoder.tick()
    assert len(decoder.requests['short']['tokens']) == 2
    assert not decoder.requests['long']['tokens']
    results = decoder.tick()
    assert 'short' in results and 'prefill' in decoder.requests['long']
    while decoder.requests:
        results.update(decoder.tick())
    assert results['long']['tokens'] == [2, 2]
    for result in results.values():
        logp, _ = policy.cached_token_stats(mx.array([result['prompt'] + result['tokens']]), len(result['prompt']))
        assert float(mx.max(mx.abs(logp - mx.array(result['old_logp'])))) < 1e-4


@pytest.mark.parametrize('tiny_qwen3', [{'hidden_size':64}], indirect=True)
def test_deferred_checkpoint_packed_rollout_resume(tiny_qwen3,tmp_path,monkeypatch):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path,monkeypatch,'grpo',tuning='qat',qat_scope='full',model_path=str(tiny_qwen3),
        deferred_failure=True,config_overrides={'checkpoint_interval_steps':3,
        'policy_publish_interval_steps':3,'packed_inference':True,'gradient_checkpointing':True,
        'logits_chunk_size':2,'prefill_chunk_size':2,'prefix_cache_tokens':64})
