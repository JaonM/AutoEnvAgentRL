"""Opt-in real Qwen3-0.6B regression for the optimized RL and CUDA export paths."""
import json
import os
from pathlib import Path
import pytest
MODEL=os.environ.get('RL_TEST_QWEN06_MODEL')
pytestmark=pytest.mark.skipif(not MODEL,reason='set RL_TEST_QWEN06_MODEL to Qwen3-0.6B MLX 4-bit')


@pytest.mark.parametrize('mode',['thinking','no-thinking'])
@pytest.mark.parametrize('decode',['single','batch'])
def test_qwen06_native_tools(tmp_path,monkeypatch,mode,decode):
    import mlx.core as mx
    from rl.model import Policy
    mx.random.seed(42)
    policy=Policy(MODEL,tuning='lora',thinking_mode=mode,critic=True,prefill_chunk_size=64)
    tools=[{'type':'function','function':{'name':'lookup',
        'description':'Look up the stock quantity for a product key.',
        'parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key']}}}]
    messages=[{'role':'system','content':'You are a tool assistant. Use lookup to obtain the quantity. Never guess. Enclose every tool call in <tool_call> and </tool_call> tags. Inside the tags write a JSON object with name and arguments. After receiving the tool result, answer with the quantity followed by " items".'},
              {'role':'user','content':'Use lookup to find the stock quantity for key A.'}]
    def sample(history):
        if decode=='single':
            result=policy.sample(history,tools=tools,max_tokens=2048,max_context=4096,greedy=True)
            assert result['behavior_scoring_max_logp_error']<=.001
            return result
        decoder=policy.batch_decoder(greedy=True)
        for i in range(2):decoder.add(i,history,2048,4096,tools=tools)
        results={}
        while decoder.requests:results.update(decoder.tick())
        assert results[0]['tokens']==results[1]['tokens']
        for result in results.values():
            scores,_=policy.cached_token_stats(mx.array([result['prompt']+result['tokens']]),len(result['prompt']))
            assert float(mx.max(mx.abs(scores-mx.array(result['old_logp']))))<=.001
        return results[0]
    call=sample(messages)
    assert call['generation_finish']=='eos' and 'protocol_error' not in call['action']
    calls=call['action'].get('tool_calls',[])
    assert len(calls)==1 and calls[0]['function']['name']=='lookup',call['text']
    assert json.loads(calls[0]['function']['arguments'])=={'key':'A'}
    assert bool(call['action'].get('reasoning_content'))==(mode=='thinking')
    calls[0]['id']='lookup-1'
    messages.extend([call['action'],{'role':'tool','tool_call_id':'lookup-1','name':'lookup','content':'{"quantity":7,"unit":"items"}'}])
    answer=sample(messages)
    assert answer['generation_finish']=='eos' and 'protocol_error' not in answer['action']
    assert '7 items' in answer['action']['content'],answer['text']
    for result in (call,answer):
        assert len(result['tokens'])==len(result['old_logp'])==len(result['old_values'])
    (tmp_path/'native_tools.json').write_text(json.dumps({'mode':mode,'decode':decode,
        'scope':'simple native tool task with explicit protocol instructions; greedy diagnostic',
        'call':call,'answer':answer},ensure_ascii=False,indent=2))



@pytest.mark.parametrize('mode',['thinking','no-thinking'])
@pytest.mark.parametrize('algorithm',['ppo','grpo'])
def test_qwen06_qlora_training_resume(tmp_path,monkeypatch,mode,algorithm):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path,monkeypatch,algorithm,tuning='lora',model_path=MODEL,thinking_mode=mode,
        deferred_failure=algorithm=='grpo',config_overrides={'layers':28,'lora_targets':'all-linear',
        'gradient_checkpointing':True,'logits_chunk_size':8,'prefill_chunk_size':64,'prefix_cache_tokens':2048,
        'checkpoint_interval_steps':3,'policy_publish_interval_steps':3,'profile_memory':True,
        'lora_merge_export':True,'lora_requantize_export':True})
    _check_cuda(tmp_path/algorithm,peft=True)


@pytest.mark.parametrize('algorithm',['ppo','grpo'])
def test_qwen06_full_qat_training_resume(tmp_path,monkeypatch,algorithm):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path,monkeypatch,algorithm,tuning='qat',qat_scope='full',model_path=MODEL,
        deferred_failure=algorithm=='grpo',config_overrides={'gradient_checkpointing':True,
        'packed_inference':True,'logits_chunk_size':8,'prefill_chunk_size':64,'prefix_cache_tokens':2048,
        'checkpoint_interval_steps':3,'policy_publish_interval_steps':3,'profile_memory':True})
    _check_cuda(tmp_path/algorithm,peft=False)


def _check_cuda(run, *, peft):
    import gc
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from mlx_lm import load
    from rl.convert_cuda import convert
    from rl.model import Policy
    from rl.model_options import policy_options
    torch.set_num_threads(4)
    gc.collect();mx.clear_cache()
    config=json.loads((run/'config.json').read_text())
    if peft:
        options=policy_options(config)
        policy=Policy(MODEL,tuning='lora',layers=config['layers'],rank=config['rank'],bits=config['bits'],**options)
        report=json.loads((run/'training_report.json').read_text())
        policy.restore(run/'snapshots'/f"policy-{report['updates']:06d}.safetensors",policy_only=True)
        model=policy.lm
        options={'adapter':run/'lora_adapter','format':'peft'}
        source=MODEL
    else:
        source=run/'qat_model';options={}
        model,_=load(source);model.set_dtype(mx.float32)
    ids=[1,2,3,4,5,6]
    expected=np.array(model(mx.array([ids])))
    np.savez(run/'mlx_logits_reference.npz',input_ids=np.array([ids]),logits=expected)
    destination=run/'cuda'
    convert(source,destination,dtype='float32',max_shard_size_mb=256,**options)
    del model
    if peft:del policy
    gc.collect();mx.clear_cache()
    actual=AutoModelForCausalLM.from_pretrained(destination/'base_model' if peft else destination,
                                               dtype=torch.float32,attn_implementation='eager').eval()
    if peft:
        from peft import PeftModel
        actual=PeftModel.from_pretrained(actual,destination/'adapter').eval()
    with torch.no_grad():logits=actual(torch.tensor([ids])).logits.numpy()
    error=float(np.max(np.abs(expected-logits)))
    # Record both a distribution metric and raw logits; constant shifts in logits
    # do not affect the policy, but large layout/scale errors must fail loudly.
    p=torch.log_softmax(torch.from_numpy(expected),dim=-1)
    q=torch.log_softmax(torch.from_numpy(logits),dim=-1)
    kl=float((p.exp()*(p-q)).sum(-1).mean())
    assert error < .15 and abs(kl)<1e-4,(error,kl)
    (run/'cuda_validation.json').write_text(json.dumps({'max_logit_error':error,'mean_kl':kl,
        'device':'cpu','CUDA_hardware_tested':False,'format':'peft' if peft else 'merged'},indent=2))
