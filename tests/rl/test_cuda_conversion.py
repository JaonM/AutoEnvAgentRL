"""Validate CUDA ecosystem files with independent PyTorch CPU loading."""
import json
import pytest
mx=pytest.importorskip('mlx.core',exc_type=ImportError)
torch=pytest.importorskip('torch',exc_type=ImportError)
from test_full_training import tiny_qwen3


@pytest.mark.parametrize('bits',[4,8])
def test_affine_unpack_matches_mlx(bits):
    from rl.convert_cuda import dequantize
    import numpy as np
    weight=mx.random.normal((5,128))
    packed,scales,biases=mx.quantize(weight,group_size=64,bits=bits)
    expected=mx.dequantize(packed,scales,biases,group_size=64,bits=bits)
    actual=dequantize(torch.from_numpy(np.array(packed)),torch.from_numpy(np.array(scales)),
                      torch.from_numpy(np.array(biases)),bits=bits,group_size=64)
    assert np.max(np.abs(actual.numpy()-np.array(expected)))<1e-6


@pytest.mark.parametrize('tiny_qwen3',[{'hidden_size':64}],indirect=True)
@pytest.mark.parametrize('mode',['full','qat','lora-merged','lora-peft','projection-qat'])
def test_transformers_and_peft_logits(tiny_qwen3,tmp_path,mode):
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx_lm.utils import save_model
    from rl.model import Policy
    from rl.lora_export import export
    from rl.convert_cuda import convert
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    base,_=load(tiny_qwen3)
    if mode!='full':
        nn.quantize(base,group_size=64,bits=4)
        save_model(tiny_qwen3,base)
        cfg=json.loads((tiny_qwen3/'config.json').read_text());cfg['quantization']={'bits':4,'group_size':64}
        (tiny_qwen3/'config.json').write_text(json.dumps(cfg))
    tuning='lora' if mode.startswith('lora') else 'full' if mode=='full' else 'qat'
    p=Policy(str(tiny_qwen3),tuning=tuning,qat_scope='full' if mode=='qat' else 'projections',
             lora_targets='all-linear',layers=2,rank=3,lora_scale=2.5)
    for _,m in p.lm.named_modules():
        if hasattr(m,'lora_b'):m.lora_b += .01
    ids=mx.array([[2,3,4,5]])
    expected=np.array(p.lm(ids))
    source=tiny_qwen3;options={}
    if mode=='full':
        source=p.export_full(tmp_path/'full')
    elif mode=='qat':
        p.export_qat(tmp_path);source=tmp_path/'qat_model'
    elif mode=='projection-qat':
        p.export_qat(tmp_path);options['qat_export']=tmp_path
    else:
        options['adapter']=export(p,tmp_path)['adapter']
        options['format']='peft' if mode=='lora-peft' else 'merged'
    destination=tmp_path/'cuda'
    convert(source,destination,dtype='float32',max_shard_size_mb=1,**options)
    restored=AutoModelForCausalLM.from_pretrained(destination/'base_model' if mode=='lora-peft' else destination,
                                                dtype=torch.float32,attn_implementation='eager').eval()
    if mode=='lora-peft':
        from peft import PeftModel
        restored=PeftModel.from_pretrained(restored,destination/'adapter').eval()
    with torch.no_grad():actual=restored(torch.tensor([[2,3,4,5]])).logits.numpy()
    # Metal packed GEMM and CPU dense GEMM need not be bit-identical.
    assert np.max(np.abs(expected-actual))<.02
    manifest=json.loads((destination/'conversion_manifest.json').read_text())
    assert manifest['optimizer_state_converted'] is False
    if mode.startswith('lora'):
        cfg=json.loads((tiny_qwen3/'config.json').read_text());cfg['test_mismatch']=True
        (tiny_qwen3/'config.json').write_text(json.dumps(cfg))
        with pytest.raises(ValueError,match='identity mismatch'):
            convert(tiny_qwen3,tmp_path/'wrong-base',dtype='float32',**options)
