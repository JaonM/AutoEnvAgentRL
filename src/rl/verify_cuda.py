"""Load an exported HF/PEFT model on CPU or CUDA and optionally compare logits."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='conversion output directory')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--dtype',choices=['auto','float32','float16','bfloat16'],default='float32')
    parser.add_argument('--reference',help='NPZ with input_ids and logits arrays from MLX')
    parser.add_argument('--output')
    parser.add_argument('--max-logit-error',type=float,default=.15)
    parser.add_argument('--max-mean-kl',type=float,default=1e-4)
    args=parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device is unavailable; use --device cpu for format/numerical validation only')
    if args.device=='cpu':torch.set_num_threads(4)
    else:torch.cuda.reset_peak_memory_stats()
    root=Path(args.model)
    peft=(root/'adapter/adapter_config.json').exists()
    dtype='auto' if args.dtype=='auto' else getattr(torch,args.dtype)
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(root/'base_model' if peft else root,
        dtype=dtype,attn_implementation='eager',local_files_only=True)
    if peft:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,root/'adapter',local_files_only=True)
    model.to(args.device).eval()
    reference=np.load(args.reference) if args.reference else None
    ids=reference['input_ids'].tolist() if reference is not None else [[1,2,3,4,5,6]]
    with torch.no_grad():logits=model(torch.tensor(ids,device=args.device)).logits.float().cpu()
    report={'device':args.device,'torch_version':torch.__version__,'format':'peft' if peft else 'merged',
            'seconds':time.perf_counter()-started,'finite_logits':bool(torch.isfinite(logits).all()),
            'cuda_hardware_tested':args.device=='cuda','gpu_name':torch.cuda.get_device_name() if args.device=='cuda' else None,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None}
    if not report['finite_logits']:raise ValueError('non-finite model output')
    if reference is not None:
        expected=torch.from_numpy(reference['logits']).float()
        if expected.shape!=logits.shape:raise ValueError('reference/output shapes differ')
        report['max_logit_error']=float((expected-logits).abs().max())
        p=expected.log_softmax(-1);q=logits.log_softmax(-1)
        report['mean_kl']=float((p.exp()*(p-q)).sum(-1).mean())
        report['passed']=report['max_logit_error']<=args.max_logit_error and abs(report['mean_kl'])<=args.max_mean_kl
    else:report['passed']=True
    text=json.dumps(report,indent=2)
    if args.output:
        with Path(args.output).open('x') as stream:stream.write(text+'\n')
    print(text)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
