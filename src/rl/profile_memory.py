"""Measure a real forward/backward/optimizer step before choosing RL budgets."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tuning',choices=['lora','qat','full'],default='lora')
    parser.add_argument('--qat-scope',choices=['projections','full'],default='projections')
    parser.add_argument('--prompt-tokens',type=int,default=128)
    parser.add_argument('--response-tokens',type=int,default=32)
    parser.add_argument('--sequences',type=int,default=1)
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--lora-targets',default='self_attn.q_proj,self_attn.v_proj')
    parser.add_argument('--gradient-checkpointing',action='store_true')
    parser.add_argument('--packed-inference',action='store_true')
    parser.add_argument('--logits-chunk-size',type=int,default=128)
    args=parser.parse_args()
    if min(args.prompt_tokens,args.response_tokens,args.sequences)<1:parser.error('budgets must be positive')
    destination=Path(args.output)
    if destination.exists():raise FileExistsError('use a new output path')
    if args.tuning=='full' or (args.tuning=='qat' and args.qat_scope=='full'):
        from .model_memory import check_full_training_memory
        check_full_training_memory(args.model,0,qat=args.tuning=='qat',packed_inference=args.packed_inference)
    import mlx.core as mx
    import mlx.optimizers as optim
    from .model import Policy
    from .train import Config
    from .actor import ActorTrainer
    from .timing import StageTimes
    times=StageTimes(mlx=mx)
    mx.random.seed(42)
    with times.measure('actor_load'):
        policy=Policy(args.model,tuning=args.tuning,qat_scope=args.qat_scope,layers=args.layers,
                      lora_targets=args.lora_targets,gradient_checkpointing=args.gradient_checkpointing,
                      logits_chunk_size=args.logits_chunk_size)
    with times.measure('reference_load'):
        reference=Policy(args.model,tuning=args.tuning,qat_scope=args.qat_scope,layers=args.layers,
                         lora_targets=args.lora_targets,inference_only=args.packed_inference and args.tuning=='qat')
        reference.freeze()
    config=Config(tuning=args.tuning,qat_scope=args.qat_scope,target_kl=0.,micro_batch_size=args.sequences,
                  max_tokens_per_micro_batch=args.sequences*(args.prompt_tokens+args.response_tokens))
    optimizer=optim.Adam(1e-5)
    rows=[]
    for i in range(args.sequences):
        ids=[2+i]*args.prompt_tokens; response=[3+i]*args.response_tokens
        with times.measure('prefill'):
            cache=policy._prefill(mx.array([ids]))
            mx.eval([c.state for c in cache])
        logp=[]
        with times.measure('decode'):
            current=ids[-1]
            for token in response:
                scores,_=policy._step(mx.array([[current]]),cache)
                logp.append(float(scores[token]))
                current=token
        rows.append({'prompt':ids,'tokens':response,'old_logp':logp,'advantage':1.,'weight':1/args.sequences})
    trainer=ActorTrainer(policy,reference,optimizer,config,times)
    metrics=trainer.step(rows)
    import psutil
    report={'config':vars(args),'scope':'synthetic-token memory probe; no task-learning claim; no rollout worker process',
            'numeric_path':{'max_logp_error':trainer.max_numerical_error,
                            'fallbacks':trainer.numeric_fallbacks,
                            'cached_gradient_microbatches':trainer.cached_gradient_microbatches,
                            'batched_gradient_microbatches':trainer.batched_gradient_microbatches},
            'timings':times.state(),'metrics':metrics,'available_bytes_after':psutil.virtual_memory().available,
            'swap_used_bytes':psutil.swap_memory().used}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(destination)


if __name__=='__main__':main()
