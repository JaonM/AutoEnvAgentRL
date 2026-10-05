"""One policy-option mapping for actor, reference, evaluation and rollout."""
from dataclasses import asdict


def policy_options(config, *, inference=False):
    values = config if isinstance(config, dict) else asdict(config)
    names = ('thinking_mode', 'qat_scope', 'lora_targets', 'lora_scale', 'lora_dropout',
             'gradient_checkpointing', 'logits_chunk_size', 'prefill_chunk_size',
             'prefix_cache_tokens', 'packed_inference')
    result = {name: values[name] for name in names if name in values}
    if inference:
        result['gradient_checkpointing'] = False
        result['cache_inference'] = True
        result['inference_only'] = bool(values.get('packed_inference') and values.get('tuning') == 'qat')
    return result
