"""Per-instance block recomputation; parameter names and other policies stay intact."""
import mlx.core as mx


def install(model):
    for layer in model.layers:
        original = type(layer)
        if getattr(original, '_rl_checkpointed', False):
            continue
        call = original.__call__
        def checkpointed(block, *args, _call=call, **kwargs):
            # Cache mutation must never be replayed during the backward pass.
            if kwargs.get('cache') is not None or (len(args) > 2 and args[2] is not None):
                return _call(block, *args, **kwargs)
            def run(parameters, *inputs):
                block.update(parameters)
                return _call(block, *inputs, **kwargs)
            return mx.checkpoint(run)(block.trainable_parameters(), *args)
        layer.__class__ = type(f'Checkpointed{original.__name__}', (original,),
                               {'__call__': checkpointed, '_rl_checkpointed': True})
