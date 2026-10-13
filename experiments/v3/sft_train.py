"""SFT cold start from successful teacher episodes.

Default (per turn, matching inference): one sample per assistant turn. The prompt is the chat
template over the conversation so far, exactly as the model sees it when serving (history
assistant turns carry no reasoning: Qwen3's template drops it and the serving loop never sends it
back), and the loss covers that turn only. --think trains the teacher's reasoning as the turn's
<think> block (turns whose reasoning is missing or longer than --reasoning-cap characters are
skipped); without it the turn follows the empty think block of enable_thinking=False.
--per-episode keeps the earlier rendering (one sample per episode, loss on every assistant turn).

Each episode (collect_teacher.py) is rendered once with the model's chat template and its tools;
the loss covers exactly the assistant turns (from after `<|im_start|>assistant\\n` through
`<|im_end|>`), never the system prompt, the user, or tool results. Full-parameter (default) or
LoRA fine-tuning with MLX; the best checkpoint by validation loss is saved as a loadable model
directory, which rl.train then starts from (--model).

    python experiments/v3/sft_train.py --teacher output/v3_sft/teacher.jsonl --model models/Qwen3-0.6B \
        --output output/v3_sft/model_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path


def render(tokenizer, episode):
    """(token ids, loss mask) for one episode; mask[i] is 1 when token i is assistant output."""
    keep = ('role', 'content', 'tool_calls', 'tool_call_id', 'name')
    messages = [{k: v for k, v in m.items() if k in keep} for m in episode['messages']]
    text = tokenizer.apply_chat_template(messages, tools=episode['tools'], tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    start, end = (tokenizer.convert_tokens_to_ids(t) for t in ('<|im_start|>', '<|im_end|>'))
    mask, inside = [0] * len(ids), False
    for i, token in enumerate(ids):
        if token == start:
            inside = tokenizer.decode(ids[i + 1:i + 3]) == 'assistant\n'
            skip = 3 if inside else 0
        if inside:
            if skip:
                skip -= 1
            else:
                mask[i] = 1
                if token == end:
                    inside = False
    return ids, mask


def render_turns(tokenizer, episode, think, cap, limit=0, seed=0):
    """[(token ids, loss mask)] for the assistant turns of one episode, as served (see module doc)."""
    keep = ('role', 'content', 'tool_calls', 'tool_call_id', 'name')
    messages = [{k: v for k, v in m.items() if k in keep} for m in episode['messages']]
    reasoning = {int(k): v for k, v in (episode.get('reasoning') or {}).items()}
    turns = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    if think:
        turns = [i for i in turns if (reasoning.get(i) or '').strip() and len(reasoning[i]) <= cap]
    if limit and len(turns) > limit:
        turns = sorted(random.Random(f"{seed}-{episode['id']}-{episode.get('seed')}").sample(turns, limit))
    out = []
    for i in turns:
        options = {'tools': episode['tools'], 'tokenize': False, 'enable_thinking': think}
        prompt = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **options)
        last = {**messages[i], **({'reasoning_content': reasoning[i].strip()} if think else {})}
        full = tokenizer.apply_chat_template(messages[:i] + [last], **options)
        if not full.startswith(prompt):
            raise ValueError(f"template prefix mismatch in {episode['id']} turn {i}")
        head = tokenizer.encode(prompt, add_special_tokens=False)
        tail = tokenizer.encode(full[len(prompt):].rstrip('\n'), add_special_tokens=False)
        out.append((head + tail, [0] * len(head) + [1] * len(tail)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', required=True, help='collect_teacher.py JSONL (only successful episodes are used)')
    parser.add_argument('--model', default='models/Qwen3-0.6B', help='floating-point base model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tuning', choices=('full', 'lora'), default='full')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--grad-accumulation', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=12288, help='longer episodes are skipped')
    parser.add_argument('--valid-share', type=float, default=0.1)
    parser.add_argument('--valid-ids', default=None,
                        help='JSON list of task ids whose episodes form the validation set (held-out domains); '
                             'otherwise --valid-share of the episodes at random')
    parser.add_argument('--evals-per-epoch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--think', action='store_true', help="train the teacher's reasoning as <think> blocks")
    parser.add_argument('--reasoning-cap', type=int, default=1500, help='longest reasoning kept (characters)')
    parser.add_argument('--turns-per-episode', type=int, default=0, help='sample at most this many turns (0: all)')
    parser.add_argument('--per-episode', action='store_true', help='one sample per episode (earlier rendering)')
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.utils import load, save_config, save_model

    model, tokenizer, config = load(args.model, return_config=True)
    if args.tuning == 'lora':
        from mlx_lm.tuner.utils import linear_to_lora_layers
        model.freeze()
        linear_to_lora_layers(model, len(model.layers), {'rank': args.lora_rank, 'scale': 2.0, 'dropout': 0.0,
                                                         'keys': None})
    grad_checkpoint(model.layers[0])

    episodes = [r for r in map(json.loads, Path(args.teacher).read_text().splitlines())
                if r.get('success') and r.get('messages')]
    random.Random(args.seed).shuffle(episodes)
    if args.per_episode:
        pairs = [(e, render(tokenizer, e)) for e in episodes]
    else:
        pairs = [(e, s) for e in episodes for s in render_turns(tokenizer, e, args.think, args.reasoning_cap,
                                                               args.turns_per_episode, args.seed)]
    kept = [(e, s) for e, s in pairs if len(s[0]) <= args.max_tokens and sum(s[1])]
    if args.valid_ids:
        held = set(json.loads(Path(args.valid_ids).read_text()))
        valid, train = [s for e, s in kept if e['id'] in held], [s for e, s in kept if e['id'] not in held]
    else:
        # Whole episodes go to validation, so no turn of a validation episode is trained on.
        held = {id(e) for e in episodes[:max(1, round(len(episodes) * args.valid_share))]}
        valid, train = [s for e, s in kept if id(e) in held], [s for e, s in kept if id(e) not in held]
    print(f'{len(episodes)} successful episodes, {len(kept)} samples within {args.max_tokens} tokens: '
          f'{len(train)} train / {len(valid)} valid; '
          f'{sum(sum(m) for _, m in train)} assistant tokens of {sum(len(i) for i, _ in train)}', flush=True)

    def loss_fn(model, ids, mask):
        logits = model(ids[None, :-1])
        targets, weights = ids[None, 1:], mask[None, 1:]
        ce = nn.losses.cross_entropy(logits, targets) * weights
        return ce.astype(mx.float32).sum() / weights.sum()

    steps = math.ceil(len(train) * args.epochs / args.grad_accumulation)
    schedule = optim.join_schedules([optim.linear_schedule(0.0, args.learning_rate, max(1, steps // 20)),
                                     optim.cosine_decay(args.learning_rate, steps)], [max(1, steps // 20)])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    def evaluate():
        total, tokens = 0.0, 0
        for ids, mask in valid:
            n = sum(mask[1:])
            total += loss_fn(model, mx.array(ids), mx.array(mask)).item() * n
            tokens += n
        return total / max(1, tokens)

    out = Path(args.output)
    history, best = [], float('inf')

    def write_model(model):
        target = out / 'model'
        save_model(target, model, donate_model=False)
        save_config(config, config_path=target / 'config.json')
        tokenizer.save_pretrained(target)
        for extra in Path(args.model).glob('generation_config.json'):
            shutil.copy(extra, target)
    order = []
    while len(order) < steps * args.grad_accumulation:
        batch = list(range(len(train)))
        random.Random(args.seed + len(order)).shuffle(batch)
        order += batch
    order = order[:steps * args.grad_accumulation]
    eval_every = max(1, steps // max(1, round(args.epochs * args.evals_per_epoch)))
    started = time.monotonic()
    baseline = evaluate()
    history.append({'step': 0, 'valid_loss': round(baseline, 4)})
    print(f'step 0 valid loss {baseline:.4f}', flush=True)
    for step in range(1, steps + 1):
        accumulated, losses = None, []
        for index in order[(step - 1) * args.grad_accumulation:step * args.grad_accumulation]:
            ids, mask = train[index]
            loss, grads = value_and_grad(model, mx.array(ids), mx.array(mask))
            accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
            losses.append(loss.item())
        optimizer.update(model, tree_map(lambda g: g / args.grad_accumulation, accumulated))
        mx.eval(model.parameters(), optimizer.state)
        if step % eval_every == 0 or step == steps:
            current = evaluate()
            row = {'step': step, 'train_loss': round(sum(losses) / len(losses), 4), 'valid_loss': round(current, 4),
                   'seconds': round(time.monotonic() - started)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if current < best:
                best = current
                if args.tuning == 'lora':  # keep the best adapters; fused into a model at the end
                    out.mkdir(parents=True, exist_ok=True)
                    mx.save_safetensors(str(out / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
                else:
                    write_model(model)
    if args.tuning == 'lora':
        model.load_weights(str(out / 'adapters.safetensors'), strict=False)
        model.update_modules(tree_unflatten([(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, 'fuse')]))
        write_model(model)
    (out / 'sft_history.json').write_text(json.dumps({'args': vars(args), 'train_episodes': len(train),
                                                      'valid_episodes': len(valid), 'history': history,
                                                      'best_valid_loss': best}, indent=1))
    print(f'best valid loss {best:.4f} (baseline {baseline:.4f}); model in {out / "model"}', flush=True)


if __name__ == '__main__':
    main()
