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
import re
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


def is_failure(content):
    """A failed tool result in any format episodes carry: our envelope or "Error: ..." text."""
    content = content or ''
    if content.startswith('Error'):
        return True
    try:
        value = json.loads(content)
    except ValueError:
        return False
    return isinstance(value, dict) and value.get('ok') is False


def failed_turns(messages):
    """Indexes of assistant turns whose tool calls came back with ok=false."""
    out, current = set(), None
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            current = index
        elif message['role'] == 'tool' and current is not None:
            try:
                if is_failure(message.get('content')):
                    out.add(current)
            except ValueError:
                pass
    return out


def redundant_turns(messages, consent_names=('consent',)):
    """Indexes of assistant turns whose tool calls only repeat earlier identical calls, with no
    successful write (a call carrying a consent token) since then: a repeated lookup."""
    out, seen = set(), set()
    results = {m.get('tool_call_id'): m.get('content') for m in messages if m['role'] == 'tool'}
    for index, message in enumerate(messages):
        calls = message.get('tool_calls') or [] if message['role'] == 'assistant' else []
        keys = [(c['function']['name'], c['function'].get('arguments')) for c in calls]
        if keys and all(k in seen for k in keys):
            out.add(index)
        seen.update(keys)
        for call in calls:
            arguments = call['function'].get('arguments') or ''
            if any(f'"{name}"' in arguments for name in consent_names):
                try:
                    if not is_failure(results.get(call.get('id'))) and results.get(call.get('id')):
                        seen.clear()  # the world changed: later lookups may repeat legitimately
                except ValueError:
                    pass
    return out


NOTICE = re.compile(r'\s*(\(System notice:[^)]*\)|（系统提示：[^）]*）)')


def strip_notices(messages):
    """Platform notices (the idle warning) appended to user turns and tool results: τ²-bench never sends
    them, so the student must not learn to wait for one."""
    out = []
    for message in messages:
        content = message.get('content')
        out.append({**message, 'content': NOTICE.sub('', content)} if isinstance(content, str) else message)
    return out


def render_episode(tokenizer, episode, skip_failed=False):
    """[(token ids, loss mask)] with the whole episode as ONE sample (non-thinking only): the chat template
    over the full conversation, an empty think block inserted (as context) before every assistant turn that
    lacks one, so each trained turn starts exactly as it does when served; the loss covers the trained
    assistant turns. The prefix is computed once instead of once per turn (4-6x fewer tokens); the only
    difference from serving is that later turns also see the earlier empty blocks in their history."""
    keep = ('role', 'content', 'tool_calls', 'tool_call_id', 'name')
    messages = strip_notices([{k: v for k, v in m.items() if k in keep} for m in episode['messages']])
    assistants = [i for i, m in enumerate(episode['messages']) if m['role'] == 'assistant']
    trained = {i for i in assistants if not episode['messages'][i].get('context_only')}
    if skip_failed:
        trained -= failed_turns(messages) | redundant_turns(
            messages, {'consent', (episode.get('naming') or {}).get('consent', 'consent')})
    text = tokenizer.apply_chat_template(messages, tools=episode['tools'], tokenize=False, enable_thinking=False)
    head, empty = '<|im_start|>assistant\n', '<think>\n\n</think>\n\n'
    pieces, cursor = [], 0
    for index in assistants:
        start = text.index(head, cursor)
        body_start = start + len(head)
        end = text.index('<|im_end|>', body_start) + len('<|im_end|>')
        body = text[body_start:end]
        pieces.append((text[cursor:body_start] + ('' if body.startswith('<think>') else empty), 0))
        if body.startswith(empty):
            pieces[-1] = (pieces[-1][0] + empty, 0)
            body = body[len(empty):]
        pieces.append((body, 1 if index in trained else 0))
        cursor = end
    pieces.append((text[cursor:].rstrip('\n'), 0))
    ids, mask = [], []
    for piece, train in pieces:
        tokens = tokenizer.encode(piece, add_special_tokens=False) if piece else []
        ids += tokens
        mask += [train] * len(tokens)
    return [(ids, mask)] if any(mask) else []


def render_turns(tokenizer, episode, think, cap, limit=0, seed=0, skip_failed=False):
    """[(token ids, loss mask)] for the assistant turns of one episode, as served (see module doc).
    With skip_failed, a turn whose tool call failed stays in later turns' context but is not trained."""
    keep = ('role', 'content', 'tool_calls', 'tool_call_id', 'name')
    messages = strip_notices([{k: v for k, v in m.items() if k in keep} for m in episode['messages']])
    reasoning = {int(k): v for k, v in (episode.get('reasoning') or {}).items()}
    turns = [i for i, m in enumerate(episode['messages']) if m['role'] == 'assistant'
             and not episode['messages'][i].get('context_only')]
    if skip_failed:
        untrained = failed_turns(messages) | redundant_turns(
            messages, {'consent', (episode.get('naming') or {}).get('consent', 'consent')})
        turns = [i for i in turns if i not in untrained]
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
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='clip the global gradient norm (0: off); one 30k-token episode can otherwise throw the run')
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
    parser.add_argument('--render', choices=('turn', 'episode'), default='turn',
                        help='turn: one sample per assistant turn (exact serving format); episode: one sample per '
                             'episode with empty think blocks inserted (non-thinking only, much cheaper)')
    parser.add_argument('--rename-tools', type=float, default=0.0,
                        help='share of episodes whose business tools and parameters are renamed (tool_rename.py)')
    parser.add_argument('--vary-results', type=float, default=0.0,
                        help='share of episodes whose tool results switch to raw JSON / "Error: ..." text')
    parser.add_argument('--relayout', type=float, default=0.0,
                        help='share of episodes whose system prompt uses another layout (tags, markdown, policy first)')
    parser.add_argument('--greet', type=float, default=0.0,
                        help='share of episodes opened by an agent greeting before the user speaks (context only)')
    parser.add_argument('--example-values', type=float, default=0.0,
                        help="share of episodes whose identifier parameters' descriptions carry a made-up example "
                             "value (τ²-bench writes \"such as '#W0000000'\"; the student must not call with it)")
    parser.add_argument('--inject-errors', type=float, default=0.0,
                        help='share of episodes where one lookup is first called with a wrong identifier (the example '
                             'value or a mistyped one) and fails; the failed call is context, the fix is trained')
    parser.add_argument('--replay', nargs='*', default=[],
                        help='extra episode files mixed in as they are (e.g. the base model\'s own successes)')
    parser.add_argument('--no-error-turns', action='store_true',
                        help='do not train turns whose tool calls failed or only repeat earlier lookups '
                             '(they stay as context)')
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.utils import load, save_config, save_model

    mx.random.seed(args.seed)  # LoRA initialisation: without it the same data and seed train a different model
    model, tokenizer, config = load(args.model, return_config=True)
    if args.tuning == 'lora':
        from mlx_lm.tuner.utils import linear_to_lora_layers
        model.freeze()
        linear_to_lora_layers(model, len(model.layers), {'rank': args.lora_rank, 'scale': 2.0, 'dropout': 0.0,
                                                         'keys': None})
    grad_checkpoint(model.layers[0])

    # Successes and the near misses sft_select chose (safe, final state right, at most one answer field wrong);
    # filtering on success alone silently dropped every near miss.
    episodes = [r for path in [args.teacher, *args.replay] for r in map(json.loads, Path(path).read_text().splitlines())
                if (r.get('success') or r.get('near_miss')) and r.get('messages')]
    random.Random(args.seed).shuffle(episodes)
    if args.rename_tools:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tool_rename import augment
        rng = random.Random(args.seed + 1)
        episodes = [augment(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.rename_tools else e
                    for e in episodes]
    if args.relayout:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tool_rename import relayout
        rng = random.Random(args.seed + 4)
        episodes = [relayout(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.relayout else e
                    for e in episodes]
    if args.greet:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tool_rename import greet
        rng = random.Random(args.seed + 3)
        episodes = [greet(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.greet else e
                    for e in episodes]
    if args.example_values or args.inject_errors:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tool_rename import example_values, inject_errors
        rng = random.Random(args.seed + 5)
        episodes = [example_values(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.example_values
                    else e for e in episodes]
        rng = random.Random(args.seed + 6)
        episodes = [inject_errors(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.inject_errors
                    else e for e in episodes]
    if args.vary_results:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tool_rename import vary_results
        rng = random.Random(args.seed + 2)
        episodes = [vary_results(e, f"{args.seed}-{e['id']}-{e.get('seed')}") if rng.random() < args.vary_results else e
                    for e in episodes]
    if args.per_episode:
        pairs = [(e, render(tokenizer, e)) for e in episodes]
    elif args.render == 'episode':
        if args.think:
            raise SystemExit('--render episode is for non-thinking training only')
        pairs = [(e, s) for e in episodes for s in render_episode(tokenizer, e, args.no_error_turns)]
    else:
        pairs = [(e, s) for e in episodes for s in render_turns(tokenizer, e, args.think, args.reasoning_cap,
                                                               args.turns_per_episode, args.seed,
                                                               args.no_error_turns)]
    kept = [(e, s) for e, s in pairs if len(s[0]) <= args.max_tokens and sum(s[1])]
    if args.valid_ids:
        held = set(json.loads(Path(args.valid_ids).read_text()))
        valid, train = [s for e, s in kept if e['id'] in held], [s for e, s in kept if e['id'] not in held]
    else:
        # Whole episodes go to validation, so no turn of a validation episode is trained on.
        held = {id(e) for e in episodes[:max(1, round(len(episodes) * args.valid_share))]}
        valid, train = [s for e, s in kept if id(e) in held], [s for e, s in kept if id(e) not in held]
    print(f'{len(episodes)} episodes ({sum(bool(e.get("near_miss")) for e in episodes)} near misses), '
          f'{len(kept)} samples within {args.max_tokens} tokens: '
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
    norms = []
    for step in range(1, steps + 1):
        accumulated, losses = None, []
        for index in order[(step - 1) * args.grad_accumulation:step * args.grad_accumulation]:
            ids, mask = train[index]
            loss, grads = value_and_grad(model, mx.array(ids), mx.array(mask))
            accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
            losses.append(loss.item())
        averaged = tree_map(lambda g: g / args.grad_accumulation, accumulated)
        if args.grad_clip:
            averaged, grad_norm = optim.clip_grad_norm(averaged, args.grad_clip)
            norms.append(grad_norm.item())
        optimizer.update(model, averaged)
        mx.eval(model.parameters(), optimizer.state)
        if step % eval_every == 0 or step == steps:
            current = evaluate()
            row = {'step': step, 'train_loss': round(sum(losses) / len(losses), 4), 'valid_loss': round(current, 4),
                   'seconds': round(time.monotonic() - started)}
            if norms:
                row.update(grad_norm_max=round(max(norms), 3), grad_norm_mean=round(sum(norms) / len(norms), 3),
                           clipped=sum(n > args.grad_clip for n in norms))
                norms = []
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
