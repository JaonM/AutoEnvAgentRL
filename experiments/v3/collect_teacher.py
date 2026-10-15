"""Teacher demonstrations for SFT cold start: a strong model (LLM_*, deepseek) plays V3 episodes
through exactly the interface the learner trains on (rl V3Episode: free-dialogue user, platform
tools renamed per episode), and every episode is kept with its grade. sft_data.py turns the
successful ones into SFT examples.

Seeds pick the world (original, approval flips, perturbed) and the tool names, so one task gives
demonstrations of both approval outcomes under different names. Eval-split tasks are skipped:
their domains stay unseen until evaluation. Rows are appended, so an interrupted run resumes.

    python experiments/v3/collect_teacher.py --tasks output/v3_sft/tasks.json --episodes 12 \
        --parallel 16 --output output/v3_sft/teacher.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path


# Teacher-only addendum to the system prompt: the saved conversation (what the student trains on)
# keeps the platform's own system prompt, so the student learns the tone, not this instruction.
STYLE = ('\n\n（示范要求，仅对你生效：对用户说话时像一位耐心的真人客服，用日常口语，金额换算成“元”来说，'
         '不要提字段名、状态码、内部编号格式、以分为单位的金额或规则原文；先说结论，再用一两句话说明原因。'
         '工具调用的参数和最终提交仍严格按工具定义填写。按需调用工具：每次只查当前这一步需要的信息，'
         '结果用得上才查，不要一拿到编号就把相关工具全查一遍；编号只用用户说过或工具返回过的，'
         '查不到就向用户核实，不要换着编号猜。）')
STYLE_EN = ('\n\n(Demonstration requirement, for you only: talk to the user like a patient human customer service '
            'agent, in everyday English; say amounts in dollars, never in cents; never mention field names, status codes, '
            'internal id formats or rule text; give the conclusion first, then one or two sentences of reasons. Tool '
            'arguments still follow the tool definitions exactly. Call tools on demand: look up only what the current '
            'step needs and will use, never every related tool as soon as you hold an id; use only ids the user said '
            'or a tool returned, and when a lookup finds nothing, check with the user instead of trying other ids.)')
INTERNAL = re.compile(r'[A-Za-z]+_[A-Za-z_]+|\d+\s*分(?!钟)|\b\d+\s*cents?\b')


def leaks(messages):
    """Assistant speech to the user that still exposes field names or amounts in cents."""
    return sum(bool(INTERNAL.search(m.get('content') or '')) for m in messages
               if m['role'] == 'assistant' and (m.get('content') or '').strip())


def teacher_view(messages, style, english=False):
    if not style:
        return messages
    return [{**messages[0], 'content': messages[0]['content'] + (STYLE_EN if english else STYLE)}, *messages[1:]]


def play(task, seed, client, max_turns, style=True, temperature=0.7):
    from rl.environments.v3 import V3Episode
    env = V3Episode(task['sandbox'])
    started = time.monotonic()
    try:
        env.reset(seed)
        tools, reasoning = env.tools, {}
        for _ in range(max_turns):
            reply = client.chat(teacher_view(env.messages, style, env.language == 'en'), tools, temperature=temperature)
            message = {'role': 'assistant', 'content': reply.get('content') or ''}
            if reply.get('tool_calls'):
                message['tool_calls'] = [
                    {'id': call.get('id') or f'call_{seed}_{index}', 'type': 'function',
                     'function': {'name': call['function']['name'],
                                  'arguments': call['function'].get('arguments') or '{}'}}
                    for index, call in enumerate(reply['tool_calls'])]
            if reply.get('reasoning_content'):
                # Kept beside the conversation (index of that assistant message), never sent back to the
                # provider; think-mode SFT can attach it as the turn's reasoning.
                reasoning[len(env.messages)] = reply['reasoning_content']
            _, done = env.step(message)
            if done:
                break
        result = env.finish()
        return {'id': task['id'], 'seed': seed, 'world': result['world'], 'naming': env.naming,
                'reward': result['final_reward'], 'success': bool(result['score'].get('success')),
                'terminated': result.get('terminated'), 'consent': result.get('consent'),
                'state_ok': result['score'].get('state_ok'), 'safety': result['score'].get('safety'),
                'field_scores': result['score'].get('field_scores'),
                'reason': result['score'].get('reason'), 'reward_type': result['score'].get('reward_type'),
                'tools': tools, 'messages': env.messages, 'reasoning': reasoning, 'style': style,
                'speech_leaks': leaks(env.messages),
                'seconds': round(time.monotonic() - started, 1)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--episodes', type=int, default=12, help='episodes (seeds) per task')
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--max-turns', type=int, default=24)
    parser.add_argument('--temperature', type=float, default=0.7, help='sampling temperature of the played model')
    parser.add_argument('--prefix', default='LLM', help='teacher model: <PREFIX>_API_KEY/_BASE_URL/_MODEL')
    parser.add_argument('--no-style', action='store_true', help='do not ask the teacher for customer-service tone')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('ENVFACTORY_V3_USER_MODE', 'natural')
    os.environ.setdefault('ENVFACTORY_V3_TOOL_NAMES', 'random')
    from env_factory_v3.core.llm import ChatClient, load_env_file
    load_env_file()
    client = ChatClient(args.prefix)

    tasks = [t for t in json.loads(Path(args.tasks).read_text())['tasks'] if t.get('split', 'train') == 'train']
    out = Path(args.output)
    done = set()
    if out.exists():
        done = {(r['id'], r['seed']) for r in map(json.loads, out.read_text().splitlines())}
    jobs = [(t, s) for t in tasks for s in range(args.episodes) if (t['id'], s) not in done]
    print(f'{len(tasks)} tasks, {len(jobs)} episodes to play', flush=True)
    lock, counts = threading.Lock(), {'done': 0, 'success': 0, 'error': 0}

    def one(job):
        task, seed = job
        from env_factory_v3.production import offpeak
        offpeak.wait(lambda text: print(text, flush=True))  # DeepSeek's off-peak hours (ENVFACTORY_OFFPEAK=0: off)
        try:
            row = play(task, seed, client, args.max_turns, style=not args.no_style, temperature=args.temperature)
        except Exception as exc:  # one failed episode is recorded and skipped
            row = {'id': task['id'], 'seed': seed, 'success': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}
        with lock:
            with out.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts['done'] += 1
            counts['success'] += row.get('success', False)
            counts['error'] += 'error' in row
            if counts['done'] % 20 == 0 or counts['done'] == len(jobs):
                print(f"{counts['done']}/{len(jobs)} played, {counts['success']} successful, "
                      f"{counts['error']} errors", flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, jobs))


if __name__ == '__main__':
    main()
