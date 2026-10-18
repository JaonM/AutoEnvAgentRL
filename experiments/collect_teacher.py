"""Teacher demonstrations for SFT cold start: a strong model (LLM_*, deepseek) plays episodes
through exactly the interface the learner trains on (rl SandboxEpisode: free-dialogue user, platform
tools renamed per episode), and every episode is kept with its grade. sft_data.py turns the
successful ones into SFT examples.

Seeds pick the world (original, approval flips, perturbed) and the tool names, so one task gives
demonstrations of both approval outcomes under different names. Eval-split tasks are skipped:
their domains stay unseen until evaluation. Rows are appended, so an interrupted run resumes.

    python experiments/collect_teacher.py --tasks output/v3_sft/tasks.json --episodes 12 \
        --parallel 16 --output output/v3_sft/teacher.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import os
import re
import threading
import time
from pathlib import Path


# Teacher-only addendum to the system prompt: the saved conversation (what the student trains on)
# keeps the platform's own system prompt, so the student learns the tone, not this instruction.
STYLE = ('\n\n（示范要求，仅对你生效：对用户说话时像一位耐心的真人客服，用日常口语，保留政策规定的币种、金额单位和精度，'
         '仅在换算关系明确且政策允许时换算，不因语言假设币种。不要提字段名、状态码、内部编号格式或规则原文；先说结论，再用一两句话说明原因。'
         '工具调用的参数和最终提交仍严格按工具定义填写。按需调用工具：每次只查当前这一步需要的信息，'
         '结果用得上才查，不要一拿到编号就把相关工具全查一遍；编号只用用户说过或工具返回过的，'
         '查不到就向用户核实，不要换着编号猜。回复简短：每次不超过三句，不用标题、列表或“处理结果如下”式的汇总，'
         '只说用户问到的和决定结果的原因；用户确认后直接执行修改，不要再复述一遍方案；需要查询就直接调用工具，'
         '不要先说“我来查一下”。用户有好几件事时，每办完一件就回复那一件的结果，不要攒到最后一起说；不用粗体、标题等格式。）')
STYLE_EN = ('\n\n(Demonstration requirement, for you only: talk to the user like a patient human customer service '
            'agent, in everyday English; preserve the policy currency, units and precision; convert only when the '
            'conversion is defined and permitted, never infer currency from language; never mention field names, status codes, '
            'internal id formats or rule text; give the conclusion first, then one or two sentences of reasons. Tool '
            'arguments still follow the tool definitions exactly. Call tools on demand: look up only what the current '
            'step needs and will use, never every related tool as soon as you hold an id; use only ids the user said '
            'or a tool returned, and when a lookup finds nothing, check with the user instead of trying other ids. Keep '
            'replies short: at most three sentences, no headings, bullet lists or "here is the outcome" summaries; say '
            'what the user asked and what decided the result; once the user confirms, make the change right away '
            'instead of restating the plan; when you need to look something up, call the tool directly instead of '
            'announcing "let me check". When the user has several requests, give each result as soon as that one is '
            'settled instead of saving them all for one final message; no bold, headings or other formatting.)')
INTERNAL = re.compile(r'[A-Za-z]+_[A-Za-z_]+')


def leaks(messages):
    """Assistant speech that exposes identifier-like field names; currency units are not leaks."""
    return sum(bool(INTERNAL.search(m.get('content') or '')) for m in messages
               if m['role'] == 'assistant' and (m.get('content') or '').strip())


def teacher_view(messages, style, english=False):
    if not style:
        return messages
    return [{**messages[0], 'content': messages[0]['content'] + (STYLE_EN if english else STYLE)}, *messages[1:]]


def play(task, seed, client, max_turns, style=True, temperature=0.7, protocol_attempts=1):
    from rl.environments.sandbox import SandboxEpisode
    from env_factory.certify.policy_gate import provider_messages
    from env_factory.certify import trajectory
    from env_factory.production.teacher import ConstrainedTeacher, TeacherProtocolError
    env = SandboxEpisode(task['sandbox'])
    teacher = None
    started = time.monotonic()
    try:
        env.reset(seed)
        rules = trajectory.constraints(env.service.request)
        if protocol_attempts > 1:
            teacher = ConstrainedTeacher(client, rules, attempts=protocol_attempts)
        tools, reasoning = env.tools, {}
        for _ in range(max_turns):
            reply = (teacher or client).chat(provider_messages(trajectory.provider_view(
                teacher_view(env.messages, style, env.language == 'en'), rules), reasoning),
                                tools, temperature=temperature)
            message = {'role': 'assistant', 'content': reply.get('content') or ''}
            if reply.get('tool_calls'):
                message['tool_calls'] = [
                    {'id': call.get('id') or f'call_{seed}_{index}', 'type': 'function',
                     'function': {'name': call['function']['name'],
                                  'arguments': call['function'].get('arguments') or '{}'}}
                    for index, call in enumerate(reply['tool_calls'])]
            if 'reasoning_content' in reply:
                # Provider continuation needs this sidecar; the student's saved dialogue stays unchanged.
                reasoning[len(env.messages)] = reply['reasoning_content']
            _, done = env.step(message)
            if done:
                break
        result = env.finish()
        policy = trajectory.audit(env.messages, rules)
        environment_success = bool(result['score'].get('success'))
        return {'id': task['id'], 'seed': seed, 'world': result['world'], 'naming': env.naming,
                'reward': result['final_reward'], 'success': environment_success and policy['passed'],
                'environment_success': environment_success, 'trajectory_policy': policy,
                'generation': teacher.report() if teacher else {'method': 'unassisted'},
                'terminated': result.get('terminated'), 'consent': result.get('consent'),
                'state_ok': result['score'].get('state_ok'), 'safety': result['score'].get('safety'),
                'field_scores': result['score'].get('field_scores'),
                'reason': result['score'].get('reason'), 'reward_type': result['score'].get('reward_type'),
                'language': env.language, 'tools': tools, 'messages': env.messages, 'reasoning': reasoning, 'style': style,
                'speech_leaks': leaks(env.messages),
                'seconds': round(time.monotonic() - started, 1)}
    except TeacherProtocolError as error:
        return {'id': task['id'], 'seed': seed, 'success': False, 'error': str(error),
                'failure_class': 'teacher_protocol_exhausted', 'generation': teacher.report(),
                'messages': env.messages, 'tools': env.tools,
                'seconds': round(time.monotonic() - started, 1)}
    finally:
        env.close()


sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_dedup import add_arguments, write_candidates
WINDOW_PRINT = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', required=True)
    add_arguments(parser)
    parser.add_argument('--episodes', type=int, default=12, help='episodes (seeds) per task')
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--max-turns', type=int, default=24)
    parser.add_argument('--temperature', type=float, default=0.7, help='sampling temperature of the played model')
    parser.add_argument('--prefix', default='LLM', help='teacher model: <PREFIX>_API_KEY/_BASE_URL/_MODEL')
    parser.add_argument('--no-style', action='store_true', help='do not ask the teacher for customer-service tone')
    parser.add_argument('--protocol-attempts', type=int, default=1,
                        help='teacher drafts per turn before execution (1 keeps unassisted sampling; >1 records rejection sampling)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', type=int, default=2,
                        help='seeds per task played first; the rest only when the pilot looks sound (teacher_review)')
    args = parser.parse_args()
    if args.protocol_attempts < 1:
        parser.error('--protocol-attempts must be positive')
    os.environ.setdefault('ENVFACTORY_USER_MODE', 'natural')
    os.environ.setdefault('ENVFACTORY_TOOL_NAMES', 'random')
    from env_factory.core.llm import ChatClient, load_env_file
    load_env_file()
    client = ChatClient(args.prefix)

    tasks = [t for t in json.loads(Path(args.tasks).read_text())['tasks'] if t.get('split', 'train') == 'train']
    out = Path(args.output)
    done = set()
    if out.exists():
        done = {(r['id'], r['seed']) for r in map(json.loads, out.read_text().splitlines())}
    previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    from teacher_review import Breaker, flags, summary, task_verdict
    # Pilot first: PILOT seeds of every task, then the rest only for tasks whose pilot looks sound (task_verdict).
    pilot = [(t, s) for t in tasks for s in range(min(args.pilot, args.episodes)) if (t['id'], s) not in done]
    print(f'{len(tasks)} tasks, {len(pilot)} pilot episodes to play', flush=True)
    lock, counts = threading.Lock(), {'done': 0, 'success': 0, 'error': 0}
    breaker, alert = Breaker(), Path(str(out) + '.alert.json')
    state = {'tripped': None, 'played': list(previous)}

    def one(job):
        task, seed = job
        if state['tripped']:
            return
        from env_factory.production import offpeak
        offpeak.wait(lambda text: print(text, flush=True))  # DeepSeek's off-peak hours (ENVFACTORY_OFFPEAK=0: off)
        try:
            row = play(task, seed, client, args.max_turns, style=not args.no_style, temperature=args.temperature,
                       protocol_attempts=args.protocol_attempts)
        except Exception as exc:  # one failed episode is recorded and skipped
            row = {'id': task['id'], 'seed': seed, 'success': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}
        row['review'] = flags(row)
        with lock:
            with out.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            state['played'].append(row)
            counts['done'] += 1
            counts['success'] += row.get('success', False)
            counts['error'] += 'error' in row
            tripped = breaker.add(row['review'])
            if tripped and not state['tripped']:
                state['tripped'] = tripped
                alert.write_text(json.dumps({'tripped': tripped, 'window': len(breaker.recent),
                                             'summary': summary(state['played'])}, ensure_ascii=False, indent=1))
                print(f'review breaker tripped {tripped}: collection stops, see {alert}', flush=True)
            if counts['done'] % 20 == 0:
                print(f"{counts['done']} played, {counts['success']} successful, {counts['error']} errors; "
                      f"flags {summary(state['played'][-WINDOW_PRINT:])['flag_rates']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, pilot))
    by_task = {}
    for row in state['played']:
        by_task.setdefault(row['id'], []).append(row)
    stopped = {}
    for task in tasks:
        rows = [r for r in by_task.get(task['id'], []) if r['seed'] < args.pilot]
        go, reason = task_verdict(rows)
        if not go:
            stopped[task['id']] = reason
    # Seed-major: one seed of every task, then the next, so one task's habit is spread over the breaker's window
    # (task-major order put a task's six identical reports side by side and tripped pool4's breaker at 8% overall).
    rest = [(t, s) for s in range(args.pilot, args.episodes) for t in tasks if t['id'] not in stopped
            if (t['id'], s) not in done]
    print(f'pilot done: {len(stopped)} tasks stopped for review, {len(rest)} more episodes to play', flush=True)
    if not state['tripped']:
        with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(one, rest))
    report = {**summary(state['played']), 'tasks_stopped': stopped, 'breaker': state['tripped']}
    Path(str(out) + '.review.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    dedup = write_candidates(state['played'], out.with_suffix('.candidates.jsonl'),
                             args.dedup_threshold, enabled=not args.no_dedup)
    print(f"dedup candidates: {dedup['input']} -> {dedup['kept']}; raw episodes unchanged", flush=True)
    print(f"review: {json.dumps({k: report[k] for k in ('episodes', 'successes', 'flag_rates')}, ensure_ascii=False)}",
          flush=True)
    if state['tripped']:
        sys.exit(4)

if __name__ == '__main__':
    main()
