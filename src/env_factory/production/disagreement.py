"""Disagreement mining: failures of a strong model that point at the task rather than the model.

The policy gate already plays every task k times with a strong model (policy-natural.json /
policy.json in the bundle). Its failures are free evidence about the reward:
- convergent failures: two or more failed episodes that end with the same committed writes and
  the same answer, different from the oracle's. Independent runs agreeing on one "wrong" outcome
  suggest the oracle, or the policy the agent reads, may be what is wrong.
- judged failures (--judge): a model (JUDGE_*, falling back to LLM_*) reads what the agent saw
  (the agent policy and the dialogue), the user's private facts and the oracle's outcome, and
  says whose fault the failure is: agent_error, reference_suspect, policy_gap or simulator.
Episodes the simulator audit already marked invalid are skipped. A task is flagged when it has a
convergent failure or a judged failure that is not agent_error; flagged tasks are listed for
review and can be left out of training (make_split.py --exclude). Nothing here changes a reward.

    python -m env_factory.production.disagreement BUNDLE_OR_DIR ... --output report.json [--judge]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
from pathlib import Path

from env_factory.runtime import canonical

VERDICTS = ('agent_error', 'reference_suspect', 'policy_gap', 'simulator')

JUDGE = """你在复核一次客服 agent 的失败。agent 只能看到“agent 手册”和对话；判分依据是独立程序算出的“标准结果”。
请判断这次失败该归咎于谁。

agent 手册：
{policy}

用户的私人信息（用户被问到时会说）：
{facts}

标准结果（判分依据）：
答案：{answer}
应发生的数据变化（按表列出应新增/应消失的行）：{changes}

agent 实际提交的写操作：{writes}
agent 实际给出的答案：{agent_answer}

对话与工具调用记录：
{transcript}

选一个结论：
- agent_error：手册写清楚了，用户也说清楚了，是 agent 自己做错或没做完。
- reference_suspect：按手册和用户说的话，agent 的做法同样正确或更正确，标准结果可疑。
- policy_gap：手册没有写清楚决定结果的规则，agent 无从得知标准结果为什么是这样。
- simulator：用户说错、漏说或编造信息，误导了 agent。
只输出一个 JSON 对象：{{"verdict": "agent_error|reference_suspect|policy_gap|simulator", "reason": "<一两句话>", "quote": "<支撑结论的手册原文或对话原话>"}}"""


def policy_reports(paths):
    """(bundle folder, gate report) for every bundle under the given paths that has a policy report."""
    found = []
    for path in map(Path, paths):
        candidates = [path] if (path / 'manifest.json').exists() else sorted(p.parent for p in path.rglob('manifest.json'))
        for folder in candidates:
            for name in ('policy-natural.json', 'policy.json'):
                if (folder / name).exists():
                    found.append((folder, json.loads((folder / name).read_text())))
                    break
    return found


def committed_writes(messages, writes):
    """[tool, args] of every write call whose result was ok, in order."""
    results = {m.get('tool_call_id'): m.get('content') for m in messages if m.get('role') == 'tool'}
    out = []
    for message in messages:
        for call in message.get('tool_calls') or []:
            name = call['function']['name']
            if name not in writes:
                continue
            try:
                ok = json.loads(results.get(call.get('id')) or '{}').get('ok') is True
                args = json.loads(call['function'].get('arguments') or '{}')
            except ValueError:
                continue
            if ok:
                out.append([name, {k: v for k, v in args.items() if k != 'consent'}])
    return out


def transcript(messages, limit=12000):
    lines = []
    for m in messages[1:]:
        if m.get('role') == 'assistant':
            if m.get('content'):
                lines.append(f"agent：{m['content']}")
            for call in m.get('tool_calls') or []:
                lines.append(f"agent 调用 {call['function']['name']}({call['function'].get('arguments')})")
        elif m.get('role') == 'tool':
            lines.append(f"工具结果：{str(m.get('content'))[:400]}")
        elif m.get('role') == 'user':
            lines.append(f"用户：{m.get('content')}")
    text = '\n'.join(lines)
    return text if len(text) <= limit else text[:limit // 2] + '\n…\n' + text[-limit // 2:]


def mine(folder, report, client=None):
    """The disagreement report of one task."""
    from env_factory.certify import bundle
    from env_factory.core.worlds import state_diff
    from env_factory.runtime import is_write
    from env_factory.serving import server
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    request = loaded['request']
    row = {'id': request['id'], 'bundle': str(folder), 'played': report.get('played'),
           'successes': report.get('successes'), 'failures': 0, 'skipped_invalid': 0, 'clusters': [], 'judged': []}
    if loaded['oracle'] is None:  # subjective task: graded by its rubric, no oracle outcome to dispute
        return {**row, 'flagged': False, 'note': 'rubric task'}
    service = server.Service(loaded['env'], loaded['oracle'], loaded['schema'], request,
                             valid_worlds=loaded['valid_worlds'])
    writes = {name for name, fn in loaded['env'].make_tools().items() if is_write(fn)}
    failures = []
    for episode in report.get('episodes') or []:
        if 'infrastructure' in episode or episode.get('success'):
            continue
        if episode.get('valid') is False:
            row['skipped_invalid'] += 1
            continue
        failures.append(episode)
    row['failures'] = len(failures)
    expected_of = {}
    for episode in failures:
        name = episode['world']
        if name not in expected_of and name in service.worlds:
            data, facts = service.worlds[name]
            outcome = loaded['oracle'].expected(json.loads(json.dumps(data)), json.loads(json.dumps(facts)),
                                                request['time']['start'])
            changes = {table: {'should_appear': d['missing'], 'should_disappear': d['unexpected']}
                       for table, d in state_diff(outcome['final_state'], data).items()}
            expected_of[name] = (outcome, changes, facts)
        episode['_writes'] = committed_writes(episode.get('messages') or [], writes)
    groups = collections.defaultdict(list)
    for index, episode in enumerate(failures):
        groups[canonical([episode['world'], episode['_writes'], episode.get('answer')])].append(index)
    for indices in groups.values():
        if len(indices) >= 2:
            first = failures[indices[0]]
            row['clusters'].append({'world': first['world'], 'episodes': len(indices), 'writes': first['_writes'],
                                    'answer': first.get('answer')})
    if client is not None:
        def judge(episode):
            outcome, changes, facts = expected_of.get(episode['world'], ({}, {}, {}))
            private = {k: v for k, v in facts.items() if not k.startswith('approve')}
            prompt = JUDGE.format(policy=request.get('agent_policy') or request['business_rules'],
                                  facts=json.dumps(private, ensure_ascii=False),
                                  answer=json.dumps(outcome.get('answer'), ensure_ascii=False),
                                  changes=json.dumps(changes, ensure_ascii=False)[:3000],
                                  writes=json.dumps(episode['_writes'], ensure_ascii=False),
                                  agent_answer=json.dumps(episode.get('answer'), ensure_ascii=False),
                                  transcript=transcript(episode.get('messages') or []))
            content = client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
            try:
                verdict = json.loads(content[content.index('{'):content.rindex('}') + 1])
            except ValueError:
                verdict = {}
            if verdict.get('verdict') not in VERDICTS:
                return {'world': episode['world'], 'verdict': None, 'reason': content[:200]}
            return {'world': episode['world'], **{k: verdict.get(k) for k in ('verdict', 'reason', 'quote')}}
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            row['judged'] = list(pool.map(judge, failures))
    suspicious = [j for j in row['judged'] if j.get('verdict') in ('reference_suspect', 'policy_gap')]
    row['flagged'] = bool(row['clusters'] or suspicious)
    row['reasons'] = (['convergent_failure'] if row['clusters'] else []) + sorted({j['verdict'] for j in suspicious})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='+', help='bundles, or folders searched for bundles with a policy report')
    parser.add_argument('--output', required=True)
    parser.add_argument('--judge', action='store_true', help='also ask a judge model about every failure')
    args = parser.parse_args()
    client = None
    if args.judge:
        from env_factory.core.llm import ChatClient, load_env_file
        load_env_file()
        client = ChatClient('JUDGE')
    rows = []
    for folder, report in policy_reports(args.paths):
        try:
            rows.append(mine(folder, report, client))
        except Exception as exc:  # one unreadable task never stops the report
            rows.append({'bundle': str(folder), 'error': f'{type(exc).__name__}: {exc}'[:300], 'flagged': False})
    verdicts = collections.Counter(j.get('verdict') for r in rows for j in r.get('judged', []))
    summary = {'tasks': len(rows), 'flagged': sum(r['flagged'] for r in rows),
               'failures': sum(r.get('failures', 0) for r in rows),
               'convergent_tasks': sum(bool(r.get('clusters')) for r in rows), 'verdicts': dict(verdicts),
               'flagged_ids': [r.get('id') for r in rows if r['flagged']]}
    Path(args.output).write_text(json.dumps({'summary': summary, 'tasks': rows}, ensure_ascii=False, indent=1,
                                            default=str))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
