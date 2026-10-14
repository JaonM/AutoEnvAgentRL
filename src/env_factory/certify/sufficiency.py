"""Policy sufficiency: does what the agent may read determine the graded outcome?

The oracle computes the outcome from the full specification; the agent only reads the agent
policy. If a rule that decides the outcome is missing from the policy, a learner is rewarded for
guessing, not for following the policy. The test removes every other difficulty: a model gets the
agent policy, the user's request and all of the user's private facts (approvals included), the
tool list and every read the reference made with its result, and must predict the writes and the
answer. The task passes when one of k predictions equals the reference exactly (writes as
[tool, args] without consent, in any order; every answer field, compared as the reward does).
Fields or writes no prediction gets right are reported as the likely policy gaps.

    python -m env_factory.certify.sufficiency BUNDLE [--k 2]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from env_factory.core import answers
from env_factory.runtime import canonical, run_reference, tool_specs

RESULT = 'sufficiency.json'

PROMPT = """你是一位严格照章办事的业务助理。下面给出你能看到的全部信息：业务手册、用户的需求、用户的全部私人信息（含用户同意或拒绝哪些操作），
以及已经替你查好的数据。请只依据业务手册，推断正确的处理结果。

业务手册：
{policy}

用户的需求：
{request}

用户的私人信息（approve_* 为 true 表示用户同意对应操作，false 表示不同意；不同意的操作不能执行）：
{facts}

可用工具：
{tools}

已经查好的数据（查询工具、参数和返回结果）：
{reads}

答案格式：
{schema}

请输出一个 JSON 对象：{{"writes": [{{"tool": "<写工具名>", "args": {{...不含 consent...}}}}], "answer": {{...按答案格式...}}}}
writes 列出按手册应当执行的全部写操作（没有就为空列表）。只输出 JSON。"""


def reference_view(loaded):
    """The reference run on the original world: its reads (with results), writes and answer."""
    request = loaded['request']
    result = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    reads = [{'tool': e['tool'], 'args': e.get('args'), 'result': e.get('result')} for e in result['trace']
             if e.get('kind') == 'tool' and e.get('status') == 'ok' and not e.get('write')]
    writes = [[e['tool'], e.get('args') or {}] for e in result['trace'] if e.get('write')]
    return reads, writes, result['answer']


def check(loaded, client, *, k=2):
    request, schema = loaded['request'], loaded['schema']
    reads, writes, answer = reference_view(loaded)
    specs = [{key: s[key] for key in ('name', 'kind', 'description', 'parameters')}
             for s in tool_specs(loaded['env'].make_tools())]
    prompt = PROMPT.format(policy=request.get('agent_policy') or request['business_rules'],
                           request=request['user_request'],
                           facts=json.dumps(request['user_facts'], ensure_ascii=False, indent=1),
                           tools=json.dumps(specs, ensure_ascii=False)[:12000],
                           reads=json.dumps(reads, ensure_ascii=False)[:20000],
                           schema=json.dumps(schema['fields'], ensure_ascii=False, indent=1))
    want = sorted(canonical(w) for w in writes)
    attempts, field_hits, write_hits = [], collections.Counter(), 0
    for number in range(k):
        content = client.chat([{'role': 'user', 'content': prompt}], temperature=0 if number == 0 else 0.7
                              ).get('content') or ''
        try:
            raw = json.loads(content[content.index('{'):content.rindex('}') + 1])
        except ValueError:
            raw = {}
        predicted = [[w.get('tool'), {a: v for a, v in (w.get('args') or {}).items() if a != 'consent'}]
                     for w in raw.get('writes') or [] if isinstance(w, dict)]
        fields = answers.compare(schema, answer, raw.get('answer') if isinstance(raw.get('answer'), dict) else None,
                                 answers.known_strings(request))
        writes_ok = sorted(canonical(w) for w in predicted) == want
        write_hits += writes_ok
        field_hits.update(name for name, ok in fields.items() if ok)
        attempts.append({'writes_ok': writes_ok, 'fields': fields, 'predicted_writes': predicted[:10],
                         'predicted_answer': raw.get('answer')})
    passed = any(a['writes_ok'] and all(a['fields'].values()) for a in attempts)
    return {'passed': passed, 'k': k, 'model': getattr(client, 'model', None),
            'gaps': {'writes': write_hits == 0 and bool(writes or any(a['predicted_writes'] for a in attempts)),
                     'fields': sorted(set(schema['fields']) - set(field_hits))},
            'reference_writes': writes, 'reference_answer': answer, 'attempts': attempts}


def run(folder, client, *, k=2):
    from env_factory.certify import bundle
    folder = Path(folder)
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    if loaded['oracle'] is None:
        report = {'passed': True, 'note': 'rubric task: no exact outcome to predict'}
    else:
        report = check(loaded, client, k=k)
    (folder / RESULT).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bundle')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--prefix', default='LLM')
    args = parser.parse_args()
    from env_factory.core.llm import ChatClient, load_env_file
    load_env_file()
    report = run(args.bundle, ChatClient(args.prefix), k=args.k)
    print(json.dumps({k: report.get(k) for k in ('passed', 'gaps', 'note')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
