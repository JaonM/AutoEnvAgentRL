"""Policy sufficiency: does what the agent may read determine the graded outcome?

The verifier computes the outcome from the full specification; the agent only reads the agent
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

查询记录（按参考处理过程的发生顺序排列）：
{reads}

时序说明：phase=initial 是第一次写操作之前的查询；phase=after_write 是写操作之后的查询，
可能是中间步骤或最终复查。这些记录不是同时成立的初始状态。请从用户本次请求开始推断完整处理过程，
不要因为后面的复查已经显示操作后的状态而省略造成该状态的操作。工具 kind=read 不属于 writes。

答案格式：
{schema}

请输出一个 JSON 对象：{{"writes": [{{"tool": "<写工具名>", "args": {{...不含 consent...}}}}], "answer": {{...按答案格式...}}}}
writes 列出按手册应当执行的全部写操作（没有就为空列表）。只输出 JSON。"""


def reference_view(loaded):
    """The reference run on the original world: its reads (with results), writes and answer."""
    request = loaded['request']
    result = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    reads, written = [], False
    for index, event in enumerate(result['trace']):
        if event.get('kind') != 'tool' or event.get('status') != 'ok':
            continue
        if event.get('write'):
            written = True
        else:
            reads.append({'tool': event['tool'], 'args': event.get('args'), 'result': event.get('result'),
                          'trace_index': index, 'phase': 'after_write' if written else 'initial'})
    writes = [[e['tool'], e.get('args') or {}] for e in result['trace'] if e.get('write')]
    return reads, writes, result['answer']


def staged_reference(loaded):
    """Expose each decision's past evidence, then receipts for the reporting stage.

    An identifier created by a write cannot be predicted before executing it. Later
    decisions can depend on that receipt, so revealing every receipt up front would
    leak future actions. The business reference and grading are not changed here.
    """
    request = loaded['request']
    result = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    evidence, decisions = [], []
    for index, event in enumerate(result['trace']):
        if event.get('kind') != 'tool' or event.get('status') != 'ok':
            continue
        action = [event['tool'], {k: v for k, v in (event.get('args') or {}).items() if k != 'consent'}]
        if event.get('write'):
            decisions.append({'evidence': list(evidence), 'expected': action})
        evidence.append({'tool': action[0], 'args': action[1], 'result': event.get('result'),
                         'trace_index': index, 'kind': 'write_receipt' if event.get('write') else 'read'})
    if not decisions:
        decisions.append({'evidence': list(evidence), 'expected': None})
    return decisions, evidence, result['answer']


STAGED_PROMPT = """You are checking whether a business policy determines the correct decisions and report.
Use only the supplied policy, customer facts, tools and evidence. Approval=false means do not perform that action.
Policy:
{policy}
Customer request:
{request}
Customer facts:
{facts}
Tools:
{tools}
Evidence in chronological order (write_receipt means an action already executed; do not repeat it):
{evidence}
{instruction}
Return only JSON. Never guess identifiers that have not appeared in the evidence.
"""


def _prediction(client, prompt, temperature):
    content = client.chat([{'role': 'user', 'content': prompt}], temperature=temperature).get('content') or ''
    try:
        raw = json.loads(content[content.index('{'):content.rindex('}') + 1])
    except ValueError:
        raw = {}
    return (raw if isinstance(raw, dict) else {}), content


def reported_answer(raw, schema):
    """Accept either an answer wrapper or the exact schema object, never infer values."""
    if isinstance(raw.get('answer'), dict):
        return raw['answer']
    if set(raw) == set(schema['fields']):
        return raw
    return None


def check(loaded, client, *, k=2):
    request, schema = loaded['request'], loaded['schema']
    decisions, receipts, answer = staged_reference(loaded)
    writes = [d['expected'] for d in decisions if d['expected'] is not None]
    specs = [{key: s[key] for key in ('name', 'kind', 'description', 'parameters')}
             for s in tool_specs(loaded['env'].make_tools())]
    context = dict(policy=request.get('agent_policy') or request['business_rules'], request=request['user_request'],
                   facts=json.dumps(request['user_facts'], ensure_ascii=False), tools=json.dumps(specs, ensure_ascii=False))
    attempts, field_hits, write_hits = [], collections.Counter(), 0
    for number in range(k):
        temperature = 0 if number == 0 else 0.7
        stages, predicted = [], []
        for stage_index, decision in enumerate(decisions):
            prompt = STAGED_PROMPT.format(**context, evidence=json.dumps(decision['evidence'], ensure_ascii=False),
                instruction='Decide the remaining write action(s) justified by the available evidence. '
                            'Independent actions may be listed together in any order. Defer actions whose '
                            'parameters require a result not yet observed; do not repeat completed actions. '
                            'Return {"writes": [{"tool": "name", "args": {}}]}; exclude consent tokens. '
                            'Return {"writes": []} if no write is required. Do not predict the final report.')
            raw, raw_content = _prediction(client, prompt, temperature)
            actions = [[w.get('tool'), {a: v for a, v in w['args'].items() if a != 'consent'}]
                       for w in (raw.get('writes') if isinstance(raw.get('writes'), list) else [])
                       if isinstance(w, dict) and isinstance(w.get('args'), dict)]
            remaining = collections.Counter(canonical(d['expected']) for d in decisions[stage_index:]
                                            if d['expected'] is not None)
            proposed = collections.Counter(map(canonical, actions))
            ok = isinstance(raw.get('writes'), list) and len(actions) == len(raw['writes']) \
                and proposed <= remaining and (canonical(decision['expected']) in proposed
                                                if decision['expected'] is not None else not proposed)
            predicted.extend(actions)
            stages.append({'writes_ok': ok, 'predicted_writes': actions,
                           'evidence_events': len(decision['evidence']), 'raw_response': raw_content})
        prompt = STAGED_PROMPT.format(**context, evidence=json.dumps(receipts, ensure_ascii=False),
            instruction='Execution is complete. Report the outcome from the policy and observed receipts. '
                        'Return {"answer": {...}} using this schema: ' + json.dumps(schema['fields'], ensure_ascii=False))
        raw, raw_content = _prediction(client, prompt, temperature)
        predicted_answer = reported_answer(raw, schema)
        fields = answers.compare(schema, answer, predicted_answer,
                                 answers.known_strings(request))
        writes_ok = all(s['writes_ok'] for s in stages)
        write_hits += writes_ok
        field_hits.update(name for name, ok in fields.items() if ok)
        attempts.append({'writes_ok': writes_ok, 'fields': fields, 'predicted_writes': predicted[:10],
                         'predicted_answer': predicted_answer, 'raw_report_response': raw_content,
                         'report_format_valid': predicted_answer is not None, 'decision_stages': stages})
    passed = any(a['writes_ok'] and all(a['fields'].values()) for a in attempts)
    return {'passed': passed, 'k': k, 'method': 'decisions_then_observed_report_v2',
            'model': getattr(client, 'model', None),
            'gaps': {'writes': write_hits == 0 and bool(writes or any(a['predicted_writes'] for a in attempts)),
                     'fields': sorted(set(schema['fields']) - set(field_hits))},
            'reference_writes': writes, 'reference_answer': answer, 'attempts': attempts}


def run(folder, client, *, k=2):
    from env_factory.certify import bundle
    folder = Path(folder)
    loaded = bundle.load_bundle(folder, require_current_platform=False)
    if loaded['verifier'] is None:
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
