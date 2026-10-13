"""Simulator disclosure audit: did the simulated user, not the agent, cause a failure?

The natural user is a model and can slip: claim not to know a fact it was given, state a value
that contradicts its facts, add a requirement it does not have, or answer against the platform's
consent decision. A failed episode caused by such a slip is not evidence about the agent, so it is
marked invalid for training (rl drops it from the group) rather than scored 0 as the agent's fault.

Only failed free-dialogue episodes are audited. A model (V3_JUDGE_*, falling back to LLM_*;
temperature 0; cached by content) reads the user's private facts, script and the dialogue and lists
faults; each fault must quote the user's words verbatim and, for withheld and contradicted facts,
name one of the user's facts, so an invented fault is discarded. The reward is never changed here.
"""
from __future__ import annotations

import copy
import hashlib
import json

KINDS = ('withheld', 'contradicted', 'invented', 'decision')

PROMPT = """你在审核一次客服对话里的“模拟用户”是否演错了。用户由程序扮演，按下面的私人信息和剧本说话。

用户的私人信息（用户本该知道、被问到时应如实说出）：
{facts}

用户剧本（其中“不记得”的信息，用户说记不清是正确的）：
{script}

对话（“平台告知”是平台替用户做出的同意或拒绝决定，用户的话必须与之一致）：
{dialogue}

只找用户一方的以下错误，客服的错误不算：
- withheld：客服明确问到某项私人信息，用户却说不知道、不记得或拒绝提供（剧本里“不记得”的除外）。
- contradicted：用户说出的值与私人信息不一致（例如编号、日期、金额、选择说错）。
- invented：用户提出了私人信息和需求里都没有、并且会改变正确处理方式的新要求或新情况。
- decision：用户的表态与平台告知的决定相反。
口语化的说法、单位换算（如“360元”对应 36000 分）、只说一部分信息都不算错。没把握就不要列。

只输出一个 JSON 对象：{{"faults": [{{"kind": "withheld|contradicted|invented|decision", "fact": "<相关私人信息的键名，invented 可为空>", "quote": "<用户原话，逐字摘录>", "reason": "<一句话>"}}]}}；没有错误时 faults 为空列表。"""


def dialogue_of(trace):
    """(agent message, platform decision or None, user reply) for every chat turn."""
    turns = []
    for entry in trace:
        if entry.get('kind') != 'chat':
            continue
        turns.append((str(entry.get('message') or ''), entry.get('decision'), str(entry.get('reply') or '')))
    return turns


def render(turns):
    lines = []
    for number, (message, decision, reply) in enumerate(turns, 1):
        lines.append(f'[{number}] 客服：{message}')
        if decision is not None:
            lines.append(f'[{number}] 平台告知：用户{"同意" if decision else "不同意"}这项操作')
        lines.append(f'[{number}] 用户：{reply}')
    return '\n'.join(lines)


class Auditor:
    def __init__(self, client):
        self.client, self.cache = client, {}

    def audit(self, facts, script, trace):
        """{"faults": [...], "valid": bool}; faults that do not quote the user or name a fact are dropped."""
        turns = dialogue_of(trace)
        if not turns:
            return {'faults': [], 'valid': True}
        prompt = PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1),
                               script=json.dumps(script or {}, ensure_ascii=False), dialogue=render(turns))
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.cache:
            from env_factory_v3.core.llm import ProviderError
            for _ in range(2):
                content = self.client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
                try:
                    raw = json.loads(content[content.index('{'):content.rindex('}') + 1])
                    if isinstance(raw, dict) and isinstance(raw.get('faults'), list):
                        break
                except ValueError:
                    continue
            else:
                raise ProviderError('disclosure auditor returned no usable verdict')
            self.cache[key] = raw['faults']
        said = '\n'.join(reply for _, _, reply in turns)
        faults = [f for f in copy.deepcopy(self.cache[key]) if verified(f, facts, said)]
        return {'faults': faults, 'valid': not faults}


def verified(fault, facts, said):
    """A fault counts only when it quotes the user's words and, where required, names a real fact."""
    if not isinstance(fault, dict) or fault.get('kind') not in KINDS:
        return False
    quote = str(fault.get('quote') or '').strip()
    if len(quote) < 2 or quote not in said:
        return False
    return fault['kind'] not in ('withheld', 'contradicted') or fault.get('fact') in facts
