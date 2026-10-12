"""Natural-language user for V3 episodes.

The simulated user speaks; the platform decides. It knows the request and its private facts
and answers in ordinary language, giving facts only when asked. Whether a proposed write is
approved never comes from the model: the platform maps the exact proposal to its approval
topic (env.approval_key) and reads the user's approval fact. The model is only told the
decision so it can phrase it. Reward therefore stays exactly as deterministic as with the
structured user.
"""
from __future__ import annotations

import json

PERSONA = """你在和一位业务助理在线对话，你是提出需求的普通用户。
你的需求（你开场时已经说过）：{request}

你知道这些私人信息，只在对方问到时按需告诉对方，不要一次全部背出来，也不要编造你不知道的信息：
{facts}

说话要求：像真人一样简短口语化，用中文；不要输出 JSON、代码或字段名；金额如果要说，按“元”说（1元=100分）。
如果对方问的你不知道，就说不清楚。不要替对方做业务判断，也不要提起这些说明本身。
如果你想知道的结果对方已经告诉你了，或者你不打算再继续聊，就简短道别，并在回复最后单独写一行“[结束]”。"""
END = '[结束]'

DECISION = {True: '（平台告知：你同意对方刚才提出的这一项具体操作。用一句话自然地表示同意。）',
            False: '（平台告知：你不同意对方刚才提出的这一项具体操作。用一句话自然地表示不同意，不必解释原因。）'}


def describe(facts):
    lines = []
    for key, value in facts.items():
        lines.append(f'- {key}: {json.dumps(value, ensure_ascii=False)}')
    return '\n'.join(lines) or '- （无）'


class UserSimulator:
    def __init__(self, client, request, facts, approvals):
        private = {k: v for k, v in facts.items() if k not in approvals}
        self.client, self.ended = client, False
        self.messages = [{'role': 'system', 'content': PERSONA.format(request=request['user_request'],
                                                                     facts=describe(private))},
                         {'role': 'assistant', 'content': request['user_request']}]

    def reply(self, message, decision=None):
        """The user's answer to one agent message; `decision` is the platform's consent verdict.
        Sets `ended` when the user says goodbye (the [结束] marker, which the agent never sees)."""
        text = message if decision is None else f'{message}\n{DECISION[decision]}'
        self.messages.append({'role': 'user', 'content': text})
        answer = (self.client.chat(self.messages, temperature=0.3).get('content') or '').strip()
        self.messages.append({'role': 'assistant', 'content': answer})
        self.ended = END in answer
        return answer.replace(END, '').strip()
