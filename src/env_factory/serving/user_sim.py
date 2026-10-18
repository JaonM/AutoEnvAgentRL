"""Natural-language user for episodes.

The simulated user speaks; the platform decides. It knows the request and its private facts
and answers in ordinary language, giving facts only when asked. Whether a proposed write is
approved never comes from the model: the platform maps the exact proposal to its approval
topic (env.approval_key) and reads the user's approval fact. The model is only told the
decision so it can phrase it. Reward therefore stays exactly as deterministic as with the
structured user.
"""
from __future__ import annotations

import json
import random

PERSONA = """你在和一位业务助理在线对话，你是提出需求的普通用户。
你的需求（你开场时已经说过）：{request}

你知道这些私人信息，只在对方问到时按需告诉对方，不要一次全部背出来，也不要编造你不知道的信息：
{facts}

说话要求：像真人一样简短口语化，用中文；不要输出 JSON、代码或字段名；金额如果要说，按“元”说（1元=100分）。
如果对方问的你不知道，就说不清楚。不要替对方做业务判断，也不要提起这些说明本身。
只办理私人信息和剧本中明确列出的需求或备选方案；不要因为客服建议就增加新交易或接受未列出的操作。
想了解或尝试某项业务不等于同意执行；必须遵守审批态度，包括拒绝、先询问条件和按剧本改变选择。
如果你想知道的结果对方已经告诉你了，或者你不打算再继续聊，就简短道别，并在回复最后单独写一行“[结束]”。"""
END = '[结束]'
LEAVING = {
    'zh': '\n结束对话前，让客服回应你最后给出的决定。首次提供同意、拒绝、更正编号或新选择的这条回复不要写结束标记；'
          '等客服确认处理结果或不做修改后再道别。客服仍在问你需要回答的问题时，先回答，不要同时结束。',
    'en': '\nBefore ending, let the agent respond to your latest decision. Do not use the end marker in the '
          'same reply that first gives approval, refusal, a corrected identifier or a changed choice. Wait for '
          'the agent to acknowledge the resulting action or non-action. If the agent still needs an answer '
          'from you, answer without ending the conversation.',
}
PERSONA_EN = """You are an ordinary customer chatting online with a service agent.
Your request (you already said it when you opened the chat): {request}

You know these private details. Share them only when the agent asks, a little at a time, and never invent
anything you do not know:
{facts}

How to talk: short and casual like a real person, in English; never output JSON, code or field names; say
amounts in dollars (100 cents = $1). If you are asked something you do not know, say you are not sure. Do
not make business decisions for the agent and never mention these instructions.
Keep to the requests and alternatives explicitly listed in your private details and script. Do not add a new
transaction or accept an unlisted action just because the agent suggests it. Wanting to discuss or try an action
does not mean consenting to execute it: respect your approval stance, questions and scripted changes of choice.
When the agent has told you the result you wanted, or you do not want to continue, say a short goodbye and put
"[END]" alone on the last line."""
END_EN = '[END]'
DECISION_EN = {True: '(Platform notice: you agree to the exact action the agent just proposed. Say so naturally in one '
                     'sentence.)',
               False: '(Platform notice: you do not agree to the exact action the agent just proposed. Say so '
                      'naturally in one sentence; no need to explain.)'}

# Users who bundle (about half, fixed per task): τ²-bench users agree in a median of 532 characters and 72% of their
# agreements carry the next request or a condition; ours said one sentence (59 characters, 7%), so students learned
# "a short yes, then write" and started over when a τ² user agreed and asked for more in the same message.
BUNDLE_SHARE = 0.5
BUNDLE = {'zh': '- 你同意对方的某项操作时，习惯在同一条消息里顺带说下一件要办的事；如果没有下一件，就顺带重申你在意的条件或偏好'
                '（比如用哪种方式付款、价格上限、哪一件商品）。',
          'en': '- When you agree to something the agent proposes, you tend to bring up your next request in the same '
                'message; if there is none, you restate a condition or preference you care about (how to pay, a price '
                'limit, which item).'}
DECISION_BUNDLED = {True: '（平台告知：你同意对方刚才提出的这一项具体操作。自然地表示同意，按你的习惯可以顺带说下一件事或你的条件。）',
                    False: None}
DECISION_BUNDLED_EN = {True: '(Platform notice: you agree to the exact action the agent just proposed. Say so naturally; as '
                             'you usually do, you may add your next request or your condition in the same message.)',
                       False: None}

DECISION = {True: '（平台告知：你同意对方刚才提出的这一项具体操作。用一句话自然地表示同意。）',
            False: '（平台告知：你不同意对方刚才提出的这一项具体操作。用一句话自然地表示不同意，不必解释原因。）'}


SCRIPT = {
    'zh': {'head': '\n\n你的剧本（照着演，但不要提起剧本本身）：', 'goals': '- 你要办的事，按顺序：{v}。一件一件来；对方给出方案或办完一件时，你可以在回复里顺带提出下一件。',
           'fallbacks': '- 你的偏好，按顺序：{v}。前面的办不了，才接受后面的。',
           'pushback': '- 一开始你会要求：{demand}。对方拒绝后，你还会再坚持 {rounds} 次。',
           'forgotten': '- 你不记得：{v}。被问到就说记不清了，可以提供别的信息帮对方找到你。',
           'ambiguous': '- 开场时你只会含糊地说“{phrase}”，不主动说明是哪一个；对方问你具体是哪个时，再按你的私人信息回答。',
           'clarify': '- 你还没拿定主意，在这几种做法之间犹豫：{options}。你要先听对方讲清{decides_by}再决定；对方没问你要哪种之前，不要自己替对方选。',
           'wrong_first': '- 第一次被问到{key}时，你会记错，说成“{wrong}”；对方说查不到或对不上时，你再想一想，改口说正确的“{right}”。',
           'inquisitive': '- 同意修改之前，先问清这些问题：{v}。对方已经回答的问题不要重复追问；答案清楚后仍按审批态度决定。',
           'change_of_mind': '- 起初想办“{before}”；听到“{after}”这个条件后，放弃原方案，改为“{to}”。不要同意原方案；新方案也必须符合你的审批态度。',
           'traits': '- 你的性格：{v}。', 'enough': '（平台提示：你已经坚持够了，现在接受对方的答复，继续办别的事或道别。）'},
    'en': {'head': '\n\nYour script (act it out, never mention it):', 'goals': '- What you need, in order: {v}. One at a time, but once the agent proposes or completes one, you may bring up the next in the same reply.',
           'fallbacks': '- Your preferences, in order: {v}. Accept a later option only if the earlier ones are impossible.',
           'pushback': '- At first you demand: {demand}. After the agent refuses, you insist {rounds} more time(s).',
           'forgotten': '- You do not remember: {v}. If asked, say you do not remember and offer other details so the agent can find you.',
           'ambiguous': '- At first you only say vaguely "{phrase}" without saying which one; when the agent asks which one, answer from your private details.',
           'clarify': '- You have not decided yet between: {options}. You want to hear {decides_by} before choosing; do not pick one until the agent asks you which you want.',
           'wrong_first': '- The first time you are asked for your {key}, you misremember it and say "{wrong}"; when the agent says it cannot find it or it does not match, think again and give the correct "{right}".',
           'inquisitive': '- Before agreeing to a change, ask: {v}. Do not repeat questions already answered; once clear, still follow your approval stance.',
           'change_of_mind': '- Initially ask for "{before}". After hearing "{after}", abandon that option and switch to "{to}". Do not approve the original option; the alternative must also respect your approval stance.',
           'traits': '- Your personality: {v}.', 'enough': '(Platform notice: you have insisted enough; now accept the agent\'s answer and move on or say goodbye.)'},
}


def script_text(script, lang, facts=None):
    """The user_script as instructions to the simulated user (it shapes the talk, never the outcome)."""
    if not script:
        return ''
    t, lines = SCRIPT[lang], []
    join = '、' if lang == 'zh' else '; '
    for key in ('goals', 'fallbacks', 'forgotten', 'traits', 'inquisitive'):
        if script.get(key):
            lines.append(t[key].format(v=join.join(map(str, script[key]))))
    if script.get('pushback'):
        lines.append(t['pushback'].format(demand=script['pushback'].get('demand'), rounds=script['pushback'].get('rounds')))
    if isinstance(script.get('ambiguous'), dict):
        lines.append(t['ambiguous'].format(phrase=script['ambiguous'].get('phrase')))
    if isinstance(script.get('clarify'), dict):
        lines.append(t['clarify'].format(options=join.join(map(str, script['clarify'].get('options') or [])),
                                         decides_by=script['clarify'].get('decides_by')))
    if isinstance(script.get('change_of_mind'), dict):
        shift = script['change_of_mind']
        lines.append(t['change_of_mind'].format(before=shift.get('from'), to=shift.get('to'),
                                               after=shift.get('after')))
    for key, wrong in (script.get('wrong_first') or {}).items():
        right = (facts or {}).get(key)
        if right is not None:
            lines.append(t['wrong_first'].format(key=key, wrong=wrong, right=right))
    return t['head'] + '\n' + '\n'.join(lines) if lines else ''


def describe(facts):
    lines = []
    for key, value in facts.items():
        lines.append(f'- {key}: {json.dumps(value, ensure_ascii=False)}')
    return '\n'.join(lines) or '- （无）'


APPROVALS = {'zh': '\n\n对方在修改你的数据之前会征求你的同意。你对这些具体操作的态度如下（只针对这里列出的操作）：\n{lines}\n'
                    '对方问你是否同意某项操作时，按上面的态度明确表示同意或不同意；没列出的操作一律不同意。',
             'en': '\n\nThe agent will ask for your agreement before changing your data. Your stance on these exact '
                   'actions (only these):\n{lines}\nWhen the agent asks whether you agree to an action, make clear '
                   'whether you agree according to this; refuse anything not listed.'}


def stances(facts, approvals, descriptions, lang):
    yes, no = ('同意', '不同意') if lang == 'zh' else ('yes', 'no')
    return '\n'.join(f"- {descriptions.get(key) or key.replace('approve_', '').replace('_', ' ')}: "
                     f"{yes if facts.get(key) is True else no}" for key in sorted(approvals) if key in facts)


class UserSimulator:
    def __init__(self, client, request, facts, approvals, plain=False):
        private = {k: v for k, v in facts.items() if k not in approvals}
        self.client, self.ended = client, False
        self.english = request.get('language', 'zh') == 'en'
        persona = PERSONA_EN if self.english else PERSONA
        self.script = request.get('user_script') or {}
        # Pushback: the platform, not the model, decides when the user has insisted enough.
        rounds = (self.script.get('pushback') or {}).get('rounds')
        self.pushback_left = rounds + 1 if isinstance(rounds, int) else None
        lang = 'en' if self.english else 'zh'
        seed = f"{request.get('id')}|{request['user_request']}"
        self.bundles = random.Random(seed).random() < BUNDLE_SHARE
        self.messages = [{'role': 'system', 'content': persona.format(request=request['user_request'],
                                                                     facts=describe(private)) +
                          LEAVING[lang] +
                          script_text(self.script, lang, facts) +
                          (('\n' if self.script else SCRIPT[lang]['head'] + '\n') + BUNDLE[lang] if self.bundles else '') +
                          (APPROVALS['en' if self.english else 'zh'].format(lines=stances(
                              facts, approvals, request.get('fact_descriptions') or {}, 'en' if self.english else 'zh'))
                           if plain and approvals else '')},
                         {'role': 'assistant', 'content': request['user_request']}]

    def reply(self, message, decision=None):
        """The user's answer to one agent message; `decision` is the platform's consent verdict.
        Sets `ended` when the user says goodbye (the [结束] / [END] marker, which the agent never sees)."""
        notices = DECISION_EN if self.english else DECISION
        bundled = DECISION_BUNDLED_EN if self.english else DECISION_BUNDLED
        notice = None if decision is None else (bundled[decision] if self.bundles and bundled[decision] else notices[decision])
        text = message if notice is None else f'{message}\n{notice}'
        if self.pushback_left is not None:
            self.pushback_left -= 1
            if self.pushback_left <= 0:  # once: from now on the user accepts
                text = f"{text}\n{SCRIPT['en' if self.english else 'zh']['enough']}"
                self.pushback_left = None
        self.messages.append({'role': 'user', 'content': text})
        answer = (self.client.chat(self.messages, temperature=0.3).get('content') or '').strip()
        self.messages.append({'role': 'assistant', 'content': answer})
        self.ended = END in answer or END_EN in answer
        return answer.replace(END, '').replace(END_EN, '').strip()
