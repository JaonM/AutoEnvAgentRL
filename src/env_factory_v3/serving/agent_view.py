"""What an agent sees of a V3 episode: function tools, opening messages and the mapping from a
function call back to a service action. Training (rl) and the policy gate both use this, so
the gate measures exactly the interface a trained policy gets.

Platform tools (answer, chat, ask, confirm, wait) and the `consent` argument of writes are
conventions of this platform, not of the businesses a trained agent will serve. So that a policy
learns to read its tool list instead of memorising these names, training renames them per
episode (naming(seed, 'random')); 'heldout' uses names never seen in training, to probe whether
a policy calls tools that are not offered. Without a naming everything keeps its canonical name.
"""
from __future__ import annotations

import copy
import json
import random
import re

RESERVED = ('v3_ask', 'v3_confirm', 'v3_chat', 'v3_wait', 'v3_answer')
# Every platform text the agent sees, per task language (request["language"]).
TEXT = {
    'zh': {
        'system': ('只使用提供的工具处理用户的请求。不要猜测用户的私人信息，需要时用 `v3_ask` 询问。每次写操作之前，先用 '
                   '`v3_confirm` 确认这次具体的工具和参数，再把返回的凭证作为 `consent` 传给这次写操作。最后调用 `v3_answer` '
                   '提交结果，或直接输出一个只含答案字段的 JSON 对象。'),
        'natural': ('你是一名业务助理，正在和用户在线对话。直接用平实的话回复用户：需要什么信息就问，说明结果，提出方案。'
                    '只使用提供的工具，不要猜测用户的私人信息。'),
        'writes': ('每次写操作之前，先用 `v3_chat` 向用户提出这次具体的写操作（附上 proposal）；用户同意后会返回授权凭证，'
                   '把它作为 `consent` 传给这次写操作。'),
        'finish': ('确认结果后，调用 `v3_answer` 记录处理结果，并用平实的话把结果告诉用户；'
                   '用户表示没有其他问题时，对话结束。'),
        # Without an answer tool the agent delivers by talking; the result is read from what it said.
        'tell': '确认结果后，用平实的话把处理结果完整告诉用户；用户表示没有其他问题时，对话结束。',
        'rules': '业务规则：',
        'clock': '当前时间：{start}（单位与业务规则一致）',
        'advancing': '；时间会推进，需要时用 `v3_wait` 等待。',
        'fixed': '；本次对话中时间不变。',
        'chat': ('请用户同意一次具体的写操作：在 `message` 里向用户说明，并在 `proposal` 里附上这次写操作'
                 '（工具名和参数）。用户同意时，结果里会带上授权凭证，把它作为 `consent` 传给这次写操作。'
                 '（其他时候直接对用户说话即可。）'),
        'chat_message': '对用户说的话', 'chat_proposal': '要用户同意的这次写操作',
        'ask': '按话题键向用户询问一项私人信息。',
        'confirm': ('请用户确认一次具体的写操作。用户同意时返回授权凭证，拒绝时返回 null；把凭证作为 `consent` '
                    '传给这次写操作。'),
        'wait': '让时间流逝（不会真的等待），返回新的当前时间。', 'wait_seconds': '要等待的时长，不小于 0',
        'answer': ('提交最终答案，结束本次对话。', '交付这次请求的最终结果；调用后对话结束。', '记录处理结果并结束本次服务。'),
        # Free dialogue: the answer tool records the result (it can be updated); the user ends the conversation.
        'record': ('记录这次请求的处理结果（结束前可以更新）；记录后把结果告诉用户。',
                   '登记处理结果；登记不会结束对话，仍需把结果告诉用户。',
                   '保存本次服务的处理结果，之后继续和用户沟通。'),
        'left_record': '（系统提示：请用 {answer} 记录处理结果，并把结果告诉用户；继续空谈，对话将被结束。）',
        'left_tell': '（系统提示：请把处理结果告诉用户；继续空谈，对话将被结束。）',
    },
    'en': {
        'system': ('Handle the user request using only the provided tools. Never guess private user facts: ask with '
                   '`v3_ask`. Before every write, call `v3_confirm` with the exact tool and arguments, then pass the '
                   'returned token as `consent` to that exact write. Finish with `v3_answer`, or output one JSON object '
                   'with exactly the answer fields.'),
        'natural': ('You are a customer service agent chatting with a user. Reply to the user directly in plain words: '
                    'ask for what you need, explain results, propose solutions. Use only the provided tools and never '
                    'guess private user details.'),
        'writes': ('Before every write, propose that exact write to the user with `v3_chat` (attach the proposal); when '
                   'the user agrees you get an authorization token, pass it as `consent` to that exact write.'),
        'finish': ('Once the outcome is settled, record it with `v3_answer` and tell the user the result in plain '
                   'words; the conversation ends when the user has nothing more to ask.'),
        'tell': ('Once the outcome is settled, tell the user the complete result in plain words; the conversation '
                 'ends when the user has nothing more to ask.'),
        'rules': 'Policy:',
        'clock': 'Current time: {start} (in the units the policy uses)',
        'advancing': '; time moves on: wait with `v3_wait` when needed.',
        'fixed': '; time does not change during this conversation.',
        'chat': ('Ask the user to approve one exact write: say it in `message` and attach the write (tool and '
                 'arguments) in `proposal`. When the user agrees, the result carries an authorization token: pass '
                 'it as `consent` to that exact write. (Otherwise just talk to the user directly.)'),
        'chat_message': 'what you say to the user', 'chat_proposal': 'the exact write to approve',
        'ask': 'Ask the user for one private fact by its topic key.',
        'confirm': ('Ask the user to confirm one exact write. Returns an authorization token when the user agrees, '
                    'null when they decline; pass the token as `consent` to that exact write.'),
        'wait': 'Let time pass (no real waiting); returns the new current time.',
        'wait_seconds': 'how long to wait, at least 0',
        'answer': ('Submit the final answer and end the conversation.',
                   'Deliver the final result of this request; this ends the conversation.',
                   'Record the outcome and close this request.'),
        'record': ('Record the outcome of this request (it can be updated before the end); then tell the user.',
                   'Log the result; logging does not end the conversation, still tell the user the result.',
                   'Save the outcome of this request, then keep talking with the user.'),
        'left_record': ('(System notice: record the outcome with {answer} and tell the user the result; more idle '
                        'talk ends the conversation.)'),
        'left_tell': '(System notice: tell the user the result; more idle talk ends the conversation.)',
    },
}
ANSWER_TEXTS, RECORD_TEXTS = TEXT['zh']['answer'], TEXT['zh']['record']


def language(instructions):
    return instructions.get('language', 'zh') if instructions.get('language') in TEXT else 'zh'


def left_notice(lang, answer=None):
    """The idle warning shown to the agent: name the answer tool when the episode offers one."""
    text = TEXT[lang if lang in TEXT else 'zh']
    return text['left_record'].format(answer=answer) if answer else text['left_tell']


PUBLIC = ('title', 'user_request', 'business_rules', 'user_topics', 'answer_schema', 'time')
ALIASES = {
    'v3_answer': ('v3_answer', 'submit_result', 'record_result', 'save_resolution', 'log_outcome',
                  'complete_request'),
    'v3_chat': ('v3_chat', 'request_approval', 'ask_user_approval', 'propose_action', 'seek_confirmation'),
    'v3_ask': ('v3_ask', 'ask_user', 'request_user_info', 'query_customer'),
    'v3_confirm': ('v3_confirm', 'confirm_with_user', 'get_approval', 'request_consent'),
    'v3_wait': ('v3_wait', 'wait_for_time', 'advance_clock', 'pass_time'),
    'consent': ('consent', 'approval_token', 'authorization', 'confirmation_token', 'user_consent'),
}
HELDOUT = {'v3_answer': 'return_outcome', 'v3_chat': 'obtain_user_ok', 'v3_ask': 'inquire_user',
           'v3_confirm': 'check_user_ok', 'v3_wait': 'let_time_pass', 'consent': 'permit_code'}
ANSWER_TEXTS = ('提交最终答案，结束本次对话。',
                '交付这次请求的最终结果；调用后对话结束。',
                '记录处理结果并结束本次服务。')
# Free dialogue: the answer tool records the result (it can be updated); the user ends the conversation.
RECORD_TEXTS = ('记录这次请求的处理结果（结束前可以更新）；记录后把结果告诉用户。',
                '登记处理结果；登记不会结束对话，仍需把结果告诉用户。',
                '保存本次服务的处理结果，之后继续和用户沟通。')


def naming(seed, mode='random', avoid=()):
    """Canonical platform name -> the name this episode uses. mode: fixed (canonical), random (drawn
    per seed from ALIASES) or heldout (names never used in training). `avoid`: business tool and
    argument names an alias must not collide with."""
    if mode == 'fixed':
        return {key: key for key in ALIASES} | {'_text': 0}
    if mode == 'heldout':
        return dict(HELDOUT) | {'_text': 0}
    if mode != 'random':
        raise ValueError(f'unknown tool naming {mode}')
    rng, avoid = random.Random(seed), set(avoid)
    chosen = {key: rng.choice([n for n in pool if n not in avoid] or [key]) for key, pool in ALIASES.items()}
    return chosen | {'_text': rng.randrange(len(TEXT['zh']['answer']))}


def platform_names():
    """Every name a platform tool ever had (to count calls to platform tools that were not offered)."""
    return {name for key, pool in ALIASES.items() if key != 'consent' for name in pool} | \
        {name for key, name in HELDOUT.items() if key != 'consent'}


def _swap(text, names):
    for key in ('v3_answer', 'v3_chat', 'v3_ask', 'v3_confirm', 'v3_wait'):
        text = text.replace(key, names[key])
    return re.sub(r'(?<![A-Za-z0-9_])consent(?![A-Za-z0-9_])', names['consent'], text)


def _rename(tools_, names):
    out = []
    for tool in copy.deepcopy(tools_):
        fn = tool['function']
        if fn['name'] == 'v3_answer':
            texts = next(t[k] for t in TEXT.values() for k in ('record', 'answer') if fn['description'] in t[k])
            fn['description'] = texts[names['_text']]
        fn['name'] = names.get(fn['name'], fn['name']) if fn['name'] in ALIASES else fn['name']
        fn['description'] = _swap(fn['description'], names)
        params = fn['parameters']
        if 'consent' in params['properties']:
            params['properties'][names['consent']] = params['properties'].pop('consent')
            params['required'] = [names['consent'] if r == 'consent' else r for r in params['required']]
        for spec in params['properties'].values():
            if isinstance(spec.get('description'), str):
                spec['description'] = _swap(spec['description'], names)
        out.append(tool)
    return out


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


def tools(instructions, category, names=None, answer_tool=True):
    """Function tools for one episode's public instructions (renamed when `names` is given). In free
    dialogue `answer_tool=False` drops the answer tool: the agent then delivers by talking."""
    out = _tools(instructions, category)
    if not answer_tool and instructions.get('user_mode') == 'natural':
        out = [t for t in out if t['function']['name'] != 'v3_answer']
    out = out if category == 'direct_response' else with_clock(out, instructions)
    return _rename(out, names) if names else out


def _tools(instructions, category):
    names = {tool['name'] for tool in instructions['tools']}
    if names & set(RESERVED):
        raise ValueError('business tool collides with a V3 runtime tool')
    out = [function(t['name'], t['description'], copy.deepcopy(t['parameters']['properties']),
                    t['parameters']['required']) for t in instructions['tools']]
    text = TEXT[language(instructions)]
    if instructions.get('user_mode') == 'natural':
        if any(t.get('kind') == 'write' for t in instructions['tools']):
            out.append(function(
                'v3_chat', text['chat'],
                {'message': {'type': 'string', 'description': text['chat_message']},
                 'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                              'required': ['tool', 'args'], 'description': text['chat_proposal']}},
                ['message', 'proposal']))
        fields = instructions['answer_schema']['fields']
        out.append(function('v3_answer', text['record'][0], copy.deepcopy(fields), list(fields)))
        return [out[-1]] if category == 'direct_response' else out
    facts = [t['key'] for t in instructions['user_topics'] if t['kind'] != 'approval']
    approvals = [t['key'] for t in instructions['user_topics'] if t['kind'] == 'approval']
    if facts:
        out.append(function('v3_ask', text['ask'], {'key': {'type': 'string', 'enum': facts}}, ['key']))
    if approvals:
        out.append(function(
            'v3_confirm', text['confirm'],
            {'key': {'type': 'string', 'enum': approvals},
             'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                          'required': ['tool', 'args']}}, ['key', 'proposal']))
    fields = instructions['answer_schema']['fields']
    out.append(function('v3_answer', text['answer'][0], copy.deepcopy(fields), list(fields)))
    return [out[-1]] if category == 'direct_response' else out


def with_clock(tools_, instructions):
    """v3_wait only for tasks whose clock advances (scheduled events)."""
    if not instructions.get('time', {}).get('advancing'):
        return tools_
    text = TEXT[language(instructions)]
    wait = function('v3_wait', text['wait'], {'seconds': {'type': 'integer', 'description': text['wait_seconds']}},
                    ['seconds'])
    return [*tools_[:-1], wait, tools_[-1]]


def messages(instructions, names=None, answer_tool=True):
    swap = (lambda text: _swap(text, names)) if names else (lambda text: text)
    if instructions.get('user_mode') == 'natural':
        # The user's opening is a user turn; the rest is the assistant's working context. The answer
        # format lives in the answer tool's parameters only, not here a second time.
        writes = any(t.get('kind') == 'write' for t in instructions['tools'])
        text, time = TEXT[language(instructions)], instructions['time']
        clock = text['clock'].format(start=time['start']) + (text['advancing'] if time.get('advancing') else text['fixed'])
        sep = '' if language(instructions) == 'zh' else ' '
        system = sep.join([text['natural']] + ([text['writes']] if writes else []) +
                          [text['finish'] if answer_tool else text['tell']])
        return [{'role': 'system', 'content': swap(system) + '\n\n' + text['rules'] + '\n' + instructions['business_rules']
                 + '\n\n' + swap(clock)},
                {'role': 'user', 'content': instructions['user_request']}]
    public = {key: instructions[key] for key in PUBLIC}
    return [{'role': 'system', 'content': swap(TEXT[language(instructions)]['system'])},
            {'role': 'user', 'content': json.dumps(public, ensure_ascii=False)}]


def action(name, params, business_tools, names=None):
    """Service action for one function call; None for the answer tool (the caller submits)."""
    if names:
        canonical = {alias: key for key, alias in names.items() if key in ALIASES and key != 'consent'}
        if name in business_tools:
            params = dict(params)
            if names['consent'] in params and 'consent' not in params:
                params['consent'] = params.pop(names['consent'])
            return {'type': 'tool', 'name': name, 'args': params}
        if name not in canonical:
            raise KeyError(f'unknown tool {name}')
        name = canonical[name]
    if name in business_tools:
        return {'type': 'tool', 'name': name, 'args': params}
    if name == 'v3_ask':
        return {'type': 'ask', 'key': params.get('key')}
    if name == 'v3_confirm':
        return {'type': 'confirm', 'key': params.get('key'), 'proposal': params.get('proposal')}
    if name == 'v3_chat':
        return {'type': 'chat', 'message': params.get('message'), 'proposal': params.get('proposal')}
    if name == 'v3_wait':
        return {'type': 'wait', 'seconds': params.get('seconds')}
    if name == 'v3_answer':
        return None
    raise KeyError(f'unknown tool {name}')


def speaks(instructions):
    """True when a plain assistant message is speech to the user (free dialogue), not an answer."""
    return instructions.get('user_mode') == 'natural'


def final_answer(text):
    """A final JSON object written as text (optionally fenced); None when it is not one."""
    text = (text or '').strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0]
    try:
        answer = json.loads(text)
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None
