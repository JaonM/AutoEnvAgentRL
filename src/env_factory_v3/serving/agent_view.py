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
SYSTEM = ('只使用提供的工具处理用户的请求。不要猜测用户的私人信息，需要时用 `v3_ask` 询问。每次写操作之前，先用 '
          '`v3_confirm` 确认这次具体的工具和参数，再把返回的凭证作为 `consent` 传给这次写操作。最后调用 `v3_answer` '
          '提交结果，或直接输出一个只含答案字段的 JSON 对象。')
SYSTEM_NATURAL = ('你是一名业务助理，正在和用户在线对话。直接用平实的话回复用户：需要什么信息就问，说明结果，提出方案。'
                  '只使用提供的工具，不要猜测用户的私人信息。')
SYSTEM_WRITES = ('每次写操作之前，先用 `v3_chat` 向用户提出这次具体的写操作（附上 proposal）；用户同意后会返回授权凭证，'
                 '把它作为 `consent` 传给这次写操作。')
SYSTEM_FINISH = ('确认结果后，调用 `v3_answer` 记录处理结果，并用平实的话把结果告诉用户；'
                 '用户表示没有其他问题时，对话结束。')
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
    return chosen | {'_text': rng.randrange(len(ANSWER_TEXTS))}


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
            texts = RECORD_TEXTS if fn['description'] in RECORD_TEXTS else ANSWER_TEXTS
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


def tools(instructions, category, names=None):
    """Function tools for one episode's public instructions (renamed when `names` is given)."""
    out = _tools(instructions, category)
    out = out if category == 'direct_response' else with_clock(out, instructions)
    return _rename(out, names) if names else out


def _tools(instructions, category):
    names = {tool['name'] for tool in instructions['tools']}
    if names & set(RESERVED):
        raise ValueError('business tool collides with a V3 runtime tool')
    out = [function(t['name'], t['description'], copy.deepcopy(t['parameters']['properties']),
                    t['parameters']['required']) for t in instructions['tools']]
    if instructions.get('user_mode') == 'natural':
        if any(t.get('kind') == 'write' for t in instructions['tools']):
            out.append(function(
                'v3_chat', '请用户同意一次具体的写操作：在 `message` 里向用户说明，并在 `proposal` 里附上这次写操作'
                '（工具名和参数）。用户同意时，结果里会带上授权凭证，把它作为 `consent` 传给这次写操作。'
                '（其他时候直接对用户说话即可。）',
                {'message': {'type': 'string', 'description': '对用户说的话'},
                 'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                              'required': ['tool', 'args'], 'description': '要用户同意的这次写操作'}},
                ['message', 'proposal']))
        fields = instructions['answer_schema']['fields']
        out.append(function('v3_answer', RECORD_TEXTS[0], copy.deepcopy(fields), list(fields)))
        return [out[-1]] if category == 'direct_response' else out
    facts = [t['key'] for t in instructions['user_topics'] if t['kind'] != 'approval']
    approvals = [t['key'] for t in instructions['user_topics'] if t['kind'] == 'approval']
    if facts:
        out.append(function('v3_ask', '按话题键向用户询问一项私人信息。',
                            {'key': {'type': 'string', 'enum': facts}}, ['key']))
    if approvals:
        out.append(function(
            'v3_confirm', '请用户确认一次具体的写操作。用户同意时返回授权凭证，拒绝时返回 null；把凭证作为 `consent` '
            '传给这次写操作。',
            {'key': {'type': 'string', 'enum': approvals},
             'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                          'required': ['tool', 'args']}}, ['key', 'proposal']))
    fields = instructions['answer_schema']['fields']
    out.append(function('v3_answer', ANSWER_TEXTS[0], copy.deepcopy(fields), list(fields)))
    return [out[-1]] if category == 'direct_response' else out


def with_clock(tools_, instructions):
    """v3_wait only for tasks whose clock advances (scheduled events)."""
    if not instructions.get('time', {}).get('advancing'):
        return tools_
    wait = function('v3_wait', '让虚拟时间流逝（不会真的等待），返回新的当前时间。',
                    {'seconds': {'type': 'integer', 'description': '要等待的虚拟时长，不小于 0'}}, ['seconds'])
    return [*tools_[:-1], wait, tools_[-1]]


def messages(instructions, names=None):
    swap = (lambda text: _swap(text, names)) if names else (lambda text: text)
    if instructions.get('user_mode') == 'natural':
        # The user's opening is a user turn; the rest is the assistant's working context. The answer
        # format lives in the answer tool's parameters only, not here a second time.
        writes = any(t.get('kind') == 'write' for t in instructions['tools'])
        time = instructions['time']
        clock = (f"当前时间：{time['start']}（虚拟时间，单位与业务规则一致）"
                 + ('；时间会推进，需要时用 `v3_wait` 等待。' if time.get('advancing') else '；本次对话中时间不变。'))
        text = SYSTEM_NATURAL + (SYSTEM_WRITES if writes else '') + SYSTEM_FINISH
        return [{'role': 'system', 'content': swap(text) + '\n\n业务规则：\n' + instructions['business_rules']
                 + '\n\n' + swap(clock)},
                {'role': 'user', 'content': instructions['user_request']}]
    public = {key: instructions[key] for key in PUBLIC}
    return [{'role': 'system', 'content': swap(SYSTEM)},
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
