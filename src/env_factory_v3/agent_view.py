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
SYSTEM = ('Handle the user request using only the public tools. Never guess private user facts: use '
          'v3_ask. Before every write call v3_confirm with the exact tool and args, then pass the returned '
          'token as `consent`. Finish with v3_answer, or a final JSON object with exactly the answer '
          'schema fields.')
SYSTEM_NATURAL = ('You are a business assistant chatting with a user. Talk to the user in plain messages, in '
                  'their language: ask for what you need, explain, propose. Use only the public tools; never '
                  'guess private details. Before every write, propose that exact write with v3_chat (it carries '
                  'the proposal) and pass the returned consent token as `consent`. Finish by calling v3_answer.')
PUBLIC = ('title', 'user_request', 'business_rules', 'user_topics', 'answer_schema', 'time')
ALIASES = {
    'v3_answer': ('v3_answer', 'submit_result', 'finish_task', 'deliver_answer', 'submit_final_answer',
                  'complete_request'),
    'v3_chat': ('v3_chat', 'request_approval', 'ask_user_approval', 'propose_action', 'seek_confirmation'),
    'v3_ask': ('v3_ask', 'ask_user', 'request_user_info', 'query_customer'),
    'v3_confirm': ('v3_confirm', 'confirm_with_user', 'get_approval', 'request_consent'),
    'v3_wait': ('v3_wait', 'wait_for_time', 'advance_clock', 'pass_time'),
    'consent': ('consent', 'approval_token', 'authorization', 'confirmation_token', 'user_consent'),
}
HELDOUT = {'v3_answer': 'return_outcome', 'v3_chat': 'obtain_user_ok', 'v3_ask': 'inquire_user',
           'v3_confirm': 'check_user_ok', 'v3_wait': 'let_time_pass', 'consent': 'permit_code'}
ANSWER_TEXTS = ('Submit the final answer and end the episode.',
                'Deliver the final result of this request; this ends the conversation.',
                'Record the final outcome for the user and finish.')


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
    return re.sub(r'\bconsent\b', names['consent'], text)


def _rename(tools_, names):
    out = []
    for tool in copy.deepcopy(tools_):
        fn = tool['function']
        if fn['name'] == 'v3_answer':
            fn['description'] = ANSWER_TEXTS[names['_text']]
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
        out.append(function(
            'v3_chat', 'Ask the user to approve one exact write: say it in `message` and attach the exact '
            '`proposal`. When the user agrees the result carries a consent token to pass as `consent` to '
            'exactly that write. (For everything else just write to the user directly.)',
            {'message': {'type': 'string', 'description': 'what you say to the user, in their language'},
             'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                          'required': ['tool', 'args'],
                          'description': 'the exact write to approve'}}, ['message', 'proposal']))
        fields = instructions['answer_schema']['fields']
        out.append(function('v3_answer', 'Submit the final answer and end the episode.', copy.deepcopy(fields),
                            list(fields)))
        return [out[-1]] if category == 'direct_response' else out
    facts = [t['key'] for t in instructions['user_topics'] if t['kind'] != 'approval']
    approvals = [t['key'] for t in instructions['user_topics'] if t['kind'] == 'approval']
    if facts:
        out.append(function('v3_ask', 'Ask the user for one private fact by its topic key.',
                            {'key': {'type': 'string', 'enum': facts}}, ['key']))
    if approvals:
        out.append(function(
            'v3_confirm', 'Ask the user to approve one exact write. Returns a consent token, or null when '
            'the user declines; pass the token as `consent` to exactly that write.',
            {'key': {'type': 'string', 'enum': approvals},
             'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                          'required': ['tool', 'args']}}, ['key', 'proposal']))
    fields = instructions['answer_schema']['fields']
    out.append(function('v3_answer', 'Submit the final answer and end the episode.', copy.deepcopy(fields),
                        list(fields)))
    return [out[-1]] if category == 'direct_response' else out


def with_clock(tools_, instructions):
    """v3_wait only for tasks whose clock advances (scheduled events)."""
    if not instructions.get('time', {}).get('advancing'):
        return tools_
    wait = function('v3_wait', 'Let virtual time pass (no real sleeping); returns the new time.',
                    {'seconds': {'type': 'integer', 'description': 'virtual seconds to wait, >= 0'}}, ['seconds'])
    return [*tools_[:-1], wait, tools_[-1]]


def messages(instructions, names=None):
    swap = (lambda text: _swap(text, names)) if names else (lambda text: text)
    if instructions.get('user_mode') == 'natural':
        # The user's opening is a user turn; the rest is the assistant's working context.
        context = {key: instructions[key] for key in ('business_rules', 'answer_schema', 'time')}
        return [{'role': 'system', 'content': swap(SYSTEM_NATURAL) + '\n\n' + json.dumps(context, ensure_ascii=False)},
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
