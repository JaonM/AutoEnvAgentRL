"""What an agent sees of a V3 episode: function tools, opening messages and the mapping from a
function call back to a service action. Training (rl) and the policy gate both use this, so
the gate measures exactly the interface a trained policy gets.
"""
from __future__ import annotations

import copy
import json

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


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


def tools(instructions, category):
    """Function tools for one episode's public instructions."""
    out = _tools(instructions, category)
    return out if category == 'direct_response' else with_clock(out, instructions)


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


def messages(instructions):
    if instructions.get('user_mode') == 'natural':
        # The user's opening is a user turn; the rest is the assistant's working context.
        context = {key: instructions[key] for key in ('business_rules', 'answer_schema', 'time')}
        return [{'role': 'system', 'content': SYSTEM_NATURAL + '\n\n' + json.dumps(context, ensure_ascii=False)},
                {'role': 'user', 'content': instructions['user_request']}]
    public = {key: instructions[key] for key in PUBLIC}
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps(public, ensure_ascii=False)}]


def action(name, params, business_tools):
    """Service action for one function call; None for v3_answer (the caller submits)."""
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
