"""What an agent sees of a V3 episode: function tools, opening messages and the mapping from a
function call back to a service action. Training (rl) and the policy gate both use this, so
the gate measures exactly the interface a trained policy gets.
"""
from __future__ import annotations

import copy
import json

RESERVED = ('v3_ask', 'v3_confirm', 'v3_answer')
SYSTEM = ('Handle the user request using only the public tools. Never guess private user facts: use '
          'v3_ask. Before every write call v3_confirm with the exact tool and args, then pass the returned '
          'token as `consent`. Finish with v3_answer, or a final JSON object with exactly the answer '
          'schema fields.')
PUBLIC = ('title', 'user_request', 'business_rules', 'user_topics', 'answer_schema', 'time')


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


def tools(instructions, category):
    """Function tools for one episode's public instructions."""
    names = {tool['name'] for tool in instructions['tools']}
    if names & set(RESERVED):
        raise ValueError('business tool collides with a V3 runtime tool')
    out = [function(t['name'], t['description'], copy.deepcopy(t['parameters']['properties']),
                    t['parameters']['required']) for t in instructions['tools']]
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


def messages(instructions):
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
    if name == 'v3_answer':
        return None
    raise KeyError(f'unknown tool {name}')


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
