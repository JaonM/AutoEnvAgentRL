"""SFT sample augmentation: renamed business tools and varied tool-result formats.

The model must read its tool list, not remember names, and recognise a failed call however the
platform reports it (τ²-bench returns raw JSON or plain "Error: ..." text, our service an envelope).

B1 (SFT on 25 tasks) called tools that do not exist on τ²-bench (get_order_items, ...): a small model
memorizes the names it trained on. An augmented episode renames every business tool and its
parameters consistently (tool list, calls, call arguments, tool results, descriptions), using
synonyms for common words (get -> fetch/lookup/retrieve, order -> purchase, ...), and shuffles the
tool list. Values, results and behaviour are untouched. Platform tools are already renamed per
episode and are left alone.
"""
from __future__ import annotations

import copy
import json
import random
import re

SYNONYMS = [
    ('get', 'fetch', 'lookup', 'retrieve', 'read', 'show'),
    ('list', 'enumerate', 'browse', 'show_all'),
    ('search', 'find', 'query', 'locate'),
    ('check', 'verify', 'validate', 'inspect', 'evaluate'),
    ('create', 'open', 'add', 'make', 'register'),
    ('update', 'modify', 'change', 'edit', 'amend'),
    ('cancel', 'void', 'revoke', 'withdraw'),
    ('delete', 'remove', 'drop'),
    ('book', 'reserve', 'schedule'),
    ('refund', 'reimburse', 'repay'),
    ('submit', 'file', 'send'),
    ('apply', 'use', 'redeem'),
    ('customer', 'client', 'member', 'account_holder'),
    ('user', 'person', 'profile'),
    ('order', 'purchase'),
    ('booking', 'reservation'),
    ('item', 'product', 'line_item'),
    ('details', 'info', 'record', 'summary'),
    ('status', 'state'),
    ('quote', 'estimate', 'price_check'),
    ('payment', 'charge'),
    ('policy', 'rules'),
    ('request', 'ticket', 'case'),
]
LOOKUP = {word: group for group in SYNONYMS for word in group}
PREFIXES = ('do', 'run', 'svc', 'api', 'tool')


def rename_word(word, rng):
    group = LOOKUP.get(word)
    return rng.choice([w for w in group if w != word]) if group else word


def new_name(name, rng, taken, change_required=True):
    parts = name.split('_')
    renamed = '_'.join(rename_word(p, rng) if rng.random() < 0.8 else p for p in parts)
    if renamed == name and change_required:
        renamed = f'{rng.choice(PREFIXES)}_{name}'
    while renamed in taken:
        renamed = f'{renamed}_{rng.randrange(2, 9)}'
    taken.add(renamed)
    return renamed


def augment(episode, seed):
    """A copy of the episode with business tools and parameters renamed consistently."""
    rng = random.Random(seed)
    episode = copy.deepcopy(episode)
    platform = set((episode.get('naming') or {}).values()) | {'v3_chat', 'v3_ask', 'v3_confirm', 'v3_wait', 'v3_answer'}
    consent = (episode.get('naming') or {}).get('consent', 'consent')
    tools, params, taken = {}, {}, set()
    for spec in episode['tools']:
        function = spec['function']
        name = function['name']
        if name in platform:
            taken.add(name)
            continue
        tools[name] = new_name(name, rng, taken)
        mapping, used = {}, set()
        for parameter in (function.get('parameters') or {}).get('properties') or {}:
            mapping[parameter] = parameter if parameter == consent else new_name(parameter, rng, used, False)
        params[name] = mapping

    def rewrite_text(text, mapping):
        for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            if old != new:
                text = re.sub(rf'\b{re.escape(old)}\b', new, text)
        return text

    for spec in episode['tools']:
        function = spec['function']
        old = function['name']
        if old not in tools:
            continue
        mapping = params[old]
        function['name'] = tools[old]
        schema = function.get('parameters') or {}
        schema['properties'] = {mapping[k]: {**v, 'description': rewrite_text(v.get('description') or '', mapping)}
                                for k, v in (schema.get('properties') or {}).items()}
        schema['required'] = [mapping.get(k, k) for k in schema.get('required') or []]
        function['description'] = rewrite_text(rewrite_text(function.get('description') or '', mapping), tools)
    rng.shuffle(episode['tools'])
    for message in episode['messages']:
        for call in message.get('tool_calls') or []:
            old = call['function']['name']
            try:
                arguments = json.loads(call['function'].get('arguments') or '{}')
            except ValueError:
                continue
            if old in tools:
                call['function']['name'] = tools[old]
                arguments = {params[old].get(k, k): v for k, v in arguments.items()}
            proposal = arguments.get('proposal') if isinstance(arguments, dict) else None
            if isinstance(proposal, dict) and proposal.get('tool') in tools:  # a write proposed for consent
                target = proposal['tool']
                arguments['proposal'] = {**proposal, 'tool': tools[target],
                                         'args': {params[target].get(k, k): v
                                                  for k, v in (proposal.get('args') or {}).items()}}
            call['function']['arguments'] = json.dumps(arguments, ensure_ascii=False)
        if message.get('role') == 'tool' and message.get('name') in tools:
            message['name'] = tools[message['name']]
        if message.get('role') == 'system':
            message['content'] = rewrite_text(message.get('content') or '', tools)
    return episode


def envelope(content, style):
    """A tool result in one of the formats agents meet: 'envelope' (ours, unchanged), 'raw' (the result
    as JSON, a failure as "Error: <message>"), or 'text' (raw, failures as "<message>")."""
    if style == 'envelope':
        return content
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return content
    if not isinstance(value, dict) or 'ok' not in value:
        return content
    if value.get('ok'):
        result = value.get('result')
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    message = str(value.get('error') or 'request failed')
    return f'Error: {message}' if style == 'raw' else message


def vary_results(episode, seed):
    """A copy of the episode whose tool results all use one randomly chosen format."""
    style = random.Random(f'{seed}-envelope').choice(['envelope', 'raw', 'raw', 'text'])
    episode = copy.deepcopy(episode)
    for message in episode['messages']:
        if message.get('role') == 'tool':
            message['content'] = envelope(message.get('content'), style)
    return episode


GREETINGS = {'en': ['Hi! How can I help you today?', 'Hello, how can I help you?', 'Hi there, what can I do for you today?'],
             'zh': ['您好，请问有什么可以帮您？', '你好，有什么需要帮忙的吗？']}


def greet(episode, seed):
    """A copy of the episode opened by an agent greeting before the user's first message (τ²-bench
    conversations start that way). The greeting is context only: it precedes every trained turn."""
    episode = copy.deepcopy(episode)
    messages = episode['messages']
    first_user = next((i for i, m in enumerate(messages) if m['role'] == 'user'), None)
    if first_user is None or any(m['role'] == 'assistant' for m in messages[:first_user]):
        return episode
    text = messages[first_user].get('content') or ''
    language = 'zh' if any('\u4e00' <= ch <= '\u9fff' for ch in text) else 'en'
    greeting = random.Random(f'{seed}-greet').choice(GREETINGS[language])
    messages.insert(first_user, {'role': 'assistant', 'content': greeting, 'context_only': True})
    return episode


# System prompt layouts: the instructions and the policy in different wrappers, so the student learns to
# find and follow the policy whatever the layout (τ²-bench uses XML-like tags; we write plain text).
LAYOUTS = {
    'tags': '<instructions>\n{instructions}\nIn each turn either send one message to the user or make tool calls.\n'
            '</instructions>\n<policy>\n{policy}\n</policy>{clock}',
    'markdown': '# Role\n{instructions}\n\n# Policy\n{policy}{clock}',
    'policy_first': '{label}\n{policy}\n\n{instructions}{clock}',
}
POLICY_LABELS = ('\n\nPolicy:\n', '\n\n业务规则：\n')


def split_system(content):
    """(instructions, label, policy, clock) of our system prompt, or None if it does not parse."""
    for label in POLICY_LABELS:
        if label in content:
            instructions, rest = content.split(label, 1)
            policy, _, clock = rest.rpartition('\n\n')
            if not policy:
                policy, clock = rest, ''
            return instructions, label.strip(), policy, ('\n\n' + clock if clock else '')
    return None


def relayout(episode, seed):
    """A copy of the episode whose system prompt uses another layout (same instructions and policy)."""
    episode = copy.deepcopy(episode)
    system = episode['messages'][0] if episode['messages'] and episode['messages'][0]['role'] == 'system' else None
    parts = split_system(system['content']) if system else None
    if parts:
        instructions, label, policy, clock = parts
        layout = LAYOUTS[random.Random(f'{seed}-layout').choice(sorted(LAYOUTS))]
        system['content'] = layout.format(instructions=instructions, label=label, policy=policy, clock=clock)
    return episode
