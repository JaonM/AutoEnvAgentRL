"""Public agent data for the SFT mix (plan E16): When2Call, ToolACE and glaive converted to our episode format.

Each sample becomes an episode like collect_teacher.py's (OpenAI-style messages with tool_calls, an OpenAI tool
list, success=True) marked public=True, so sft_train.py renders it the same way and skips the augmentations that
assume our platform. Kept: When2Call asking for missing information (most) and declining out-of-scope requests;
ToolACE multi-turn tool use (pythonic calls parsed with ast; tool names with spaces renamed consistently); glaive
function calls. Dropped: unparsable samples, turns with more than MAX_CALLS calls, and any sample sharing an 8-gram
with τ²-bench's policies or task scenarios (contamination check).

    python experiments/public_data.py --root data/public_agent --output data/public_agent/public_sft.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

MAX_CALLS = 3
TYPES = {'str': 'string', 'string': 'string', 'int': 'integer', 'integer': 'integer', 'float': 'number',
         'number': 'number', 'bool': 'boolean', 'boolean': 'boolean', 'list': 'array', 'array': 'array',
         'dict': 'object', 'object': 'object'}


def safe(name):
    return re.sub(r'\W+', '_', name).strip('_') or 'tool'


def schema(spec):
    """An OpenAI function tool from the datasets' function descriptions (types like "str" or "dict" normalised)."""
    def norm(value):
        if isinstance(value, dict):
            out = {k: norm(v) for k, v in value.items() if k not in ('default', 'required') or isinstance(v, list)}
            if isinstance(out.get('type'), str):
                base = out['type'].split(',')[0].split('[')[0].strip().lower()
                out['type'] = TYPES.get(base, 'string')
            return out
        if isinstance(value, list):
            return [norm(v) for v in value]
        return value
    parameters = norm(spec.get('parameters') or {'type': 'object', 'properties': {}})
    parameters['type'] = 'object'
    return {'type': 'function', 'function': {'name': safe(spec['name']), 'description': spec.get('description', ''),
                                             'parameters': parameters}}


def call(name, arguments, index):
    return {'id': f'call_{index}', 'type': 'function',
            'function': {'name': safe(name), 'arguments': json.dumps(arguments, ensure_ascii=False)}}


def when2call(path, rng, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rng.shuffle(rows)
    asking = [r for r in rows if '?' in r['messages'][-1]['content']]
    declining = [r for r in rows if '?' not in r['messages'][-1]['content']]
    picked = asking[:int(count * 0.75)] + declining[:count - int(count * 0.75)]
    out = []
    for index, row in enumerate(picked):
        tools = [schema(json.loads(t) if isinstance(t, str) else t) for t in row.get('tools') or []]
        out.append({'id': f'when2call-{index}', 'messages': row['messages'], 'tools': tools, 'source': 'when2call'})
    return out


def pythonic_calls(text, names):
    """[(name, arguments)] from "[Name(a=1), Other Name(b='x')]", or None."""
    body = text.strip()
    if not (body.startswith('[') and body.endswith(']')):
        return None
    for name in sorted(names, key=len, reverse=True):
        body = body.replace(f'{name}(', f'{safe(name)}(')
    try:
        tree = ast.parse(body, mode='eval').body
        if not isinstance(tree, ast.List):
            return None
        calls = []
        for node in tree.elts:
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.args:
                return None
            calls.append((node.func.id, {k.arg: ast.literal_eval(k.value) for k in node.keywords}))
        return calls
    except (SyntaxError, ValueError):
        return None


def toolace(path, rng, count):
    rows = json.loads(Path(path).read_text())
    rng.shuffle(rows)
    out = []
    for row in rows:
        if len(out) >= count:
            break
        match = re.search(r'\[\{.*\}\]', row['system'], re.S)
        try:
            specs = json.loads(match.group(0)) if match else []
        except ValueError:
            continue
        names = [s['name'] for s in specs if isinstance(s, dict) and 'name' in s]
        messages, pending, ok, index = [], [], True, 0
        for turn in row['conversations']:
            role, value = turn['from'], turn['value']
            if role == 'user':
                messages.append({'role': 'user', 'content': value})
            elif role == 'assistant':
                calls = pythonic_calls(value, names) if value.strip().startswith('[') else None
                if calls is None and value.strip().startswith('['):
                    ok = False
                    break
                if calls:
                    if len(calls) > MAX_CALLS:
                        ok = False
                        break
                    pending = [call(name, args, index + i) for i, (name, args) in enumerate(calls)]
                    index += len(calls)
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': pending})
                else:
                    messages.append({'role': 'assistant', 'content': value})
            elif role == 'tool':
                try:
                    results = json.loads(value)
                except ValueError:
                    ok = False
                    break
                for item, made in zip(results if isinstance(results, list) else [results], pending):
                    messages.append({'role': 'tool', 'tool_call_id': made['id'], 'name': made['function']['name'],
                                     'content': json.dumps(item.get('results', item) if isinstance(item, dict) else item,
                                                           ensure_ascii=False)})
        has_calls = any(m.get('tool_calls') for m in messages)
        if ok and has_calls and messages and messages[-1]['role'] == 'assistant':
            out.append({'id': f'toolace-{len(out)}', 'messages': messages, 'tools': [schema(s) for s in specs],
                        'source': 'toolace'})
    return out


def glaive(path, rng, count):
    rows = json.loads(Path(path).read_text())
    rng.shuffle(rows)
    out = []
    decoder = json.JSONDecoder()
    for row in rows:
        if len(out) >= count:
            break
        text, specs, position = row['system'], [], row['system'].find('{')
        while 0 <= position < len(text):
            try:
                spec, end = decoder.raw_decode(text, position)
            except ValueError:
                break
            specs.append(spec)
            position = text.find('{', end)
        if not specs:
            continue
        parts = re.split(r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)', row['chat'])
        messages, pending, ok, index = [], None, True, 0
        for marker, value in zip(parts[1::2], parts[2::2]):
            value = value.replace('<|endoftext|>', '').strip()
            if marker == 'USER:':
                messages.append({'role': 'user', 'content': value})
            elif marker == 'ASSISTANT:' and value.startswith('<functioncall>'):
                try:
                    payload = json.loads(value[len('<functioncall>'):].strip())
                    arguments = payload.get('arguments') or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments) if arguments.strip().startswith('{') else {}
                except ValueError:
                    try:
                        payload = ast.literal_eval(value[len('<functioncall>'):].strip())
                        arguments = payload.get('arguments') or {}
                        arguments = ast.literal_eval(arguments) if isinstance(arguments, str) else arguments
                    except (ValueError, SyntaxError):
                        ok = False
                        break
                pending = call(payload['name'], arguments, index)
                index += 1
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [pending]})
            elif marker == 'ASSISTANT:':
                messages.append({'role': 'assistant', 'content': value})
            elif marker == 'FUNCTION RESPONSE:' and pending:
                messages.append({'role': 'tool', 'tool_call_id': pending['id'], 'name': pending['function']['name'],
                                 'content': value})
                pending = None
        if ok and any(m.get('tool_calls') for m in messages) and messages[-1]['role'] == 'assistant':
            out.append({'id': f'glaive-{len(out)}', 'messages': messages, 'tools': [schema(s) for s in specs],
                        'source': 'glaive'})
    return out


def grams(text, n=8):
    words = re.findall(r'\w+', text.lower())
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def tau_grams(tau_root):
    found = set()
    for path in Path(tau_root).glob('*/policy.md'):
        found |= grams(path.read_text())
    for path in Path(tau_root).glob('*/tasks.json'):
        found |= grams(path.read_text())
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', default='data/public_agent')
    parser.add_argument('--tau', default='.bench/tau2-repo/data/tau2/domains', help='τ²-bench domains (contamination)')
    parser.add_argument('--when2call', type=int, default=150)
    parser.add_argument('--toolace', type=int, default=200)
    parser.add_argument('--glaive', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rng, root = random.Random(args.seed), Path(args.root)
    rows = (when2call(root / 'when2call/train/when2call_train_sft.jsonl', rng, args.when2call)
            + toolace(root / 'toolace/data.json', rng, args.toolace)
            + glaive(root / 'glaive/glaive-function-calling-v2.json', rng, args.glaive))
    banned = tau_grams(args.tau)
    kept = [r for r in rows if not (grams(json.dumps(r['messages'], ensure_ascii=False)) & banned)]
    with open(args.output, 'w') as handle:
        for row in kept:
            handle.write(json.dumps({**row, 'seed': 0, 'success': True, 'public': True}, ensure_ascii=False) + '\n')
    import collections
    print(json.dumps({'kept': len(kept), 'contaminated': len(rows) - len(kept),
                      'by_source': dict(collections.Counter(r['source'] for r in kept))}))


if __name__ == '__main__':
    main()
