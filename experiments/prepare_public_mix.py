"""Select 1000 distinct local public agent episodes for the 1:1 world-data control.

No model calls. Public examples are format-checked, not executed in V3 sandboxes.
The benchmark is used solely by a local exclusion filter, never to author data.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re

import jsonschema
from public_data import when2call, toolace, glaive, grams, tau_grams
from sft_train import render_episode


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(row):
    messages = [{k: v for k, v in m.items() if k not in ('tool_call_id', 'tool_calls')}
                for m in row['messages']]
    for source, dest in zip(row['messages'], messages):
        if source.get('tool_calls'):
            dest['calls'] = [{'name': c['function']['name'],
                             'arguments': json.loads(c['function']['arguments'])} for c in source['tool_calls']]
    content = {'messages': messages, 'tools': sorted(row['tools'], key=lambda t: t['function']['name'])}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(row):
    tools = {}
    for tool in row['tools']:
        f = tool['function']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', f['name']) or f['name'] in tools:
            raise ValueError('invalid/duplicate function name')
        jsonschema.Draft7Validator.check_schema(f['parameters'])
        tools[f['name']] = f['parameters']
    if not tools or not row['messages'] or row['messages'][-1]['role'] != 'assistant':
        raise ValueError('missing tools or final assistant response')
    pending, seen = {}, set()
    for message in row['messages']:
        role = message['role']
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('unknown message role')
        if role != 'tool' and pending:
            raise ValueError('missing tool results')
        if role == 'tool':
            key = message.get('tool_call_id')
            if key not in pending or message.get('name') != pending[key]:
                raise ValueError('orphan/mismatched tool result')
            del pending[key]
        for call in message.get('tool_calls') or []:
            name, key = call['function']['name'], call['id']
            if role != 'assistant' or name not in tools or key in seen:
                raise ValueError('invalid tool call')
            args = json.loads(call['function']['arguments'])
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            jsonschema.Draft7Validator(tools[name]).validate(args)
            pending[key] = name
            seen.add(key)
    if pending:
        raise ValueError('unresolved tool calls')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/public_agent'))
    parser.add_argument('--tau', type=Path, default=Path('.bench/tau2-repo/data/tau2/domains'))
    parser.add_argument('--model', default='models/Qwen3-0.6B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    specs = [('when2call', when2call, 'when2call/train/when2call_train_sft.jsonl', 300),
             ('toolace', toolace, 'toolace/data.json', 400),
             ('glaive', glaive, 'glaive/glaive-function-calling-v2.json', 300)]
    exclusions = sorted([*args.tau.glob('*/policy.md'), *args.tau.glob('*/tasks.json')])
    if not any(p.name == 'tasks.json' for p in exclusions):
        raise ValueError('benchmark exclusion corpus unavailable')
    banned = tau_grams(args.tau)
    selected, seen, report, source_hashes = [], set(), {}, {}
    for source, convert, filename, target in specs:
        path = args.root / filename
        source_hashes[str(path)] = sha(path)
        candidates = convert(path, random.Random(f'{args.seed}:{source}'), 12000)
        counts = collections.Counter()
        for candidate in candidates:
            if counts['selected'] == target:
                break
            counts['examined'] += 1
            try:
                validate(candidate)
            except (ValueError, KeyError, TypeError, jsonschema.ValidationError, jsonschema.SchemaError):
                counts['invalid_structure_or_arguments'] += 1
                continue
            if grams(json.dumps({'messages': candidate['messages'], 'tools': candidate['tools']}, ensure_ascii=False)) & banned:
                counts['benchmark_overlap'] += 1
                continue
            key = signature(candidate)
            if key in seen:
                counts['duplicate'] += 1
                continue
            rendered = render_episode(tokenizer, candidate, skip_failed=True)
            if len(rendered) != 1 or len(rendered[0][0]) > 32768 or not sum(rendered[0][1]):
                counts['untrainable'] += 1
                continue
            systems = [m['content'] for m in candidate['messages'] if m['role'] == 'system']
            text = tokenizer.decode(rendered[0][0])
            if any(system not in text for system in systems):
                raise ValueError('original system prompt changed during training rendering')
            seen.add(key)
            selected.append({**candidate, 'id': f'public:{source}:{key}', 'source_conversion_id': candidate['id'],
                             'seed': args.seed, 'public': True, 'success': True,
                             'qualification': 'public_format_and_token_checks_only', 'content_sha256': key})
            counts['selected'] += 1
            counts['with_original_system'] += bool(systems)
        report[source] = dict(counts)
        if counts['selected'] != target:
            raise ValueError(f'insufficient qualified {source} episodes: {counts}')
    target = args.output / 'public_train.jsonl'
    target.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in selected))
    manifest = {'samples': len(selected), 'ratio_basis': 'full dialogue samples, not tokens',
                'system_prompt_policy': 'Preserve original system messages verbatim; Qwen appends its standard tool specification.',
                'world_train_samples': 1000, 'public_train_samples': 1000, 'source_counts': report,
                'seed': args.seed, 'source_hashes': source_hashes, 'public_sha256': sha(target),
                'code_hashes': {str(p): sha(p) for p in (Path(__file__), Path('experiments/public_data.py'))},
                'exclusion_hashes': {str(p): sha(p) for p in exclusions},
                'limits': ['8-gram exclusion is a conservative overlap filter, not proof of no contamination.',
                           'Public tool results are dataset demonstrations, not independently executed V3 successes.',
                           'Inherited converter admits at most three calls per assistant turn.',
                           'Exact normalized dialogue/tool duplicates are excluded; semantic paraphrases can remain.']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'samples': len(selected), 'source_counts': report}, ensure_ascii=False))


if __name__ == '__main__':
    main()
