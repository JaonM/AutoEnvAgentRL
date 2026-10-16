"""Relabel the answer contracts of already-certified tasks so a concise teacher can succeed on them.

pool1/pool3 tasks grade a median of 8-13 fields (up to 35) and their agent policy ends with "in the final response,
report every completed part: ...", so the teacher wrote long reports and the student copied them (2026-10-16). Their
environments and verifiers are sound, so instead of generating new tasks each one is copied and relabelled:
  1. every graded field's expected value is computed on all served worlds; a field constant across them cannot be a
     deciding factor;
  2. a model picks the outcome and its deciding factors (checks.size_problems: at most 5 fields / 8 values; others
     become "graded": false) and rewrites the agent policy's final-report clause to list only those (the original
     clause must be quoted verbatim);
  3. the copy is re-certified with every gate (bundle.requalify); only copies that pass are used.

    python experiments/relabel_contracts.py --tasks A.json B.json --output output/relabel [--parallel 6] [--limit N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

PROMPT = """You trim the answer contract of a customer-service task to what decides its outcome.

The agent will tell the user the result in a sentence or two. Only the graded fields below are checked against what it
said; a write's effect is checked on the database separately. Keep as graded only the outcome and the few factors that
decide it: what happened (done or refused, the new status, the final amount the user pays or gets back) and the deciding
factor a wrong reading of the rules would change (why it was refused, which fee tier applied, the one date that matters).
Drop lookups, restated rules, intermediate numbers and confirmations of what did not change. A field whose value is the
same in every world ("constant") cannot be a deciding factor. With several goals, keep each goal's outcome.
Prefer a summary value (a total, a count, an overall status) over a per-item list when the summary decides what the
user gets; keep a per-item list only when the user needs each item's result separately.
At most 5 graded fields and 8 graded values in all (an object or array of objects counts each property).

User request: {request}
User goals: {goals}

Graded fields (name: description | distinct values across {worlds} worlds | example):
{fields}

Final-report clause of the agent policy (quote the part to replace verbatim; it may be absent):
{clause}

Answer with one JSON object:
{{"keep": ["<field>", ...],
  "clause": {{"original": "<verbatim text from the policy to replace, or empty>",
              "replacement": "<the same instruction listing only what the kept fields cover, briefly>"}}}}"""

CLAUSE = re.compile(r'(?is)([^\n]*\b(final (response|answer|reply)|最终(回复|答复|答案)|最后(告诉|回复))[^\n]*(\n\s*[-*•].*)*)')


def graded_values(spec):
    from env_factory.checks import graded_values as count
    return count(spec)


def field_variation(loaded):
    from env_factory.certify import bundle
    from env_factory.core import answers
    service = bundle.service(loaded, user_mode='structured')
    start = loaded['request']['time']['start']
    seen = {}
    for name in loaded['valid_worlds'] or service.order:
        data, facts = service.worlds[name]
        answer = loaded['verifier'].expected(copy.deepcopy(data), copy.deepcopy(facts), start)['answer']
        for field in answers.graded_fields(loaded['schema']):
            seen.setdefault(field, []).append(json.dumps(answer.get(field), sort_keys=True, ensure_ascii=False))
    return seen, len(loaded['valid_worlds'] or service.order)


def relabel(task, out_root, client):
    from env_factory.certify import bundle
    from env_factory.checks import size_problems
    source = Path(task['sandbox'])
    target = out_root / task['id'] / 'bundle'
    report = {'id': task['id'], 'source': str(source)}
    loaded = bundle.load_bundle(source)
    schema, request = loaded['schema'], loaded['request']
    if not size_problems(schema['fields']):
        report['status'] = 'already small'
    variation, worlds = field_variation(loaded)
    fields = '\n'.join(f"- {name}: {spec.get('description', '')[:200]} | {len(set(variation.get(name, [])))} | "
                       f"{(variation.get(name) or ['?'])[0][:80]}"
                       for name, spec in schema['fields'].items() if spec.get('graded', True))
    policy = request.get('agent_policy') or request['business_rules']
    found = CLAUSE.search(policy)
    feedback, choice = '', None
    for attempt in range(3):
        prompt = PROMPT.format(request=request.get('user_request', '')[:600],
                               goals=json.dumps((request.get('user_script') or {}).get('goals', []), ensure_ascii=False),
                               worlds=worlds, fields=fields, clause=found.group(1)[:3000] if found else '(none)') + feedback
        text = client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
        try:
            choice = json.loads(text[text.index('{'):text.rindex('}') + 1])
        except ValueError:
            feedback = '\n\nYour last answer was not one JSON object.'
            continue
        keep = [k for k in choice.get('keep', []) if k in schema['fields']]
        trial = {k: v for k, v in schema['fields'].items() if k in keep}
        problems = size_problems(trial) + ([] if keep else ['keep at least one field'])
        original = (choice.get('clause') or {}).get('original') or ''
        if original and original not in policy:
            if attempt == 2:  # last try: keep the trimmed fields, leave the policy text as it is
                choice['clause'] = {}
            else:
                problems.append('the clause "original" is not verbatim text of the policy; copy it exactly, or leave '
                                'it empty')
        if not problems:
            break
        feedback = f'\n\nYour last answer had problems: {"; ".join(problems)}. Answer again.'
        choice = None
    if choice is None:
        return {**report, 'status': 'no valid choice'}
    keep = [k for k in choice['keep'] if k in schema['fields']]
    if target.parent.exists():
        shutil.rmtree(target.parent)
    shutil.copytree(source, target)
    new_schema = copy.deepcopy(schema)
    for name, spec in new_schema['fields'].items():
        if name not in keep:
            spec['graded'] = False
    (target / 'answer_schema.json').write_text(json.dumps(new_schema, ensure_ascii=False, indent=1))
    clause = choice.get('clause') or {}
    new_request = copy.deepcopy(request)
    if clause.get('original') and clause.get('replacement'):
        key = 'agent_policy' if request.get('agent_policy') else 'business_rules'
        new_request[key] = policy.replace(clause['original'], clause['replacement'], 1)
    new_request['relabelled'] = {'from': str(source), 'kept': keep}
    (target / 'request.json').write_text(json.dumps(new_request, ensure_ascii=False, indent=1))
    manifest = json.loads((target / 'manifest.json').read_text())
    manifest['files'] = {name: bundle.sha256(target / name) for name in manifest['files']}
    manifest['relabelled_from'] = str(source)
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    for stale in ('policy-natural.json', 'docker_image.json'):
        (target / stale).unlink(missing_ok=True)
    gates = bundle.requalify(target)
    graded_before = sum(1 for s in schema['fields'].values() if s.get('graded', True))
    return {**report, 'status': 'passed' if gates['passed'] else 'gates failed', 'bundle': str(target),
            'graded_before': graded_before, 'kept': keep, 'clause_replaced': bool(clause.get('original')),
            'failed_gates': [g for g, r in gates.items() if isinstance(r, dict) and r.get('passed') is False]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    from env_factory.core.llm import ChatClient, load_env_file
    load_env_file(ROOT / '.env')
    client = ChatClient('LLM', timeout=180)
    tasks = [t for path in args.tasks for t in json.loads(Path(path).read_text())['tasks']]
    tasks = tasks[:args.limit] if args.limit else tasks
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in (out_root / 'report.jsonl').read_text().splitlines()} \
        if (out_root / 'report.jsonl').exists() else set()
    todo = [t for t in tasks if t['id'] not in done]
    print(f'{len(tasks)} tasks, {len(todo)} to relabel', flush=True)

    def one(task):
        try:
            row = relabel(task, out_root, client)
        except Exception as exc:  # noqa: BLE001 - one task never stops the rest
            row = {'id': task['id'], 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:300]}
        with open(out_root / 'report.jsonl', 'a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"{row['id']}: {row['status']} {row.get('graded_before', '')} -> {len(row.get('kept') or [])}", flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, todo))
    rows = [json.loads(line) for line in (out_root / 'report.jsonl').read_text().splitlines()]
    passed = [r for r in rows if r['status'] == 'passed']
    (out_root / 'tasks.json').write_text(json.dumps({'tasks': [
        {'id': r['id'] + 'r', 'sandbox': r['bundle'], 'split': 'train', 'weight': 1.0} for r in passed]}, indent=1))
    print(json.dumps({'tasks': len(rows), 'passed': len(passed),
                      'statuses': {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}}}))


if __name__ == '__main__':
    main()
