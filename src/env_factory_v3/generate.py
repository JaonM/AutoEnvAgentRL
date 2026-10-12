"""Task generation front end: new build requests for new business domains.

Each request comes from its own model session in an isolated folder (prompts.REQUEST_AUTHOR)
and must pass `checks request`: a valid request, an opening without red flags, and approval
facts that are booleans named approve_*. Near-duplicates (character 3-gram Jaccard similarity
of title, opening and rules) are dropped. The category mix follows V2's skeleton by default:
direct 20%, simple 20%, ordinary multi-step 20%, complex multi-step 40%. With rubric_share > 0 that
fraction of the multi-step slots become subjective tasks (reward_type "rubric").

    python -m env_factory_v3 generate --domains domains.json --count 30 --output DIR
"""
from __future__ import annotations

import concurrent.futures
import json
import re
from pathlib import Path

from env_factory_v3 import checks, openings, prompts, spec

MIX = (('direct_response', 'direct', 0.2), ('simple_agentic', 'simple', 0.2),
       ('multi_step_agentic', 'ordinary', 0.2), ('multi_step_agentic', 'complex', 0.4))
SHAPES = {
    'direct': 'Every number the answer needs is in the opening (as the user would quote a price list or a '
              'receipt); no tables or tools are needed; one exact numeric or enum answer.',
    'simple': 'One lookup through read tools answers it (for example an eligibility check); the user knows '
              'the ids; no writes.',
    'ordinary': 'Several reads whose results feed each other (search, filter, compute); no writes.',
    'complex': 'Two or more steps with at least one write that needs the user\'s approval (facts '
               'approve_first / approve_second ...); independent approvals, partial success, budgets or '
               'capacities, so refusing or failing one step matters.',
    'subjective': 'A subjective request with no single correct reply: the user wants advice, a comparison '
                  'with a recommendation, an explanation of a refusal or a charge, or help with a complaint. '
                  'The assistant still needs the tools for the facts (two or more reads; a write only if the '
                  'user asks for one). Graded by a rubric, so set "reward_type": "rubric".',
}
SIMILAR = 0.6


def opening_problems(text):
    return [f'user_request reads unnatural: {flag}' for flag in openings.red_flags(text)]


def check_request(folder):
    """Self-check for a generated request.json (run by the request author session)."""
    path = Path(folder) / 'request.json'
    if not path.exists():
        return ['request.json is missing'], []
    try:
        request = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f'request.json is not JSON: {exc}'], []
    request.setdefault('time', {'start': 10, 'tick': 0})
    problems = spec.validate(request)
    if problems:
        return problems, []
    problems += opening_problems(request['user_request'])
    facts = request['user_facts']
    approvals = [k for k in facts if k.startswith('approve')]
    problems += [f'approval fact {k} must be a boolean' for k in approvals if not isinstance(facts[k], bool)]
    expected = json.loads((Path(folder) / 'assignment.json').read_text()) if (Path(folder) / 'assignment.json').exists() else {}
    if expected.get('category') and request['category'] != expected['category']:
        problems.append(f'category must be {expected["category"]}')
    wanted = 'rubric' if expected.get('shape') == 'subjective' else 'business'
    if expected and request.get('reward_type', 'business') != wanted:
        problems.append(f'reward_type must be "{wanted}" for this task')
    if expected.get('shape') == 'complex' and not approvals:
        problems.append('a complex task needs at least one approve_* fact for a write')
    if expected.get('shape') in ('direct',) and any(request['tables'].values()):
        problems.append('a direct task needs no table rows')
    return problems, [f"{request['category']}: {request['title']}"]


def grams(text):
    text = re.sub(r'\s+', '', text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def similarity(a, b):
    left, right = grams(a), grams(b)
    return len(left & right) / max(1, len(left | right))


def signature(request):
    return request['title'] + request['user_request'] + request['business_rules']


def plan(domains, count, rubric_share=0.0):
    """(domain, category, shape) assignments following MIX, domains round-robin per shape.
    rubric_share turns that fraction of the multi-step slots (from the end) into subjective tasks."""
    slots = []
    for category, shape, share in MIX:
        slots += [(category, shape)] * round(count * share)
    slots = (slots + [MIX[-1][:2]] * count)[:count]
    subjective = round(count * rubric_share)
    for index in range(len(slots) - 1, -1, -1):
        if subjective and slots[index][0] == 'multi_step_agentic':
            slots[index] = ('multi_step_agentic', 'subjective')
            subjective -= 1
    return [(domains[i % len(domains)], category, shape) for i, (category, shape) in enumerate(slots)]


def generate(domains, count, output, *, runner, parallel=2, timeout=600, python='python', rubric_share=0.0):
    output = Path(output)
    output.mkdir(parents=True)
    assignments = plan(domains, count, rubric_share)

    def one(index, assignment):
        domain, category, shape = assignment
        folder = output / 'sessions' / f'{index:03d}'
        folder.mkdir(parents=True)
        (folder / 'assignment.json').write_text(json.dumps(
            {'domain': domain, 'category': category, 'shape': shape, 'id': f'gen-{index:03d}'}, ensure_ascii=False))
        name, hint = openings.persona(f'gen-{index:03d}')
        runner(folder, prompts.REQUEST_AUTHOR.format(python=python, domain=domain['name'], persona=name,
                                                       persona_hint=hint,
                                                       detail=domain.get('detail', ''), category=category,
                                                       shape=SHAPES[shape], id=f'gen-{index:03d}',
                                                       reward_type='rubric' if shape == 'subjective' else 'business'),
               timeout=timeout, role='request_author')
        report = checks.run('request', folder)
        return index, assignment, report

    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda pair: one(*pair), enumerate(assignments)))
    kept, rows = [], []
    for index, (domain, category, shape), report in results:
        row = {'index': index, 'domain': domain['name'], 'category': category, 'shape': shape,
               'passed': report['passed'], 'problems': report['problems'][:5]}
        if report['passed']:
            request = json.loads((output / 'sessions' / f'{index:03d}' / 'request.json').read_text())
            request.setdefault('time', {'start': 10, 'tick': 0})
            twin = next((k for k in kept if similarity(signature(k), signature(request)) >= SIMILAR), None)
            if twin is not None:
                row.update(passed=False, problems=[f'near-duplicate of {twin["id"]}'])
            else:
                request['id'] = f'gen-{index:03d}'
                request['source'] = {'generated': domain['name'], 'shape': shape}
                kept.append(request)
                (output / 'requests').mkdir(exist_ok=True)
                (output / 'requests' / f'{request["id"]}.json').write_text(
                    json.dumps(request, ensure_ascii=False, indent=1))
        rows.append(row)
    summary = {'requested': count, 'kept': len(kept), 'domains': [d['name'] for d in domains], 'rows': rows}
    (output / 'generation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary
