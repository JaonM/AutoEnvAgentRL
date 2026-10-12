"""Task generation front end: new build requests for new business domains, in two stages.

Stage one: each task comes from its own codex session in an isolated folder
(prompts.REQUEST_AUTHOR), which writes rules, tables, user facts and a user scene (situation,
goal, what the user knows up front, what they want to be told), but no opening. It must pass
`checks request`: a valid request, a scene without rules, protocol or virtual time, and boolean
approve_* facts. Stage two: another model (prompts.OPENING_WRITER) writes the user's first message
from the scene and a persona only, so it cannot restate rules it never saw. The opening must have
no red flag and must say every number in the scene's known information. A third step rewrites
the rules as the agent-facing policy manual (policies.write; request["agent_policy"]). Near-duplicates (character 3-gram Jaccard similarity
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

from env_factory_v3 import checks, openings, policies, prompts, spec

MIX = (('direct_response', 'direct', 0.2), ('simple_agentic', 'simple', 0.2),
       ('multi_step_agentic', 'ordinary', 0.2), ('multi_step_agentic', 'complex', 0.4))
SHAPES = {
    'direct': 'Every number the answer needs is in the opening (as the user would quote a price list or a '
              'receipt); no tables or tools are needed; one exact numeric or enum answer.',
    'simple': 'One lookup through read tools returns the records the answer needs (for example the order and '
              'its items); the agent applies the rules to decide (for example eligibility); the user knows the '
              'ids; no writes.',
    'ordinary': 'Several reads whose results feed each other (search, filter, compute); no writes.',
    'complex': 'A real multi-step job, not a short one with many rules. The shortest correct solution needs '
               'at least 6 public tool calls (reads and writes; asking the user or asking for approval does not '
               'count) and at least two writes, each needing the user\'s approval (facts approve_first / '
               'approve_second ...). Build in: decoy rows a careless agent would pick; one step whose obvious '
               'first choice is refused or ineligible (a read reveals it, or a write is rejected with a business '
               'error), so the agent must find another route; and a dependency between writes (a shared budget, '
               'capacity or ordering), so the user refusing one approval changes what a later step must do.',
    'subjective': 'A subjective request with no single correct reply: the user wants advice, a comparison '
                  'with a recommendation, an explanation of a refusal or a charge, or help with a complaint. '
                  'The assistant still needs the tools for the facts (two or more reads; a write only if the '
                  'user asks for one). Graded by a rubric, so set "reward_type": "rubric". The rules state '
                  'policies, not a reply checklist: at most 3 things the reply must cover, so there is room '
                  'for judgement in how to advise and explain.',
}
SIMILAR = 0.6
MAX_DIRECT_NUMBERS, MAX_WANTS = 8, 3  # what a user says in one opening message
# Difficulty floor: public tool calls of the built reference solution, checked after the build (run.py).
MIN_REFERENCE_CALLS = {'complex': 6}


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
    scene = request.get('user_scene')
    scene_problems = scene_issues(scene)
    if scene_problems:
        return scene_problems, []
    # The opening is written later from the scene alone; validate the rest with a stand-in.
    problems = spec.validate({**request, 'user_request': request.get('user_request') or scene['goal']})
    if problems:
        return problems, []
    problems += [f'user_scene {p}' for p in opening_problems(json.dumps(scene, ensure_ascii=False))
                 if '金额用' not in p]
    facts = request['user_facts']
    approvals = [k for k in facts if k.startswith('approve')]
    problems += [f'approval fact {k} must be a boolean' for k in approvals if not isinstance(facts[k], bool)]
    expected = json.loads((Path(folder) / 'assignment.json').read_text()) if (Path(folder) / 'assignment.json').exists() else {}
    if expected.get('category') and request['category'] != expected['category']:
        problems.append(f'category must be {expected["category"]}')
    wanted = 'rubric' if expected.get('shape') == 'subjective' else 'business'
    if expected and request.get('reward_type', 'business') != wanted:
        problems.append(f'reward_type must be "{wanted}" for this task')
    if expected.get('shape') == 'complex' and len(approvals) < 2:
        problems.append('a complex task needs at least two writes, each with its own approve_* fact')
    if expected.get('shape') in ('direct',) and any(request['tables'].values()):
        problems.append('a direct task needs no table rows')
    if expected.get('shape') == 'direct' and not scene_numbers(scene):
        problems.append('a direct task needs every number of its computation in user_scene.known')
    if expected.get('shape') == 'direct' and len(scene_numbers(scene)) > MAX_DIRECT_NUMBERS:
        problems.append(f'user_scene.known has {len(scene_numbers(scene))} numbers; a user says at most '
                        f'{MAX_DIRECT_NUMBERS} in one breath: simplify the computation')
    if len(scene.get('wants_to_know', [])) > MAX_WANTS:
        problems.append(f'user_scene.wants_to_know has more than {MAX_WANTS} items; put reporting duties in '
                        'business_rules')
    return problems, [f"{request['category']}: {request['title']}"]


def scene_issues(scene):
    if not isinstance(scene, dict):
        return ['user_scene is missing: {"situation", "goal", "known", "wants_to_know"}']
    problems = [f'user_scene.{key} must be nonempty text' for key in ('situation', 'goal')
                if not (isinstance(scene.get(key), str) and scene[key].strip())]
    if not isinstance(scene.get('known'), dict):
        problems.append('user_scene.known must be an object')
    if not isinstance(scene.get('wants_to_know'), list):
        problems.append('user_scene.wants_to_know must be a list')
    return problems


def scene_numbers(scene):
    """Every number in the scene's known information (the opening must say each of them)."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            found.append(value)
    walk(scene.get('known', {}))
    return found


DIGITS = '零一二三四五六七八九'


def _under_10k(n):
    parts, zero = [], False
    for value, unit in ((1000, '千'), (100, '百'), (10, '十'), (1, '')):
        digit = n // value % 10
        if digit:
            parts.append(('零' if zero else '') + DIGITS[digit] + unit)
            zero = False
        elif parts:
            zero = True
    return ''.join(parts)


def chinese(n):
    """Chinese readings of a non-negative integer below 10^8: 12 -> 十二, 2 -> 二/两, 180 -> 一百八十/一百八."""
    if not 0 <= n < 10 ** 8:
        return set()
    if n == 0:
        return {'零'}
    high, low = divmod(n, 10000)
    text = (_under_10k(high) + '万' + ('零' if 0 < low < 1000 else '') if high else '') + _under_10k(low)
    if text.startswith('一十'):
        text = text[1:]
    forms = {text}
    if re.search(r'[一二三四五六七八九][百千万][一二三四五六七八九][十百千]$', text):
        forms.add(text[:-1])                       # 一百八十 -> 一百八, 一千五百 -> 一千五
    forms |= {'两' + f[1:] for f in forms if f.startswith('二') and (n < 10 or not f.startswith('二十'))}
    return forms


def spoken(number):
    """Ways a user may say a number: as is, in Chinese numerals (两, 十二), or (cents) in 元 such as
    1.8 / 1块8 / 一块八."""
    forms = {f'{number:g}'}
    if isinstance(number, int) and not isinstance(number, bool):
        forms |= chinese(number)
        yuan = number / 100
        forms.add(f'{yuan:g}')
        whole, cents = divmod(number, 100)
        if not cents:
            forms |= chinese(whole)
        elif cents % 10 == 0:
            forms |= {f'{w}块{DIGITS[cents // 10]}' for w in chinese(whole)} | {f'{whole}块{cents // 10}'}
    return forms


def missing_numbers(opening, scene):
    return [n for n in scene_numbers(scene) if not any(form in opening for form in spoken(n))]


def write_opening(client, request, persona_key, attempts=3, avoid=()):
    """Stage two: the user's first message from the scene alone (never the rules, tables or facts)."""
    scene = request['user_scene']
    name, hint = openings.persona(persona_key)
    feedback = ''
    avoid_text = f"不要使用这些已被其他用户用滥的说法：{'、'.join(avoid)}。" if avoid else ''
    for _ in range(attempts):
        text = (client.chat([{'role': 'user', 'content': prompts.OPENING_WRITER.format(
            situation=scene['situation'], goal=scene['goal'], known=json.dumps(scene['known'], ensure_ascii=False),
            wants='；'.join(map(str, scene['wants_to_know'])) or '（无）', persona=name, persona_hint=hint,
            feedback=feedback, avoid=avoid_text)}], temperature=0.8).get('content') or '').strip()
        flags, missing = openings.red_flags(text), missing_numbers(text, scene)
        flags += [f'用滥的说法：{phrase}' for phrase in avoid if phrase in text]
        if text and not flags and not missing:
            return text, name, []
        feedback = f'上一条：{text}\n问题：禁用说法 {flags}；漏说的数字 {missing}。请重写。\n'
    return None, name, [f'opening: red flags {flags}, missing numbers {missing}']


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


def repeated_phrases(texts, size=6, share=0.3):
    """Phrases of `size`+ characters that more than `share` of the openings (and at least 3) use."""
    counts = {}
    for text in texts:
        seen = {text[i:i + size] for i in range(len(text) - size + 1)}
        for gram in seen:
            counts[gram] = counts.get(gram, 0) + 1
    limit = max(3, int(share * len(texts)) + 1)
    return sorted(gram for gram, n in counts.items() if n >= limit and not any(c in gram for c in '0123456789'))


def respread(kept, client, output):
    """Rewrite openings that share stock phrases with too many others (once)."""
    phrases = repeated_phrases([r['user_request'] for r in kept])
    rewritten = []
    for request in kept:
        hits = [p for p in phrases if p in request['user_request']]
        if not hits:
            continue
        text, _, _ = write_opening(client, request, request['id'], avoid=tuple(hits))
        if text:
            request['user_request'] = text
            (output / 'requests' / f"{request['id']}.json").write_text(json.dumps(request, ensure_ascii=False, indent=1))
            rewritten.append(request['id'])
    return rewritten


def generate(domains, count, output, *, runner, parallel=2, timeout=600, python='python', rubric_share=0.0,
             opening_client=None):
    """Stage one: a codex session per task writes rules, tables, facts and a user scene. Stage two:
    opening_client (V3_OPENING_*, falling back to LLM_*) writes the opening from the scene alone."""
    if opening_client is None:
        from env_factory_v3.llm import ChatClient, load_env_file
        load_env_file()
        opening_client = ChatClient('V3_OPENING')
    output = Path(output)
    output.mkdir(parents=True)
    assignments = plan(domains, count, rubric_share)

    def one(index, assignment):
        domain, category, shape = assignment
        folder = output / 'sessions' / f'{index:03d}'
        folder.mkdir(parents=True)
        (folder / 'assignment.json').write_text(json.dumps(
            {'domain': domain, 'category': category, 'shape': shape, 'id': f'gen-{index:03d}'}, ensure_ascii=False))
        detail = domain.get('detail', '')
        if domain.get('process') or domain.get('mechanisms'):
            detail += f" 业务环节：{domain.get('process', '')}；需要体现的机制：{'、'.join(domain.get('mechanisms', []))}。"
        runner(folder, prompts.REQUEST_AUTHOR.format(python=python, domain=domain['name'],
                                                       detail=detail, category=category,
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
            opening, persona_name, opening_problems_ = write_opening(opening_client, request, f'gen-{index:03d}')
            row.update(persona=persona_name)
            if opening is None:
                row.update(passed=False, problems=opening_problems_)
                rows.append(row)
                continue
            request['user_request'] = opening
            policy, policy_problems = policies.write(opening_client, request)
            if policy is None:
                row.update(passed=False, problems=[f'agent policy: {p}' for p in policy_problems[:4]])
                rows.append(row)
                continue
            request['agent_policy'] = policy
            twin = next((k for k in kept if similarity(signature(k), signature(request)) >= SIMILAR), None)
            if twin is not None:
                row.update(passed=False, problems=[f'near-duplicate of {twin["id"]}'])
            else:
                request['id'] = f'gen-{index:03d}'
                request['source'] = {'generated': domain['name'], 'family': domain.get('family'),
                                     'shape': shape, 'persona': persona_name,
                                     'opening_writer': getattr(opening_client, 'model', None)}
                kept.append(request)
                (output / 'requests').mkdir(exist_ok=True)
                (output / 'requests' / f'{request["id"]}.json').write_text(
                    json.dumps(request, ensure_ascii=False, indent=1))
        rows.append(row)
    rewritten = respread(kept, opening_client, output)
    summary = {'requested': count, 'kept': len(kept), 'domains': [d['name'] for d in domains],
               'openings_rewritten_for_repetition': rewritten,
               'opening_diversity': openings.diversity([k['user_request'] for k in kept]), 'rows': rows}
    (output / 'generation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary
