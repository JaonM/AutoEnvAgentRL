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

    python -m env_factory generate --domains domains.json --count 30 --output DIR
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
from pathlib import Path

from env_factory.authoring.codex import InfrastructureError
from env_factory import checks
from env_factory.authoring import prompts
from env_factory.core import spec
from env_factory.core.numerals import chinese, spoken  # noqa: F401  (chinese: re-exported for callers)
from env_factory.production import openings, agent_policy as policies

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
# User script elements (docs/domain-world-design.md §3): each interactive task gets one to three,
# drawn with these probabilities; direct tasks get none.
SCRIPT_ELEMENTS = {'multi_goal': 0.5, 'fallback': 0.35, 'inquisitive': 0.35, 'pushback': 0.3, 'wrong_first': 0.25,
                   'change_of_mind': 0.2, 'ambiguous': 0.2, 'clarify': 0.2, 'forgotten': 0.2, 'out_of_scope': 0.1}
MIN_ELEMENTS, MAX_ELEMENTS = 2, 4  # interactive tasks: richer scripts make longer, more realistic dialogues
# Appended to every interactive task's user_script.traits: customers answer what they are asked, one
# piece at a time, so the agent has to elicit the details (dialogues in the first pools ended after a
# median of 3 user replies).
DISCLOSURE = {'zh': '只回答对方明确问到的那一项信息，一次只说一项，不主动补充其他细节',
              'en': 'answers only exactly what is asked, one piece of information at a time, never volunteering other details'}
# Customer identification is a norm of customer service, not a script element: most interactive
# tasks make the agent ask who the customer is before account-specific lookups or changes.
IDENTITY_SHARE = 0.7
SCRIPT_TEXT = {
    'fallback': ('Fallback: the user wants option A and accepts option B (or C) only if A is impossible. Make A '
                 'impossible in the data (out of stock, ineligible, full). Give each option its own boolean approve_* '
                 'fact (true for the acceptable fallback, false for options the user would refuse) and make '
                 'env.approval_key map a proposal to the option it names. user_script.fallbacks lists the options in '
                 'preference order, in the user\'s words.'),
    'pushback': ('Pushback: the user first demands something the business rules do not allow (an extra refund, a fee '
                 'waiver, an exception). The correct outcome does not grant it. user_script.pushback = {"demand": '
                 '"<in the user\'s words>", "rounds": 1-3, "then": "accept"}: after being refused the user insists '
                 'that many times, then accepts.'),
    'multi_goal': ('Multiple goals: the user has two or three related requests in one conversation; user_script.goals '
                   'lists them in order (later ones are raised after the first is handled); the expected outcome '
                   'covers all of them.'),
    'out_of_scope': ('Out of scope: besides its real request, the user asks for something this business does not '
                     'handle at all (not covered by the rules, unlike a refused demand). The correct outcome changes '
                     'nothing for it; the rules say how such requests are answered (decline, point elsewhere, or a '
                     'handover the domain itself provides). user_script.out_of_scope = "<the request in the user\'s '
                     'words>", also listed in user_script.goals.'),
    'inquisitive': ('Inquisitive: before agreeing to any change, the user asks one or two questions about its '
                    'consequences (a fee, what happens to the old booking, when the refund arrives) that the rules '
                    'answer; they decide only after hearing the answers. Add a trait saying so in the user\'s words '
                    'to user_script.traits and set user_script.inquisitive = ["<question>", ...].'),
    'change_of_mind': ('Change of mind: the user first asks for option A, and after hearing a condition the rules '
                       'impose on it (a fee, a delay, a restriction) switches to option B. Give each option its own '
                       'boolean approve_* fact (A false, B true) and make env.approval_key map a proposal to the option '
                       'it names. Set user_script.change_of_mind = {"from": "<A>", "to": "<B>", "after": "<what makes '
                       'them switch>"} and add a trait describing the switch in the user\'s words.'),
    'wrong_first': ('Wrong first: the user misremembers one identifier the agent must ask for (an order id, a '
                    'booking reference, an email) and first gives a wrong value that exists nowhere in the data, so '
                    'the lookup fails; asked again, the user gives the right one. Keep the right value in user_facts '
                    '(not in user_scene.known) and set user_script.wrong_first = {"<that fact key>": "<the wrong '
                    'value, plausible: one character or digit off>"}. The correct outcome is unchanged.'),
    'ambiguous': ('Ambiguous request: the user refers vaguely to something ("change that order", "my booking") '
                  'while several records fit (make at least two matching rows for this customer in the data). The '
                  'one they mean is a user fact the opening does not reveal; the agent must ask which before acting. '
                  'Set user_script.ambiguous = {"phrase": "<the vague words>", "fact": "<the fact key that '
                  'identifies the intended record>"}.'),
    'clarify': ('Unclear intent: the user mixes two or three possible actions without choosing ("maybe change my '
                'flight, or cancel it, depending on the cost"). Which one they want is decided by information the '
                'agent has to give first (a fee, an availability). Give each action its own boolean approve_* fact '
                '(true only for the one the user would choose given the real data) and make env.approval_key map a '
                'proposal to the action it names. Set user_script.clarify = {"options": ["<action in the user\'s '
                'words>", ...], "decides_by": "<what the user needs to hear before choosing>"}. The user only chooses '
                'after the agent explains the options; the agent must not act before asking.'),
    'identity': ('Identify the customer: the business rules require the agent to identify the customer (an account '
                 'id, email, phone or name plus postcode) before any account-specific lookup or change, and the tools '
                 'need that identifier. Put the identifier in user_facts, not in user_scene.known, so the opening '
                 'does not volunteer it and the agent has to ask.'),
    'forgotten': ('Forgotten information: the user does not remember one identifier an agent would normally ask for '
                  '(an order id, an email). Leave it out of user_facts and user_scene.known, list its name in '
                  'user_script.forgotten, and give the user another way to be identified or found (for example name '
                  'plus postcode) that the business rules and tools support.'),
}
SIMILAR = 0.6
MAX_DIRECT_NUMBERS, MAX_WANTS = 8, 3  # what a user says in one opening message
TASK_TIMEOUT = 900  # seconds to write one opening and policy before giving up on a stalled connection
# Difficulty floor: public tool calls of the built reference solution, checked after the build (production/run.py).
MIN_REFERENCE_CALLS = {'complex': 6}


LANGUAGE_NAMES = {'zh': 'Chinese', 'en': 'English'}
CURRENCY = {'zh': '元', 'en': 'dollars'}


def languages(count, english_share=0.0):
    """Language per task index: english_share of the tasks in English, spread evenly over the plan."""
    english = round(count * english_share)
    return ['en' if english and (i * english) // count != ((i + 1) * english) // count else 'zh'
            for i in range(count)]


def scripts(count, shapes, seed=0):
    """Script elements per task index: one to three for interactive tasks, none for direct ones."""
    import random
    out = []
    for index, shape in enumerate(shapes):
        if shape == 'direct':
            out.append([])
            continue
        rng = random.Random(f'{seed}-script-{index}')
        chosen = [name for name, share in SCRIPT_ELEMENTS.items() if rng.random() < share][:MAX_ELEMENTS]
        while len(chosen) < MIN_ELEMENTS:
            extra = rng.choices(list(SCRIPT_ELEMENTS), weights=list(SCRIPT_ELEMENTS.values()))[0]
            if extra not in chosen:
                chosen.append(extra)
        if rng.random() < IDENTITY_SHARE or {'wrong_first', 'forgotten'} & set(chosen):
            chosen.append('identity')  # besides the (up to three) script elements
        out.append(chosen)
    return out


def script_prompt(elements):
    if not elements:
        return 'User script: none for this task; omit user_script.'
    return ('User script (write "user_script" in request.json): {"traits": ["<one or two traits of the user>"], ...} '
            'plus these elements:\n' + '\n'.join(f'- {SCRIPT_TEXT[e]}' for e in elements))


def script_problems(request, elements):
    """The user_script promised by the assignment, consistent with the facts."""
    if not elements:
        return []
    script = request.get('user_script')
    if not isinstance(script, dict):
        return ['user_script is missing (this task has script elements: ' + ', '.join(elements) + ')']
    problems, facts = [], request['user_facts']
    if not (isinstance(script.get('traits'), list) and script['traits']):
        problems.append('user_script.traits must list one or two traits')
    approvals = [k for k in facts if k.startswith('approve')]
    if 'fallback' in elements and not (isinstance(script.get('fallbacks'), list) and len(script['fallbacks']) >= 2
                                       and len(approvals) >= 2):
        problems.append('fallback: user_script.fallbacks needs two or more options and each its own approve_* fact')
    push = script.get('pushback')
    if 'pushback' in elements and not (isinstance(push, dict) and isinstance(push.get('demand'), str) and push['demand']
                                       and push.get('rounds') in (1, 2, 3) and push.get('then') == 'accept'):
        problems.append('pushback: user_script.pushback must be {"demand": str, "rounds": 1-3, "then": "accept"}')
    if 'multi_goal' in elements and not (isinstance(script.get('goals'), list) and len(script['goals']) >= 2):
        problems.append('multi_goal: user_script.goals needs two or three goals in order')
    scope = script.get('out_of_scope')
    if 'out_of_scope' in elements and not (isinstance(scope, str) and scope.strip()
                                           and scope in (script.get('goals') or [])):
        problems.append('out_of_scope: user_script.out_of_scope must be the request, also listed in user_script.goals')
    wrong = script.get('wrong_first')
    if 'wrong_first' in elements:
        if not (isinstance(wrong, dict) and len(wrong) == 1):
            problems.append('wrong_first: user_script.wrong_first must be {"<fact key>": "<wrong value>"}')
        else:
            (key, value), = wrong.items()
            text = json.dumps(request.get('tables'), ensure_ascii=False)
            if key not in facts:
                problems.append(f'wrong_first: {key} must be a user fact (the right value)')
            elif not isinstance(value, str) or not value or value == facts[key] or f'"{value}"' in text:
                problems.append('wrong_first: the wrong value must differ from the fact and exist nowhere in the tables')
    if 'inquisitive' in elements and not (isinstance(script.get('inquisitive'), list) and script['inquisitive']):
        problems.append('inquisitive: user_script.inquisitive must list the questions the user asks before agreeing')
    shift = script.get('change_of_mind')
    if 'change_of_mind' in elements and not (isinstance(shift, dict) and shift.get('from') and shift.get('to')
                                             and shift.get('after') and len(approvals) >= 2):
        problems.append('change_of_mind: user_script.change_of_mind needs {"from", "to", "after"} and an approve_* '
                        'fact per option')
    if elements and 'direct' not in elements:
        known = (request.get('user_scene') or {}).get('known') or {}
        if isinstance(known, dict) and len(known) > 2:
            problems.append(f'user_scene.known has {len(known)} items; an interactive task opens with at most 2')
    amb = script.get('ambiguous')
    if 'ambiguous' in elements:
        known = json.dumps((request.get('user_scene') or {}).get('known') or {}, ensure_ascii=False)
        if not (isinstance(amb, dict) and isinstance(amb.get('phrase'), str) and amb['phrase'].strip()
                and amb.get('fact') in facts):
            problems.append('ambiguous: user_script.ambiguous must be {"phrase": str, "fact": <a user fact key>}')
        elif str(facts[amb['fact']]) in known:
            problems.append('ambiguous: the opening must not reveal which record the user means')
    clarify = script.get('clarify')
    if 'clarify' in elements and not (isinstance(clarify, dict) and isinstance(clarify.get('options'), list)
                                      and 2 <= len(clarify['options']) <= 3 and clarify.get('decides_by')
                                      and len(approvals) >= 2):
        problems.append('clarify: user_script.clarify needs {"options": 2-3 actions, "decides_by": str} and an '
                        'approve_* fact per action')
    if 'identity' in elements:
        known = json.dumps((request.get('user_scene') or {}).get('known') or {}, ensure_ascii=False)
        identifiers = [k for k, v in facts.items() if not k.startswith('approve') and isinstance(v, (str, int))
                       and str(v) not in known]
        if not identifiers:
            problems.append('identity: user_facts needs a customer identifier that user_scene.known does not reveal')
    forgotten = script.get('forgotten')
    if 'forgotten' in elements:
        known = (request.get('user_scene') or {}).get('known') or {}
        if not (isinstance(forgotten, list) and forgotten):
            problems.append('forgotten: user_script.forgotten must name the identifier the user does not remember')
        elif any(name in facts or name in known for name in forgotten):
            problems.append('forgotten: a forgotten identifier must be absent from user_facts and user_scene.known')
    return problems


def opening_problems(text, language='zh'):
    return [f'user_request reads unnatural: {flag}' for flag in openings.red_flags(text, language)]


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
    language = request.get('language', 'zh')
    problems += [f'user_scene {p}' for p in opening_problems(json.dumps(scene, ensure_ascii=False), language)
                 if '金额用' not in p]
    facts = request['user_facts']
    approvals = [k for k in facts if k.startswith('approve')]
    problems += [f'approval fact {k} must be a boolean' for k in approvals if not isinstance(facts[k], bool)]
    expected = json.loads((Path(folder) / 'assignment.json').read_text()) if (Path(folder) / 'assignment.json').exists() else {}
    if expected.get('language', 'zh') != language:
        problems.append(f'language must be "{expected.get("language", "zh")}" and every text field written in it')
    problems += script_problems(request, expected.get('script') or [])
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


def missing_numbers(opening, scene):
    return [n for n in scene_numbers(scene) if not any(form in opening for form in spoken(n))]


def write_opening(client, request, persona_key, attempts=3, avoid=()):
    """Stage two: the user's first message from the scene alone (never the rules, tables or facts)."""
    scene, language = request['user_scene'], request.get('language', 'zh')
    english = language == 'en'
    name, hint = openings.persona(persona_key, language)
    feedback = ''
    if english:
        avoid_text = f"do not use these overused phrases: {'; '.join(avoid)}." if avoid else ''
    else:
        avoid_text = f"不要使用这些已被其他用户用滥的说法：{'、'.join(avoid)}。" if avoid else ''
    template = prompts.OPENING_WRITER_EN if english else prompts.OPENING_WRITER
    for _ in range(attempts):
        text = (client.chat([{'role': 'user', 'content': template.format(
            situation=scene['situation'], goal=scene['goal'], known=json.dumps(scene['known'], ensure_ascii=False),
            wants=('; ' if english else '；').join(map(str, scene['wants_to_know'])) or ('(none)' if english else '（无）'),
            persona=name, persona_hint=hint, feedback=feedback, avoid=avoid_text)}],
            temperature=0.8).get('content') or '').strip()
        flags, missing = openings.red_flags(text, language), missing_numbers(text, scene)
        flags += [f'overused: {phrase}' for phrase in avoid if phrase in text]
        if text and not flags and not missing:
            return text, name, []
        feedback = (f'Previous: {text}\nProblems: phrases to avoid {flags}; numbers not mentioned {missing}. Rewrite.\n'
                    if english else f'上一条：{text}\n问题：禁用说法 {flags}；漏说的数字 {missing}。请重写。\n')
    return None, name, [f'opening: red flags {flags}, missing numbers {missing}']


def grams(text):
    text = re.sub(r'\s+', '', text)
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def similarity(a, b):
    left, right = grams(a), grams(b)
    return len(left & right) / max(1, len(left | right))


def signature(request):
    return request['title'] + request['user_request'] + request['business_rules']


def parse_mix(text):
    """'direct=0.1,simple=0.1,ordinary=0.2,complex=0.6' -> {shape: share} (shares sum to 1)."""
    mix = {name.strip(): float(share) for name, share in (part.split('=') for part in text.split(',') if part.strip())}
    known = {shape for _, shape, _ in MIX}
    if set(mix) - known or abs(sum(mix.values()) - 1) > 1e-6:
        raise ValueError(f'mix needs shares of {sorted(known)} summing to 1, got {mix}')
    return mix


def plan(domains, count, rubric_share=0.0, mix=None):
    """(domain, category, shape) assignments following MIX (or `mix`: {shape: share}), domains
    round-robin per shape. rubric_share turns that fraction of the multi-step slots (from the end)
    into subjective tasks."""
    table = MIX if mix is None else [(category, shape, mix.get(shape, 0.0)) for category, shape, _ in MIX]
    slots = []
    for category, shape, share in table:
        slots += [(category, shape)] * round(count * share)
    slots = (slots + [MIX[-1][:2]] * count)[:count]
    subjective = round(count * rubric_share)
    for index in range(len(slots) - 1, -1, -1):
        if subjective and slots[index][0] == 'multi_step_agentic':
            slots[index] = ('multi_step_agentic', 'subjective')
            subjective -= 1
    return [(domains[i % len(domains)], category, shape) for i, (category, shape) in enumerate(slots)]


def repeated_phrases(texts, size=6, share=0.3, language='zh'):
    """Phrases that more than `share` of the openings (and at least 3) use: `size`+ characters in
    Chinese, `size` // 2 + 1 consecutive words (four by default) in English."""
    counts = {}
    for text in texts:
        if language == 'en':
            words, n = re.findall(r"[a-z']+", text.lower()), size // 2 + 1
            seen = {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}
        else:
            seen = {text[i:i + size] for i in range(len(text) - size + 1)}
        for gram in seen:
            counts[gram] = counts.get(gram, 0) + 1
    limit = max(3, int(share * len(texts)) + 1)
    return sorted(gram for gram, n in counts.items() if n >= limit and not any(c in gram for c in '0123456789'))


def respread(kept, client, output):
    """Rewrite openings that share stock phrases with too many others (once)."""
    phrases = {language: repeated_phrases([r['user_request'] for r in kept if r.get('language', 'zh') == language],
                                          language=language) for language in LANGUAGE_NAMES}
    rewritten = []
    for request in kept:
        language = request.get('language', 'zh')
        text = request['user_request'].lower() if language == 'en' else request['user_request']
        hits = [p for p in phrases[language] if p in text]
        if not hits:
            continue
        text, _, _ = write_opening(client, request, request['id'], avoid=tuple(hits))
        if text:
            request['user_request'] = text
            (output / 'requests' / f"{request['id']}.json").write_text(json.dumps(request, ensure_ascii=False, indent=1))
            rewritten.append(request['id'])
    return rewritten


def generate(domains, count, output, *, runner, parallel=2, timeout=600, python='python', rubric_share=0.0,
             opening_client=None, english_share=0.0, mix=None):
    """Stage one: a codex session per task writes rules, tables, facts and a user scene. Stage two:
    opening_client (OPENING_*, falling back to LLM_*) writes the opening from the scene alone."""
    if opening_client is None:
        from env_factory.core.llm import ChatClient, load_env_file
        load_env_file()
        opening_client = ChatClient('OPENING')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)  # a rerun reuses the sessions that already passed
    assignments = plan(domains, count, rubric_share, **({'mix': mix} if mix else {}))
    langs = languages(count, english_share)
    elements = scripts(count, [shape for _, _, shape in assignments])

    def one(index, assignment):
        domain, category, shape = assignment
        folder = output / 'sessions' / f'{index:03d}'
        if (folder / 'request.json').exists():
            report = checks.run('request', folder)
            if report['passed']:
                return index, assignment, report
        folder.mkdir(parents=True, exist_ok=True)
        language = langs[index]
        (folder / 'assignment.json').write_text(json.dumps(
            {'domain': domain, 'category': category, 'shape': shape, 'id': f'gen-{index:03d}', 'language': language,
             'script': elements[index]},
            ensure_ascii=False))
        detail = domain.get('detail', '')
        if domain.get('process') or domain.get('mechanisms'):
            detail += f" 业务环节：{domain.get('process', '')}；需要体现的机制：{'、'.join(domain.get('mechanisms', []))}。"
        try:
            runner(folder, prompts.REQUEST_AUTHOR.format(python=python, domain=domain['name'],
                                                       detail=detail, category=category,
                                                       shape=SHAPES[shape], id=f'gen-{index:03d}',
                                                       reward_type='rubric' if shape == 'subjective' else 'business',
                                                       language=language, language_name=LANGUAGE_NAMES[language],
                                                       script=script_prompt(elements[index]),
                                                       currency=CURRENCY[language]),
                   timeout=timeout, role='request_author')
        except InfrastructureError as exc:  # codex at capacity or offline: this request waits for a rerun
            return index, assignment, {'passed': False, 'problems': [f'infrastructure: {exc}'[:300]],
                                       'infrastructure': True}
        report = checks.run('request', folder)
        return index, assignment, report

    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda pair: one(*pair), enumerate(assignments)))
    stalled = sum(1 for _, _, report in results if report.get('infrastructure'))
    if stalled:
        # Sessions that passed stay on disk and are reused by the rerun; only these are retried.
        raise InfrastructureError(f'{stalled}/{len(results)} request sessions hit infrastructure errors; rerun')
    def finish(item):
        """Stage two and three for one request: the opening, then the agent policy (model calls only)."""
        index, (domain, category, shape), report = item
        row = {'index': index, 'domain': domain['name'], 'category': category, 'shape': shape,
               'language': langs[index], 'passed': report['passed'], 'problems': report['problems'][:5]}
        if not report['passed']:
            return row, None
        request = json.loads((output / 'sessions' / f'{index:03d}' / 'request.json').read_text())
        request.setdefault('time', {'start': 10, 'tick': 0})
        request['language'] = langs[index]
        opening, persona_name, opening_problems_ = write_opening(opening_client, request, f'gen-{index:03d}')
        row.update(persona=persona_name)
        if opening is None:
            return {**row, 'passed': False, 'problems': opening_problems_}, None
        request['user_request'] = opening
        policy, policy_problems = policies.write(opening_client, request)
        if policy is None:
            return {**row, 'passed': False, 'problems': [f'agent policy: {p}' for p in policy_problems[:4]]}, None
        request['agent_policy'] = policy
        if elements[index] and isinstance(request.get('user_script'), dict):
            traits = request['user_script'].setdefault('traits', [])
            if DISCLOSURE[langs[index]] not in traits:
                traits.append(DISCLOSURE[langs[index]])
        request['source'] = {'generated': domain['name'], 'family': domain.get('family'), 'shape': shape,
                             'persona': persona_name, 'language': langs[index], 'script': elements[index],
                             'opening_writer': getattr(opening_client, 'model', None)}
        return row, request

    # Openings and policies are independent model calls: write them in parallel, then drop
    # near-duplicates in index order (deterministic whatever the completion order).
    pool = concurrent.futures.ThreadPoolExecutor(max(parallel, 8))
    started, futures = {}, {}

    def timed(item):
        started[item[0]] = time.monotonic()
        return finish(item)

    for item in results:
        futures[pool.submit(timed, item)] = item
    finished, pending = {}, set(futures)
    while pending:  # a stalled provider connection must not hold the whole batch (client timeouts do not always fire)
        done, pending = concurrent.futures.wait(pending, timeout=30)
        for future in done:
            index, (domain, category, shape), report = futures[future]
            try:
                finished[index] = future.result()
            except Exception as exc:
                finished[index] = ({'index': index, 'domain': domain['name'], 'category': category, 'shape': shape,
                                    'language': langs[index], 'passed': False, 'problems': [repr(exc)[:300]]}, None)
        now = time.monotonic()
        for future in [f for f in pending if now - started.get(futures[f][0], now) > TASK_TIMEOUT]:
            index, (domain, category, shape), report = futures[future]
            finished[index] = ({'index': index, 'domain': domain['name'], 'category': category, 'shape': shape,
                                'language': langs[index], 'passed': False,
                                'problems': ['timed out writing the opening or policy (stalled provider connection)']},
                               None)
            pending.discard(future)
    pool.shutdown(wait=False, cancel_futures=True)
    finished = [finished[index] for index, _, _ in results]
    kept, rows = [], []
    for row, request in finished:
        if request is not None:
            twin = next((k for k in kept if similarity(signature(k), signature(request)) >= SIMILAR), None)
            if twin is not None:
                row.update(passed=False, problems=[f'near-duplicate of {twin["id"]}'])
            else:
                request['id'] = f"gen-{row['index']:03d}"
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
