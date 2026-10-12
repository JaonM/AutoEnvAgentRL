"""Build requests: the single input of the V3 pipeline, and the brief shown to models.

request.json
    id, family, category (direct_response | simple_agentic | multi_step_agentic), title,
    user_request   what the user says to the agent (public)
    business_rules implementation spec: the only semantic authority for env, oracle, arbiter and gates
    language       optional; "zh" (default) or "en": the language of the scene, opening, policy, the
                   simulated user and every platform text the agent sees
    agent_policy   optional; the same policies in business language (no tables, fields or clock
                   internals, no solution steps). When present it is all the agent ever sees of the rules
    tables         {table: [row objects]} initial data
    user_facts     {key: value} private user facts and decisions (approve* keys are consent answers)
    fact_descriptions  optional {key: public description of what the fact is}
    time           {"start": int, "tick": int}; tick 0 freezes the clock (required in M1)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

CATEGORIES = ('direct_response', 'simple_agentic', 'multi_step_agentic')
# business: one correct outcome, graded exactly against the oracle; rubric: subjective, graded by criteria.
REWARD_TYPES = ('business', 'rubric')
LANGUAGES = ('zh', 'en')
REQUIRED = ('id', 'family', 'category', 'title', 'user_request', 'business_rules', 'tables', 'user_facts')


class RequestError(ValueError):
    pass


def validate(request):
    problems = [f'missing {key}' for key in REQUIRED if key not in request]
    if problems:
        return problems
    if request['category'] not in CATEGORIES:
        problems.append(f'category must be one of {CATEGORIES}')
    tables = request['tables']
    if not isinstance(tables, dict) or not all(isinstance(rows, list) for rows in tables.values()):
        problems.append('tables must map names to row lists')
    else:
        for name, rows in tables.items():
            if not all(isinstance(row, dict) for row in rows):
                problems.append(f'{name}: every row must be an object')
                continue
            ids = [row['id'] for row in rows if 'id' in row]
            if len(set(map(json.dumps, ids))) != len(ids):
                problems.append(f'{name}: row ids must be unique')
    if not isinstance(request['user_facts'], dict):
        problems.append('user_facts must be an object')
    time = request.get('time', {'start': 0, 'tick': 0})
    if not (isinstance(time, dict) and isinstance(time.get('start'), int) and time.get('tick') == 0):
        problems.append('time must be {"start": int, "tick": 0}; the clock advances only by explicit waits')
    for text in ('user_request', 'business_rules', 'title'):
        if not isinstance(request.get(text), str) or not request[text].strip():
            problems.append(f'{text} must be nonempty text')
    if request.get('language', 'zh') not in LANGUAGES:
        problems.append(f'language must be one of {LANGUAGES}')
    if 'agent_policy' in request and not (isinstance(request['agent_policy'], str) and request['agent_policy'].strip()):
        problems.append('agent_policy must be nonempty text when present')
    events = tables.get('scheduled_events', []) if isinstance(tables, dict) else []
    for event in events:
        if not (isinstance(event, dict) and isinstance(event.get('at'), int) and isinstance(event.get('set'), dict)
                and event.get('table') in tables and event.get('table') != 'scheduled_events'
                and any(r.get('id') == event.get('row_id') for r in tables[event['table']])):
            problems.append(f'scheduled event {event.get("id") if isinstance(event, dict) else event} needs id, '
                            'integer at, an existing table and row_id, and a set object')
    if request.get('reward_type', 'business') not in REWARD_TYPES:
        problems.append(f'reward_type must be one of {REWARD_TYPES}')
    clarifications = request.get('clarifications', [])
    if not (isinstance(clarifications, list) and all(isinstance(c, str) and c.strip() for c in clarifications)):
        problems.append('clarifications must be a list of nonempty texts')
    return problems


def load(path):
    request = json.loads(Path(path).read_text())
    request.setdefault('time', {'start': 0, 'tick': 0})
    request.setdefault('fact_descriptions', {})
    problems = validate(request)
    if problems:
        raise RequestError('; '.join(problems))
    return request


def fact_topics(request):
    """Public list of what the user can be asked about (keys and descriptions, never values)."""
    descriptions = request.get('fact_descriptions') or {}
    topics = []
    for key in sorted(request['user_facts']):
        kind = 'approval' if key.startswith('approve') and isinstance(request['user_facts'][key], bool) else 'fact'
        topics.append({'key': key, 'kind': kind,
                       'description': descriptions.get(key) or (
                           'the user approves the corresponding operation when asked to confirm it'
                           if kind == 'approval' else f'user private fact "{key}"')})
    return topics


def authority_texts(request):
    """The only text a verdict may quote: the user's request, the business rules and the business
    owner's clarifications (never the model-written answer schema or the platform's summaries)."""
    return [' '.join(text.split()) for text in
            (request['user_request'], request['business_rules'], *request.get('clarifications', []))]


def quoted(request, quote):
    """True when `quote` is copied verbatim (whitespace-insensitive) from one authority text."""
    quote = ' '.join(str(quote or '').split())
    return len(quote) >= 4 and any(quote in text for text in authority_texts(request))


def brief(request, schema=None):
    tables = '\n'.join(f'- {name}: {len(rows)} rows; columns {sorted(rows[0]) if rows else "(empty)"}'
                       for name, rows in request['tables'].items())
    topics = '\n'.join(f'- {t["key"]} ({t["kind"]}): {t["description"]}' for t in fact_topics(request))
    if 'scheduled_events' in request['tables']:
        clock = (f"Starts at now = {request['time']['start']} and advances only when the agent waits. Rows of "
                 "scheduled_events (platform data, hidden from tools) are applied with "
                 "env_factory_v3.runtime.apply_event when the clock reaches their `at`; the oracle must apply "
                 "them the same way for the waits a correct handling performs.")
    else:
        clock = f"Fixed at now = {request['time']['start']} for this task."
    clarified = ''.join(f'- {c}\n' for c in request.get('clarifications', []))
    if clarified:
        clarified = '\n## Business owner clarifications (part of the business rules)\n' + clarified
    text = f"""# {request['title']}

Category: {request['category']}

## User request (what the user says to the agent)
{request['user_request']}

## Business rules (the only authority)
{request['business_rules']}
{clarified}
## Tables (full initial rows in data.json)
{tables}

## What the user can be asked (values in facts.json are private to the user)
{topics}

## Virtual time
{clock}
"""
    if request.get('reward_type') == 'rubric':
        text += ('\n## Grading\nSubjective task: there is no single correct reply. The conversation and the final '
                 'reply are graded against rubric criteria (each grounded in the request or the rules) and against '
                 'being grounded in what the tools and the user actually said.\n')
    if schema is not None:
        text += '\n## Answer schema (answer_schema.json)\n```json\n' + json.dumps(schema, ensure_ascii=False, indent=2) + '\n```\n'
    return text


def materialize(request, folder, schema=None):
    """Write the shared inputs every model role works from."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'brief.md').write_text(brief(request, schema))
    (folder / 'data.json').write_text(json.dumps(request['tables'], ensure_ascii=False, indent=1))
    (folder / 'facts.json').write_text(json.dumps(request['user_facts'], ensure_ascii=False, indent=1))
    (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1))
    if schema is not None:
        (folder / 'answer_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=1))


def expand_tables(business):
    """Expand V2 business tables, including compact sequence/cycle generators."""
    data = {}
    for table in business['tables']:
        if 'rows' in table:
            data[table['id']] = copy.deepcopy(table['rows'])
            continue
        plan, rows = table['generator'], []
        for index in range(plan['count']):
            row = {}
            for column, spec in plan['columns'].items():
                if spec['kind'] == 'sequence':
                    value = spec['start'] + index * spec['step']
                    if 'prefix' in spec:
                        value = spec['prefix'] + str(value).zfill(spec.get('width', 0))
                elif spec['kind'] == 'cycle':
                    position = index + plan.get('seed', 0) + spec.get('offset', 0)
                    value = copy.deepcopy(spec['values'][position % len(spec['values'])])
                elif spec['kind'] == 'constant':
                    value = copy.deepcopy(spec['value'])
                else:
                    raise RequestError(f'unsupported V2 generator kind {spec["kind"]}')
                row[column] = value
            rows.append(row)
        data[table['id']] = rows
    return data


def from_v2_frozen(frozen_root, sample_id, *, start=10):
    """Convert one frozen V2 cross-business sample into a V3 request."""
    root = Path(frozen_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    sample = next(s for s in manifest['samples'] if s['id'] == sample_id)
    folder = root / f'{sample_id:02d}'
    business = json.loads((folder / 'business.json').read_text())
    facts = json.loads((folder / 'user.json').read_text())['facts']
    request = {'id': f"v2-{sample['business_family'].split('_')[0]}-{sample_id:02d}",
               'family': sample['business_family'],
               'category': sample['category'], 'title': sample['title'],
               'user_request': sample['raw_request'], 'business_rules': business['policy'],
               'tables': expand_tables(business), 'user_facts': facts, 'fact_descriptions': {},
               'time': {'start': start, 'tick': 0},
               'source': {'frozen': str(root), 'sample': sample_id}}
    problems = validate(request)
    if problems:
        raise RequestError('; '.join(problems))
    return request


def findings_report(status):
    """Open questions for the business owner from a build's spec_findings (markdown)."""
    lines = [f"# Open business-rule questions: {status['id']}", '']
    for index, finding in enumerate(status.get('spec_findings', []), 1):
        worlds = finding.get('worlds') or ([finding['world']] if finding.get('world') else [])
        lines += [f"## {index}. {finding.get('issue') or finding.get('verdict', 'finding')}",
                  f"- rule text in question: {finding.get('brief_quote', '')}",
                  f"- why it is unclear: {finding.get('reason', '')}",
                  f"- worlds affected: {', '.join(worlds) or 'the task itself'}", '']
    if len(lines) == 2:
        lines.append('No open questions.')
    lines.append('Answer each question in one sentence; `clarify` adds the answers to the request.')
    return '\n'.join(lines) + '\n'


def clarify(request, answers):
    """A new request whose business rules carry the owner's answers."""
    request = json.loads(json.dumps(request))
    request['clarifications'] = [*request.get('clarifications', []), *[a.strip() for a in answers if a.strip()]]
    problems = validate(request)
    if problems:
        raise RequestError('; '.join(problems))
    return request
