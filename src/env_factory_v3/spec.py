"""Build requests: the single input of the V3 pipeline, and the brief shown to models.

request.json
    id, family, category (direct_response | simple_agentic | multi_step_agentic), title,
    user_request   what the user says to the agent (public)
    business_rules policy text (public; the only semantic authority)
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
        problems.append('time must be {"start": int, "tick": 0} (non-frozen clocks are not supported in M1)')
    for text in ('user_request', 'business_rules', 'title'):
        if not isinstance(request.get(text), str) or not request[text].strip():
            problems.append(f'{text} must be nonempty text')
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
    """The only text a verdict may quote: the user's request and the business rules (never the
    model-written answer schema or the platform's table and topic summaries)."""
    return [' '.join(request['user_request'].split()), ' '.join(request['business_rules'].split())]


def quoted(request, quote):
    """True when `quote` is copied verbatim (whitespace-insensitive) from one authority text."""
    quote = ' '.join(str(quote or '').split())
    return len(quote) >= 4 and any(quote in text for text in authority_texts(request))


def brief(request, schema=None):
    tables = '\n'.join(f'- {name}: {len(rows)} rows; columns {sorted(rows[0]) if rows else "(empty)"}'
                       for name, rows in request['tables'].items())
    topics = '\n'.join(f'- {t["key"]} ({t["kind"]}): {t["description"]}' for t in fact_topics(request))
    text = f"""# {request['title']}

Category: {request['category']}

## User request (what the user says to the agent)
{request['user_request']}

## Business rules (the only authority)
{request['business_rules']}

## Tables (full initial rows in data.json)
{tables}

## What the user can be asked (values in facts.json are private to the user)
{topics}

## Virtual time
Fixed at now = {request['time']['start']} for this task.
"""
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
    request = {'id': f'v2-cross-{sample_id:02d}', 'family': sample['business_family'],
               'category': sample['category'], 'title': sample['title'],
               'user_request': sample['raw_request'], 'business_rules': business['policy'],
               'tables': expand_tables(business), 'user_facts': facts, 'fact_descriptions': {},
               'time': {'start': start, 'tick': 0},
               'source': {'frozen': str(root), 'sample': sample_id}}
    problems = validate(request)
    if problems:
        raise RequestError('; '.join(problems))
    return request
