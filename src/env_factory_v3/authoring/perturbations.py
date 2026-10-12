"""Perturbation audit: catch rules env and oracle both ignore on perturbed worlds.

Agreement on a perturbed world proves nothing when both sides ignore the same rule (an
inactive subscription still refunded). The signal: a perturbation changes a row the task acts
on (named by the user, written, or reported in the answer), yet the reference commits the same
writes and gives the same answer as on the original world. A fresh auditor that sees only the brief
decides, per world, whether the outcome should have changed.
"""
from __future__ import annotations

import json

from env_factory_v3.core import spec
from env_factory_v3.core.worlds import variants
from env_factory_v3.runtime import canonical, run_reference

VERDICTS = ('consistent', 'should_change', 'ambiguous')
LIMIT = 8


def _strings(value):
    if isinstance(value, dict):
        return {s for v in value.values() for s in _strings(v)}
    if isinstance(value, list):
        return {s for v in value for s in _strings(v)}
    return {value} if isinstance(value, str) else set()


def _outcome(run):
    return canonical([[e['tool'], e['args']] for e in run['trace'] if e.get('write')]), canonical(run['answer'])


def changed_cells(original, data, touched, written=()):
    cells = []
    for table, rows in original.items():
        for before, after in zip(rows, data.get(table, [])):
            if not isinstance(before, dict) or before.get('id') not in touched:
                continue
            for field, value in before.items():
                if after.get(field) != value:
                    cells.append({'table': table, 'id': before['id'], 'field': field,
                                  'original': value, 'perturbed': after.get(field),
                                  'row_written_by_task': before['id'] in written})
    return cells


def insensitive(env, request, *, perturbed=20, excluded=()):
    """Perturbed worlds that change rows the task touches but leave writes and answer unchanged."""
    start = request['time']['start']
    worlds = variants(request['tables'], request['user_facts'], perturbed=perturbed)
    name, original_data, facts = worlds[0]
    original = run_reference(env, original_data, facts, start=start)
    # Rows the task acts on: named by the user, written, or reported, not merely listed.
    written = _strings([e.get('args') for e in original['trace'] if e.get('write')])
    touched = written | _strings(original['answer']) | _strings(facts)
    found = []
    for name, data, facts_ in worlds:
        if not name.startswith('perturb') or name in excluded:
            continue
        cells = changed_cells(original_data, data, touched, written)
        if not cells:
            continue
        try:
            run = run_reference(env, data, facts_, start=start)
        except Exception:
            continue
        if _outcome(run) == _outcome(original):
            found.append({'world': name, 'changed_cells': cells,
                          'committed_writes': [{'tool': e['tool'], 'args': e['args']}
                                               for e in run['trace'] if e.get('write')],
                          'answer': run['answer']})
    # Flags flipped and values zeroed are the likeliest to matter; the auditor sees those first.
    weight = lambda entry: -sum(isinstance(c['original'], bool) or c['perturbed'] in (0, None, '')
                                for c in entry['changed_cells'])
    return sorted(found, key=weight)[:LIMIT]


def verdict(folder, found, request):
    """Validate perturbation_audit.json against the worlds it must cover."""
    try:
        decisions = json.loads((folder / 'perturbation_audit.json').read_text())['worlds']
        assert isinstance(decisions, list)
    except Exception:
        return {'valid': False, 'reason': 'perturbation_audit.json missing or malformed'}
    by_world = {}
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get('verdict') not in VERDICTS:
            return {'valid': False, 'reason': f'bad verdict in {decision}'}
        if decision['verdict'] != 'consistent' and not spec.quoted(request, decision.get('brief_quote')):
            return {'valid': False, 'reason': 'brief_quote not found in the user request or business rules: '
                    f'{str(decision.get("brief_quote"))[:120]}'}
        by_world[decision.get('world')] = decision
    missing = [f['world'] for f in found if f['world'] not in by_world]
    if missing:
        return {'valid': False, 'reason': f'perturbation_audit.json does not cover {missing}'}
    chosen = [by_world[f['world']] for f in found]
    return {'valid': True, 'decisions': chosen,
            'should_change': [d['world'] for d in chosen if d['verdict'] == 'should_change'],
            'ambiguous': [d['world'] for d in chosen if d['verdict'] == 'ambiguous'], 'reason': None}
