"""Branch audit: catch refusal cascades that env and oracle may share.

The differential check cannot see an error both implementations make. The likeliest shared
error is what happens after the user refuses one approval: the original world approves
everything, so the brief rarely spells out that branch, and both sides may guess the same way.
A cascade is an approval flip after which the reference never takes a step (a later confirmation
or a tool call with the same arguments) that the original world takes after that approval. Every cascade must be justified by a verbatim brief quote from a fresh auditor.
"""
from __future__ import annotations

import json

from env_factory_v3 import spec
from env_factory_v3.runtime import canonical, run_reference
from env_factory_v3.worlds import variants

VERDICTS = ('supported', 'unsupported', 'ambiguous')


def _confirms(trace):
    keys = []
    for entry in trace:
        if entry.get('kind') == 'confirm' and entry['key'] not in keys:
            keys.append(entry['key'])
    return keys


def _step(entry):
    if entry.get('kind') == 'confirm':
        return f"confirm {entry['key']}"
    if entry.get('kind') == 'tool':
        return f"{entry['tool']} {canonical(entry.get('args', {}))}"
    return None


def _summary(run):
    return {'asked_approvals': _confirms(run['trace']),
            'committed_writes': [{'tool': e['tool'], 'args': e['args']} for e in run['trace'] if e.get('write')],
            'answer': run['answer']}


def _skipped(original, flipped, denied):
    """Steps the original takes after asking `denied` that the refused world never takes at all.
    The write the refused consent was for is expected to vanish; re-reads of earlier steps do too."""
    trace = original['trace']
    at = next(i for i, e in enumerate(trace) if e.get('kind') == 'confirm' and e['key'] == denied)
    proposal = trace[at]['proposal']
    seen = {_step(e) for e in flipped['trace']}
    skipped = []
    for entry in trace[at + 1:]:
        step = _step(entry)
        if step is None or step in seen or step in skipped:
            continue
        if entry.get('kind') == 'tool' and canonical({'tool': entry['tool'], 'args': entry.get('args')}) == canonical(proposal):
            continue
        skipped.append(step)
    return skipped


def cascades(env, request):
    """Approval flips after which the reference drops steps the all-approved original takes."""
    start = request['time']['start']
    worlds = variants(request['tables'], request['user_facts'], perturbed=0)
    original = run_reference(env, worlds[0][1], worlds[0][2], start=start)
    asked = _confirms(original['trace'])
    found = []
    for name, data, facts in worlds[1:]:
        denied = name.split(':', 1)[1]
        if denied not in asked:
            continue
        run = run_reference(env, data, facts, start=start)
        skipped = _skipped(original, run, denied)
        if skipped:
            found.append({'world': name, 'denied': denied,
                          'skipped_approvals': [s.split(' ', 1)[1] for s in skipped if s.startswith('confirm ')],
                          'skipped_steps': skipped, 'original': _summary(original), 'flipped': _summary(run)})
    return found


def _counts(trace):
    counts = {}
    for entry in trace:
        key = (f"confirm {entry['key']}" if entry.get('kind') == 'confirm'
               else entry['tool'] if entry.get('kind') == 'tool' else None)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def unrepaired(env, request, found):
    """Cascades still present after a repair. A repaired refusal world may call the skipped tools
    with different arguments (state differs after a refusal), so the check is by count: it must
    call each skipped tool and ask each skipped approval at least as often as the original world."""
    start = request['time']['start']
    worlds = {name: (data, facts) for name, data, facts in
              variants(request['tables'], request['user_facts'], perturbed=0)}
    trace = run_reference(env, *worlds['original'], start=start)['trace']
    left = []
    for cascade in found:
        original = _counts(trace)
        # The refused write itself is expected to vanish from the refusal world.
        for entry in trace:
            if entry.get('kind') == 'confirm' and entry['key'] == cascade['denied']:
                tool = entry['proposal']['tool']
                original[tool] = original.get(tool, 0) - 1
        flipped = _counts(run_reference(env, *worlds[cascade['world']], start=start)['trace'])
        needed = {step.split(' ', 1)[0] if not step.startswith('confirm ') else step
                  for step in cascade['skipped_steps']}
        if any(flipped.get(key, 0) < original.get(key, 0) for key in needed):
            left.append(cascade['world'])
    return left


def verdict(folder, found, request):
    """Validate audit.json against the cascades it must cover."""
    try:
        decisions = json.loads((folder / 'audit.json').read_text())['branches']
        assert isinstance(decisions, list)
    except Exception:
        return {'valid': False, 'reason': 'audit.json missing or malformed'}
    by_world = {}
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get('verdict') not in VERDICTS:
            return {'valid': False, 'reason': f'bad verdict in {decision}'}
        if not spec.quoted(request, decision.get('brief_quote')):
            return {'valid': False, 'reason': 'brief_quote not found in the user request or business rules: '
                    f'{str(decision.get("brief_quote"))[:120]}'}
        by_world[decision.get('world')] = decision
    missing = [c['world'] for c in found if c['world'] not in by_world]
    if missing:
        return {'valid': False, 'reason': f'audit.json does not cover {missing}'}
    chosen = [by_world[c['world']] for c in found]
    return {'valid': True, 'decisions': chosen,
            'ambiguous': [d['world'] for d in chosen if d['verdict'] == 'ambiguous'],
            'unsupported': [d['world'] for d in chosen if d['verdict'] == 'unsupported'], 'reason': None}
