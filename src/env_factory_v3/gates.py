"""Qualification gates. Every gate is mandatory; there is no weighted threshold.

G1 static      generated code passes the sandbox rules, the answer schema is valid and every
              tool parameter has a type and a description (the agent sees nothing else)
G2 differential env.py's reference and the independent oracle agree (state and answer) on the
               original world, every approval flip and perturbed worlds; the reference earns
               reward 1.0 everywhere
G3 negatives   auto-derived wrong behaviours never succeed: doing nothing, replaying a stale
               answer, skipping consent, stopping after the first write, forging consent
G4 category    the reference trace has the structure its declared category requires
G5 http        replaying the reference through the HTTP service gives the in-process reward
Policy gates (frontier pass@k, adversarial agent) are reported as not_run in M1.
"""
from __future__ import annotations

import copy
import traceback

from env_factory_v3 import answers, reward
from env_factory_v3.runtime import (BusinessError, InvalidWorld, canonical, is_write, run_policy, run_reference,
                                    tool_spec_problems)
from env_factory_v3.worlds import state_diff, variants


def _expected(oracle, data, facts, now):
    result = oracle.expected(copy.deepcopy(data), copy.deepcopy(facts), now)
    if not (isinstance(result, dict) and 'answer' in result and 'final_state' in result):
        raise ValueError('oracle.expected must return {"answer": ..., "final_state": ...}')
    return result


def compare_worlds(env, oracle, schema, request, *, perturbed=20, excluded=()):
    """Run reference and oracle on every world variant and grade the reference.

    `excluded` names perturbed worlds where the business rules were found ambiguous; they are
    reported but neither compared nor used for training.
    """
    start = request['time']['start']
    excluded = set(excluded)
    worlds = []
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=perturbed):
        entry = {'world': name}
        if name in excluded:
            if not name.startswith('perturb'):
                raise ValueError(f'only perturbed worlds can be excluded, not {name}')
            entry.update(status='excluded_ambiguous')
            worlds.append(entry)
            continue
        try:
            expected = _expected(oracle, data, facts, start)
        except InvalidWorld as exc:
            entry.update(status='invalid_world', reason=str(exc)[:200])
            worlds.append(entry)
            continue
        except Exception:
            entry.update(status='oracle_error', error=traceback.format_exc(limit=3))
            worlds.append(entry)
            continue
        problems = answers.answer_problems(schema, expected['answer'])
        if problems:
            entry.update(status='oracle_error', error='oracle answer violates schema: ' + '; '.join(problems[:3]))
            worlds.append(entry)
            continue
        try:
            result = run_reference(env, data, facts, start=start)
        except Exception:
            entry.update(status='env_error', error=traceback.format_exc(limit=3))
            worlds.append(entry)
            continue
        grade = reward.grade(result, expected, schema, initial=data, reference_calls=result['tool_calls'])
        entry.update(status='agree' if grade['success'] and grade['reward'] == 1.0 else 'mismatch',
                     reward=grade['reward'], fields=grade['fields'], state_diff=grade['state_diff'],
                     format_problems=grade['format_problems'], expected_answer=expected['answer'],
                     reference_answer=result['answer'], reference_calls=result['tool_calls'],
                     writes=sum(1 for e in result['trace'] if e.get('write')),
                     data=data if name.startswith('perturb') else None, facts=facts,
                     trace_tail=result['trace'][-12:])
        worlds.append(entry)
    return worlds


def differential(worlds):
    counts = {}
    for world in worlds:
        counts[world['status']] = counts.get(world['status'], 0) + 1
    core = [w for w in worlds if not w['world'].startswith('perturb')]
    passed = (counts.get('mismatch', 0) == 0 and counts.get('env_error', 0) == 0
              and counts.get('oracle_error', 0) == 0 and all(w['status'] == 'agree' for w in core)
              and counts.get('agree', 0) >= len(core) + 5)
    return {'passed': passed, 'counts': counts,
            'mismatches': [w for w in worlds if w['status'] in ('mismatch', 'env_error', 'oracle_error')]}


class _Wrapped:
    """An api wrapper that changes one behaviour of the reference agent."""

    BusinessError = BusinessError

    def __init__(self, episode, mode):
        self.episode, self.mode = episode, mode

    def ask(self, key):
        return self.episode.ask(key)

    def wait(self, seconds):
        return self.episode.wait(seconds)

    def confirm(self, key, proposal):
        if self.mode == 'no_consent':
            return None
        if self.mode == 'partial' and any(e.get('write') for e in self.episode.trace):
            return None
        return self.episode.confirm(key, proposal)

    def call(self, name, **args):
        if self.mode == 'forged' and is_write(self.episode.tools.get(name)):
            args['consent'] = 'consent-forged'
        return self.episode.call(name, **args)


def negatives(env, oracle, schema, request, worlds):
    """Auto-derived wrong behaviours. Each applicable one must not succeed."""
    start = request['time']['start']
    data, facts = request['tables'], request['user_facts']
    original = next(w for w in worlds if w['world'] == 'original')
    expected = _expected(oracle, data, facts, start)
    has_effects = bool(state_diff(data, expected['final_state']))
    reference_calls = original['reference_calls']
    checks = {}

    def judge(name, policy, world_data, world_facts, world_expected, *, must_be_zero=False):
        try:
            result = run_policy(env, policy, world_data, world_facts, start=start)
        except Exception as exc:  # a crashing wrong behaviour cannot earn reward
            checks[name] = {'applicable': True, 'passed': True, 'note': f'policy raised {type(exc).__name__}'}
            return
        grade = reward.grade(result, world_expected, schema, initial=world_data, reference_calls=reference_calls)
        passed = not grade['success'] and (grade['reward'] == 0.0 if must_be_zero else grade['reward'] < 1.0)
        checks[name] = {'applicable': True, 'passed': passed, 'reward': grade['reward'], 'success': grade['success']}

    judge('null_policy', lambda api: {}, data, facts, expected)
    stale = [w for w in worlds if w['world'].startswith('perturb') and w['status'] == 'agree'
             and canonical(w['expected_answer']) != canonical(expected['answer'])][:5]
    if stale:
        results = []
        for world in stale:
            world_expected = _expected(oracle, world['data'], world['facts'], start)
            judge('stale_answer', lambda api: copy.deepcopy(expected['answer']), world['data'], world['facts'], world_expected)
            results.append(checks.pop('stale_answer'))
        checks['stale_answer'] = {'applicable': True, 'passed': all(r['passed'] for r in results), 'worlds': len(results)}
    else:
        checks['stale_answer'] = {'applicable': False, 'note': 'no perturbed world changes the expected answer'}
    if has_effects:
        judge('no_consent', lambda api: env.reference(_Wrapped(api, 'no_consent')), data, facts, expected)
        judge('forged_consent', lambda api: env.reference(_Wrapped(api, 'forged')), data, facts, expected, must_be_zero=True)
    else:
        checks['no_consent'] = checks['forged_consent'] = {'applicable': False, 'note': 'task has no state effects'}
    if original.get('writes', 0) >= 2:
        judge('partial', lambda api: env.reference(_Wrapped(api, 'partial')), data, facts, expected)
    else:
        checks['partial'] = {'applicable': False, 'note': 'reference performs fewer than two writes'}
    applicable = [c for c in checks.values() if c.get('applicable')]
    return {'passed': bool(applicable) and all(c['passed'] for c in applicable)
            and checks['null_policy']['passed'], 'checks': checks}


def _values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)
    elif isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool)):
        yield value


def trace_structure(trace, facts):
    """Tool calls, writes and whether a call consumed a value only seen in an earlier result."""
    fact_values = {canonical(v) for v in _values(facts)}
    seen, dependent, calls, writes = set(), False, 0, 0
    for entry in trace:
        if entry['kind'] != 'tool':
            continue
        calls += 1
        writes += bool(entry.get('write'))
        used = {canonical(v) for v in _values(entry.get('args', {}))} - fact_values
        if used & seen:
            dependent = True
        if entry.get('status') == 'ok':
            seen |= {canonical(v) for v in _values(entry.get('result'))}
    return {'tool_calls': calls, 'writes': writes, 'result_dependency': dependent}


def category(request, env):
    result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'])
    structure = trace_structure(result['trace'], request['user_facts'])
    declared = request['category']
    if declared == 'direct_response':
        ok = structure['tool_calls'] == 0
    elif declared == 'simple_agentic':
        ok = structure['tool_calls'] >= 1 and structure['writes'] == 0 and not structure['result_dependency']
    else:
        ok = structure['tool_calls'] >= 2 and (structure['result_dependency'] or structure['writes'] >= 1)
    return {'passed': ok, 'declared': declared, 'structure': structure}


def qualify(env, oracle, schema, request, *, perturbed=20, http=True, excluded=()):
    """Run G2-G5 (G1 is enforced when modules are loaded). Returns the full gate report."""
    schema_problems = answers.check_schema(schema)
    tool_problems = tool_spec_problems(env.make_tools())
    report = {'static': {'passed': not schema_problems and not tool_problems, 'schema_problems': schema_problems,
                         'tool_problems': tool_problems[:10]},
              'excluded_worlds': sorted(excluded)}
    worlds = compare_worlds(env, oracle, schema, request, perturbed=perturbed, excluded=excluded)
    report['differential'] = differential(worlds)
    report['valid_worlds'] = [w['world'] for w in worlds if w['status'] == 'agree']
    report['negatives'] = (negatives(env, oracle, schema, request, worlds) if report['differential']['passed']
                           else {'passed': False, 'note': 'not run: differential failed'})
    report['category'] = category(request, env) if report['differential']['passed'] else {'passed': False}
    if http and report['differential']['passed']:
        from env_factory_v3.server import replay_check
        report['http'] = replay_check(env, oracle, schema, request)
    else:
        report['http'] = {'passed': False, 'note': 'not run'}
    report['policy'] = {'status': 'not_run', 'note': 'frontier pass@k and adversarial agent arrive in M2'}
    report['passed'] = all(report[g]['passed'] for g in ('static', 'differential', 'negatives', 'category', 'http'))
    report['worlds'] = worlds
    return report
