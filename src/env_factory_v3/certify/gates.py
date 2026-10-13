"""Qualification gates. Every gate is mandatory; there is no weighted threshold.

G1 static      generated code passes the sandbox rules, the answer schema is valid and every
              tool parameter has a type and a description (the agent sees nothing else)
natural        (readiness for the free-dialogue user, not a G1-G5 gate) approval_key maps each
              exact write to the approval topic the reference confirms
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

from env_factory_v3.core import answers, reward
from env_factory_v3.runtime import (BusinessError, Episode, InvalidWorld, canonical, is_write, run_policy,
                                    run_reference, tool_spec_problems)
from env_factory_v3.core.worlds import state_diff, variants


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
        problems = answers.answer_problems(schema, expected['answer']) + reward.alternative_problems(expected, schema, data)
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

    judge('null_policy', lambda api: {}, data, facts, expected, must_be_zero=True)  # doing nothing earns 0
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


def approval_problems(env, request):
    """Free dialogue needs env.approval_key(tool, args, facts): the approval topic that governs an
    exact write (platform side; the agent never sees it). It must name the topic the reference
    itself confirms, on every approval world."""
    from env_factory_v3.core import spec
    approvals = {t['key'] for t in spec.fact_topics(request) if t['kind'] == 'approval'}
    if not approvals:
        return []
    if not callable(getattr(env, 'approval_key', None)):
        return ['env.py must define approval_key(tool, args, facts) -> approval topic for an exact write']
    problems = []
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=0):
        for entry in run_reference(env, data, facts, start=request['time']['start'])['trace']:
            if entry.get('kind') != 'confirm':
                continue
            proposal = entry['proposal']
            try:
                key = env.approval_key(proposal['tool'], copy.deepcopy(proposal['args']), copy.deepcopy(facts))
            except Exception as exc:
                key = f'<raised {type(exc).__name__}>'
            if key != entry['key']:
                problems.append(f'{name}: approval_key({proposal["tool"]}, ...) = {key!r} but the reference '
                                f'confirms {entry["key"]!r}')
    return problems[:10]


def qualify(env, oracle, schema, request, *, perturbed=20, http=True, excluded=()):
    """Run G2-G5 (G1 is enforced when modules are loaded). Returns the full gate report."""
    schema_problems = answers.check_schema(schema)
    tool_problems = tool_spec_problems(env.make_tools())
    approval = approval_problems(env, request)
    report = {'static': {'passed': not schema_problems and not tool_problems, 'schema_problems': schema_problems,
                         'tool_problems': tool_problems[:10]},
              # Free dialogue readiness: not needed for the structured user, required for the natural one.
              'natural': {'passed': not approval, 'approval_problems': approval},
              'excluded_worlds': sorted(excluded)}
    worlds = compare_worlds(env, oracle, schema, request, perturbed=perturbed, excluded=excluded)
    report['differential'] = differential(worlds)
    report['valid_worlds'] = [w['world'] for w in worlds if w['status'] == 'agree']
    report['negatives'] = (negatives(env, oracle, schema, request, worlds) if report['differential']['passed']
                           else {'passed': False, 'note': 'not run: differential failed'})
    report['category'] = category(request, env) if report['differential']['passed'] else {'passed': False}
    if http and report['differential']['passed']:
        from env_factory_v3.serving.server import replay_check
        report['http'] = replay_check(env, oracle, schema, request)
    else:
        report['http'] = {'passed': False, 'note': 'not run'}
    report['policy'] = {'status': 'not_run', 'note': 'run `policy BUNDLE` (independent model, k episodes)'}
    report['gate_names'] = list(BUSINESS_GATES)
    report['passed'] = all(report[g]['passed'] for g in BUSINESS_GATES)
    report['worlds'] = worlds
    return report


BUSINESS_GATES = ('static', 'differential', 'negatives', 'category', 'http')
RUBRIC_GATES = ('static', 'rubric', 'negatives', 'category', 'http')
EMPTY_REPLY, GENERIC_REPLY = '', '好的，已经帮您处理好了，还有其他需要吗？'


def gate_names(report):
    return report.get('gate_names') or list(BUSINESS_GATES)


def _fabricate(reply):
    """The reference reply with every digit changed: a reply that misquotes the facts."""
    return ''.join(str((int(c) + 3) % 10) if c.isdigit() else c for c in reply)


def qualify_rubric(env, rubric_spec, schema, request, *, judge=None, http=True):
    """Gates for a rubric task. Without a judge (offline container check) only the deterministic
    gates run; the judge-based ones were passed at build time."""
    from env_factory_v3.core import rubric as rubric_mod
    start = request['time']['start']
    tool_problems = tool_spec_problems(env.make_tools())
    rubric_problems = rubric_mod.check_rubric(rubric_spec, request)
    approval = approval_problems(env, request)
    report = {'static': {'passed': not tool_problems and not rubric_problems, 'tool_problems': tool_problems[:10],
                         'rubric_problems': rubric_problems[:10]},
              'natural': {'passed': not approval, 'approval_problems': approval}, 'excluded_worlds': []}
    worlds = variants(request['tables'], request['user_facts'], perturbed=0)
    runs = {name: run_reference(env, data, facts, start=start) for name, data, facts in worlds}
    facts_of = {name: facts for name, _, facts in worlds}
    forged = Episode(env.make_tools(), request['tables'], request['user_facts'], start=start)
    try:
        env.reference(_Wrapped(forged, 'forged'))
    except Exception:  # the forged write is rejected; the violation is what matters
        pass
    writes = any(e.get('write') for e in runs['original']['trace'])
    report['negatives'] = {'passed': not writes or forged.safety_violations > 0,
                           'checks': {'forged_consent': {'applicable': writes,
                                                         'safety_violations': forged.safety_violations}}}
    report['category'] = category(request, env)
    if judge is None:
        report['rubric'] = {'passed': True, 'note': 'not run offline: needs the judge model (passed at build time)'}
        report['http'] = {'passed': True, 'note': 'not run offline: grading needs the judge model'}
        report['valid_worlds'] = [name for name, _, _ in worlds]
    else:
        score = lambda name, reply, **kw: judge.score(request, rubric_spec, rubric_mod.transcript(
            runs[name], facts_of[name]), reply, **kw)['score']
        reply = lambda name: (runs[name]['answer'] or {}).get('reply') or ''
        references = {name: score(name, reply(name)) for name in runs}
        original = reply('original')
        nothing = run_policy(env, lambda api: {'reply': GENERIC_REPLY}, request['tables'], request['user_facts'],
                             start=start)
        null_reward = rubric_mod.grade(nothing, rubric_spec, request, request['user_facts'], judge,
                                       reference_calls=runs['original']['tool_calls'])['reward']
        probes = {'empty': score('original', EMPTY_REPLY), 'generic': score('original', GENERIC_REPLY),
                  'null_policy_reward': null_reward,
                  'fabricated': score('original', _fabricate(original)) if any(c.isdigit() for c in original)
                  else None,
                  'repeat': score('original', original, fresh=True)}
        good = [name for name, value in references.items() if value >= rubric_mod.PASS]
        # Probes are judged relative to the reference: criteria about the conversation itself (asked
        # first, acted only on the user's own records) still earn credit with an empty reply.
        ceiling = lambda: min(rubric_mod.PASS - 0.2, references.get('original', 0) - 0.3)
        checks = {'references_pass': references.get('original', 0) >= rubric_mod.PASS and len(good) == len(references),
                  'empty_low': probes['empty'] <= ceiling(), 'generic_low': probes['generic'] <= ceiling(),
                  'fabricated_lower': probes['fabricated'] is None
                  or probes['fabricated'] <= references['original'] - 0.15,
                  'stable': abs(probes['repeat'] - references['original']) <= 0.15,
                  'null_policy_zero': probes['null_policy_reward'] == 0.0}
        report['rubric'] = {'passed': all(checks.values()), 'checks': checks, 'reference_scores': references,
                            'probe_scores': probes}
        report['valid_worlds'] = good
        if http and report['rubric']['passed']:
            from env_factory_v3.serving.server import replay_check
            report['http'] = replay_check(env, None, schema, request, rubric=rubric_spec, judge=judge)
        else:
            report['http'] = {'passed': False, 'note': 'not run'}
    report['policy'] = {'status': 'not_run', 'note': 'run `policy BUNDLE` (independent model, k episodes)'}
    report['gate_names'] = list(RUBRIC_GATES)
    report['passed'] = all(report[g]['passed'] for g in RUBRIC_GATES)
    return report
