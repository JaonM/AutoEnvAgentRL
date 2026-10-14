"""build pipeline: request -> answer schema -> (env.py || oracle.py) -> differential
arbitration -> qualification gates -> bundle.

One shared repair budget covers every corrective model session (author/oracle retry,
arbitration, gate repair). Infrastructure refusals never consume it and never count as
business failures.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

from env_factory import checks
from env_factory.authoring import branches, perturbations, prompts
from env_factory.authoring.codex import InfrastructureError
from env_factory.certify import bundle, gates
from env_factory.core import answers, rubric, spec
from env_factory.core.sandbox import load_module

DEFAULT_TIMEOUTS = {'schema': 240, 'author': 720, 'oracle': 600, 'rubric_author': 400, 'arbiter': 600, 'audit': 300,
                    'perturbation_audit': 300, 'repair': 600}


class Build:
    def __init__(self, request, output, *, runner, repairs=2, perturbed=20, timeouts=None, python=None, judge=None):
        self.request, self.output = request, Path(output).resolve()
        self.runner, self.budget, self.perturbed = runner, repairs, perturbed
        self.timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
        self.python = python or sys.executable
        self.status = {'id': request['id'], 'category': request['category'], 'family': request['family'],
                       'stages': {}, 'calls': [], 'repairs_used': 0, 'training_ready': False}
        self.started = time.monotonic()
        self.excluded = []
        self._judge = judge  # rubric tasks only; created from JUDGE_* / LLM_* when needed

    # -- helpers ------------------------------------------------------------------
    def save(self):
        self.status['elapsed_seconds'] = round(time.monotonic() - self.started, 1)
        (self.output / 'status.json').write_text(json.dumps(self.status, ensure_ascii=False, indent=1, default=str))

    def stage(self, name, **values):
        self.status['stages'].setdefault(name, {}).update(values)
        self.save()

    def call(self, folder, template, role):
        record = self.runner(folder, template.format(python=self.python), timeout=self.timeouts[role], role=role)
        self.status['calls'].append({k: record.get(k) for k in ('role', 'status', 'seconds', 'usage', 'attempts')})
        self.save()
        return record

    def spend(self):
        if self.budget <= 0:
            return False
        self.budget -= 1
        self.status['repairs_used'] += 1
        return True

    def fail(self, failure_class, reason):
        self.status.update(failure_class=failure_class, reason=reason, training_ready=False)
        self.save()
        return self.status

    def workdir(self, name, schema=None, copy_from=()):
        folder = self.output / name
        spec.materialize(self.request, folder, schema)
        for source in copy_from:
            shutil.copy(source, folder / Path(source).name)
        return folder

    # -- stages -------------------------------------------------------------------
    def run(self):
        self.output.mkdir(parents=True)
        identity = bundle.platform_identity()
        self.status['platform_identity'] = identity
        (self.output / 'request.json').write_text(json.dumps(self.request, ensure_ascii=False, indent=1))
        try:
            result = self._run()
        except InfrastructureError as exc:
            result = self.fail('infrastructure', str(exc)[:500])
        except Exception:
            result = self.fail('harness_error', traceback.format_exc(limit=6))
        if bundle.platform_identity() != identity:
            result = self.fail('harness_error', 'platform changed during the build; discard this run')
        return result

    def _run(self):
        if self.request.get('reward_type') == 'rubric':
            return self._run_rubric()
        folder = self.workdir('schema')
        self.call(folder, prompts.SCHEMA, 'schema')
        report = checks.run('schema', folder)
        self.stage('schema', passed=report['passed'], problems=report['problems'][:5])
        if not report['passed']:
            return self.fail('generation', 'answer schema failed its check')
        schema = json.loads((folder / 'answer_schema.json').read_text())

        author, oracle = self.workdir('author', schema), self.workdir('oracle', schema)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.call, author, prompts.AUTHOR, 'author'),
                       pool.submit(self.call, oracle, prompts.ORACLE, 'oracle')]
            for future in futures:
                future.result()
        for role, folder_, template in (('env', author, prompts.AUTHOR), ('oracle', oracle, prompts.ORACLE)):
            report = checks.run(role, folder_)
            if not report['passed'] and self.spend():
                retry = template + '\nA previous attempt in this folder failed its check; read check-' + role + '.json first.\n'
                self.call(folder_, retry, 'author' if role == 'env' else 'oracle')
                report = checks.run(role, folder_)
            self.stage(role, passed=report['passed'], problems=report['problems'][:5])
            if not report['passed']:
                return self.fail('generation', f'{role} failed its self-check')

        arbiter = self.workdir('arbiter', schema, copy_from=(author / 'env.py', oracle / 'oracle.py'))
        report = checks.run('diff', arbiter)
        self.stage('differential', passed=report['passed'], before=report['notes'], problems=report['problems'][:8])
        if not report['passed']:
            if not self.spend():
                return self.fail('disagreement', 'env and oracle disagree; no repair budget left')
            self.call(arbiter, prompts.ARBITER, 'arbiter')
            verdict = self.verdict(arbiter)
            self.stage('arbitration', **verdict)
            if not verdict['valid']:
                return self.fail('arbitration_invalid', verdict['reason'])
            if verdict['ambiguous']:
                core = sorted(w for w in verdict['ambiguous_worlds'] if not w.startswith('perturb'))
                self.status['spec_findings'] = [d for d in verdict['decisions'] if d['wrong_side'] == 'ambiguous']
                if core or not verdict['ambiguous_worlds']:
                    return self.fail('spec_ambiguous', f'business rules support both readings on {core or "unnamed worlds"}')
                # Ambiguity confined to perturbed worlds: the task itself is well defined there; those
                # worlds are excluded from comparison and from training, and reported to the owner.
                self.excluded = sorted(verdict['ambiguous_worlds'])
                (arbiter / 'excluded_worlds.json').write_text(json.dumps(self.excluded))
                self.stage('arbitration', excluded_worlds=self.excluded)
            report = checks.run('diff', arbiter)
            self.stage('differential', passed=report['passed'], after=report['notes'], problems=report['problems'][:8])
            if not report['passed'] or not checks.run('env', arbiter)['passed'] or not checks.run('oracle', arbiter)['passed']:
                return self.fail('disagreement', 'env and oracle still disagree after arbitration')

        failure = self.audit_branches(arbiter, schema) or self.audit_perturbations(arbiter, schema)
        if failure:
            return failure

        report = self.qualify(arbiter, schema)
        if not report['passed'] and self.spend():
            (arbiter / 'gate_report.json').write_text(json.dumps(self.failed_gates(report), ensure_ascii=False, indent=1, default=str))
            self.call(arbiter, prompts.REPAIR, 'repair')
            if checks.run('env', arbiter)['passed'] and checks.run('diff', arbiter)['passed']:
                report = self.qualify(arbiter, schema)
        if not report['passed']:
            failed = [g for g in gates.gate_names(report) if not report[g]['passed']]
            return self.fail('gates', f'failed gates: {failed}')
        manifest = bundle.export(arbiter, self.output / 'request.json', report, self.output / 'bundle',
                                 status={'training_ready': True, 'policy_gate': 'not_run'})
        self.status.update(training_ready=True, bundle=str(self.output / 'bundle'),
                           bundle_files=manifest['files'], failure_class=None)
        self.save()
        return self.status

    def audit_branches(self, arbiter, schema):
        """Refusal cascades env and oracle agree on still need a verbatim brief justification
        from a fresh auditor that sees neither implementation. Not a repair: no budget spent."""
        found = branches.cascades(load_module(arbiter / 'env.py', role='env'), self.request)
        self.stage('branch_audit', cascades=[{k: c[k] for k in ('world', 'denied', 'skipped_steps')} for c in found])
        if not found:
            return None
        folder = self.workdir('audit', schema)
        (folder / 'branches.json').write_text(json.dumps(found, ensure_ascii=False, indent=1))
        self.call(folder, prompts.AUDITOR, 'audit')
        verdict = branches.verdict(folder, found, self.request)
        self.stage('branch_audit', **verdict)
        if not verdict['valid']:
            return self.fail('audit_invalid', verdict['reason'])
        if verdict['ambiguous']:
            self.status.setdefault('spec_findings', []).extend(
                d for d in verdict['decisions'] if d['verdict'] == 'ambiguous')
            return self.fail('spec_ambiguous', f'refusal branch not decided by the business rules on {verdict["ambiguous"]}')
        if not verdict['unsupported']:
            return None
        if not self.spend():
            return self.fail('branch_unsupported', f'env and oracle skip steps the rules require on {verdict["unsupported"]}')
        # Both sides share the error, so both are fixed, from the auditor's quoted finding.
        for name in ('branches.json', 'audit.json'):
            shutil.copy(folder / name, arbiter / name)
        self.call(arbiter, prompts.BRANCH_REPAIR, 'repair')
        if not all(checks.run(kind, arbiter)['passed'] for kind in ('env', 'oracle', 'diff')):
            return self.fail('branch_unsupported', 'env and oracle no longer pass their checks after the branch repair')
        unsupported = [c for c, d in zip(found, verdict['decisions']) if d['verdict'] == 'unsupported']
        left = branches.unrepaired(load_module(arbiter / 'env.py', role='env'), self.request, unsupported)
        self.stage('branch_audit', repaired=not left, still_cascading=left)
        if left:
            return self.fail('branch_unsupported', f'refusal cascades remain after the branch repair on {left}')
        return None

    def audit_perturbations(self, arbiter, schema):
        """Perturbed worlds where both sides ignore a change to the rows the task acts on need a
        fresh auditor's verdict. Rules both ignore are repaired on both sides (one unit of budget);
        otherwise those worlds, like ambiguous ones, are excluded from training and reported."""
        load = lambda: load_module(arbiter / 'env.py', role='env')
        found = perturbations.insensitive(load(), self.request, perturbed=self.perturbed, excluded=self.excluded)
        self.stage('perturbation_audit', insensitive=[f['world'] for f in found])
        if not found:
            return None
        folder = self.workdir('perturbation_audit', schema)
        (folder / 'perturbations.json').write_text(json.dumps(found, ensure_ascii=False, indent=1, default=str))
        self.call(folder, prompts.PERTURBATION_AUDITOR, 'perturbation_audit')
        verdict = perturbations.verdict(folder, found, self.request)
        self.stage('perturbation_audit', **verdict)
        if not verdict['valid']:
            return self.fail('audit_invalid', verdict['reason'])
        findings = [d for d in verdict['decisions'] if d['verdict'] != 'consistent']
        self.status.setdefault('spec_findings', []).extend(d for d in findings if d['verdict'] == 'ambiguous')
        drop = set(verdict['ambiguous'])
        if verdict['should_change']:
            repaired = False
            if self.spend():
                for name in ('perturbations.json', 'perturbation_audit.json'):
                    shutil.copy(folder / name, arbiter / name)
                self.call(arbiter, prompts.PERTURBATION_REPAIR, 'repair')
                if all(checks.run(kind, arbiter)['passed'] for kind in ('env', 'oracle', 'diff')):
                    still = {f['world'] for f in perturbations.insensitive(
                        load(), self.request, perturbed=self.perturbed, excluded=self.excluded)}
                    repaired = not (still & set(verdict['should_change']))
                else:
                    return self.fail('perturbation_unsupported', 'env and oracle no longer pass their checks '
                                     'after the perturbation repair')
            if not repaired:
                drop |= set(verdict['should_change'])
                self.status.setdefault('spec_findings', []).extend(
                    d for d in findings if d['verdict'] == 'should_change')
            self.stage('perturbation_audit', repaired=repaired)
        if drop:
            self.excluded = sorted(set(self.excluded) | drop)
            (arbiter / 'excluded_worlds.json').write_text(json.dumps(self.excluded))
            self.stage('perturbation_audit', excluded_worlds=sorted(drop))
        return None

    def _run_rubric(self):
        """Subjective task: env.py and an independently written rubric, qualified by the judge."""
        schema = rubric.SCHEMA
        author, critic = self.workdir('author', schema), self.workdir('rubric', schema)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.call, author, prompts.AUTHOR + prompts.AUTHOR_RUBRIC, 'author'),
                       pool.submit(self.call, critic, prompts.RUBRIC_AUTHOR, 'rubric_author')]
            for future in futures:
                future.result()
        for role, folder_, template, kind in (('env', author, prompts.AUTHOR + prompts.AUTHOR_RUBRIC, 'author'),
                                               ('rubric', critic, prompts.RUBRIC_AUTHOR, 'rubric_author')):
            report = checks.run(role, folder_)
            if not report['passed'] and self.spend():
                retry = template + f'\nA previous attempt in this folder failed its check; read check-{role}.json first.\n'
                self.call(folder_, retry, kind)
                report = checks.run(role, folder_)
            self.stage(role, passed=report['passed'], problems=report['problems'][:5])
            if not report['passed']:
                return self.fail('generation', f'{role} failed its self-check')
        final = self.workdir('final', schema, copy_from=(author / 'env.py', critic / 'rubric.json'))
        report = self.qualify_rubric(final)
        if not report['passed'] and self.spend():
            (final / 'gate_report.json').write_text(json.dumps(
                {g: report[g] for g in gates.gate_names(report) if not report[g]['passed']},
                ensure_ascii=False, indent=1, default=str))
            self.call(final, prompts.REPAIR, 'repair')  # env.py only: the rubric stays independent
            if checks.run('env', final)['passed']:
                report = self.qualify_rubric(final)
        if not report['passed']:
            failed = [g for g in gates.gate_names(report) if not report[g]['passed']]
            return self.fail('gates', f'failed gates: {failed}')
        manifest = bundle.export(final, self.output / 'request.json', report, self.output / 'bundle',
                                 status={'training_ready': True, 'policy_gate': 'not_run', 'reward_type': 'rubric'})
        self.status.update(training_ready=True, bundle=str(self.output / 'bundle'),
                           bundle_files=manifest['files'], failure_class=None)
        self.save()
        return self.status

    def qualify_rubric(self, folder):
        env = load_module(folder / 'env.py', role='env')
        spec_ = json.loads((folder / 'rubric.json').read_text())
        started = time.monotonic()
        report = gates.qualify_rubric(env, spec_, rubric.SCHEMA, self.request, judge=self.judge())
        self.stage('gates', seconds=round(time.monotonic() - started, 1),
                   **{g: report[g]['passed'] for g in gates.gate_names(report)},
                   rubric_detail={k: report['rubric'].get(k) for k in ('checks', 'reference_scores', 'probe_scores')})
        return report

    def judge(self):
        if self._judge is None:
            self._judge = bundle.default_judge()
        return self._judge

    def qualify(self, folder, schema):
        env = load_module(folder / 'env.py', role='env')
        oracle = load_module(folder / 'oracle.py', role='oracle')
        started = time.monotonic()
        report = gates.qualify(env, oracle, schema, self.request, perturbed=self.perturbed, excluded=self.excluded)
        self.stage('gates', seconds=round(time.monotonic() - started, 1),
                   **{g: report[g]['passed'] for g in gates.gate_names(report)},
                   negatives_detail=report['negatives'].get('checks'), category_detail=report['category'])
        return report

    @staticmethod
    def failed_gates(report):
        out = {}
        for gate in ('differential', 'negatives', 'category', 'http'):
            if not report[gate]['passed']:
                detail = dict(report[gate])
                if gate == 'differential':
                    detail['mismatches'] = detail.get('mismatches', [])[:3]
                out[gate] = detail
        return out

    def verdict(self, folder):
        path = folder / 'verdict.json'
        try:
            decisions = json.loads(path.read_text())['decisions']
            assert isinstance(decisions, list) and decisions
        except Exception:
            return {'valid': False, 'ambiguous': False, 'reason': 'verdict.json missing or malformed'}
        for decision in decisions:
            if decision.get('wrong_side') not in ('env', 'oracle', 'both', 'ambiguous'):
                return {'valid': False, 'ambiguous': False, 'reason': f'bad wrong_side in {decision}'}
            if not spec.quoted(self.request, decision.get('brief_quote')):
                return {'valid': False, 'ambiguous': False, 'reason': 'brief_quote not found in the user request or '
                        f'business rules: {str(decision.get("brief_quote"))[:120]}'}
        mismatched = {m['world'] for m in json.loads((folder / 'mismatches.json').read_text())} \
            if (folder / 'mismatches.json').exists() else set()
        ambiguous_worlds = set()
        for decision in decisions:
            if decision['wrong_side'] != 'ambiguous':
                continue
            worlds = decision.get('worlds')
            if not isinstance(worlds, list) or not worlds or not set(worlds) <= mismatched:
                # Unnamed or unknown worlds: treat the ambiguity as affecting the task itself.
                return {'valid': True, 'ambiguous': True, 'decisions': decisions, 'ambiguous_worlds': [],
                        'reason': None}
            ambiguous_worlds |= set(worlds)
        ambiguous = any(d['wrong_side'] == 'ambiguous' for d in decisions)
        return {'valid': True, 'ambiguous': ambiguous, 'decisions': decisions,
                'ambiguous_worlds': sorted(ambiguous_worlds), 'reason': None}


def build(request, output, *, runner, **options):
    return Build(request, output, runner=runner, **options).run()


def batch(requests, output, *, runner, parallel=1, **options):
    """Build several requests; at most `parallel` builds (each uses up to 2 sessions)."""
    output = Path(output)
    output.mkdir(parents=True)
    safe = lambda text: re.sub(r'[^A-Za-z0-9_.-]+', '-', text)
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda r: build(r, output / safe(r['id']), runner=runner, **options), requests))
    summary = {'total': len(results), 'training_ready': sum(bool(r['training_ready']) for r in results),
               'failure_classes': {}, 'results': [{k: r.get(k) for k in ('id', 'category', 'training_ready',
                                                   'failure_class', 'reason', 'elapsed_seconds', 'repairs_used')}
                                                  for r in results]}
    for r in results:
        if not r['training_ready']:
            key = r.get('failure_class') or 'unknown'
            summary['failure_classes'][key] = summary['failure_classes'].get(key, 0) + 1
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary
