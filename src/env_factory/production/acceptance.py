"""Frozen external acceptance and bounded implementation repair, without domain branches.

Checker commands are caller-owned configuration, never generated repair output. They
run outside the isolated author session. The repairer may change env.py only; policy,
request, answer schema, independent verifier, source snapshots and test oracles stay
fixed. Specification/teacher/infrastructure failures stop instead of changing truth.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

from env_factory import checks
from env_factory.authoring.codex import InfrastructureError
from env_factory.certify import bundle, gates
from env_factory.core import answers, spec
from env_factory.core.sandbox import load_module


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + '\n')


def hashes(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


class Suite:
    def __init__(self, value):
        self.path = Path(value).resolve() if isinstance(value, (str, Path)) else None
        self.config = json.loads(self.path.read_text()) if self.path else copy.deepcopy(value)
        if not isinstance(self.config, dict):
            raise ValueError('acceptance suite must be an object')
        checks_ = self.config.get('checks')
        if not isinstance(checks_, list) or not checks_:
            raise ValueError('acceptance suite needs a nonempty checks list')
        paths = [self.path] if self.path else []
        names = []
        for check in checks_:
            if not isinstance(check, dict):
                raise ValueError('each acceptance check must be an object')
            name, command = check.get('name', ''), check.get('command')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                raise ValueError('checker name must be a safe directory name')
            if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
                raise ValueError('checker command must be a nonempty argv list (no shell)')
            if not check.get('protected_files'):
                raise ValueError('declare checker code and oracle inputs as protected_files')
            if check.get('failure_class', 'implementation') not in ('implementation', 'specification', 'teacher', 'infrastructure'):
                raise ValueError('unsupported acceptance failure class')
            report = Path(check.get('report', 'report.json'))
            if report.is_absolute() or '..' in report.parts:
                raise ValueError('report must stay in the checker evidence directory')
            names.append(name)
            paths.extend(check['protected_files'])
        if len(names) != len(set(names)):
            raise ValueError('checker names must be unique')
        self.context = [Path(p).resolve() for p in self.config.get('repair_context', [])]
        self.frozen = hashes([*paths, *self.context])

    def intact(self):
        try:
            return hashes(self.frozen) == self.frozen
        except OSError:
            return False

    def evaluate(self, folder, evidence):
        if not self.intact():
            return {'passed': False, 'failure_class': 'acceptance_tampered', 'checks': []}
        results = []
        for check in self.config['checks']:
            out = Path(evidence) / check['name']
            out.mkdir(parents=True, exist_ok=True)
            command = [arg.replace('{bundle}', str(Path(folder).resolve())).replace('{evidence}', str(out.resolve()))
                       for arg in check['command']]
            try:
                process = subprocess.run(command, capture_output=True, text=True,
                                         timeout=check.get('timeout', 300), check=False)
                (out / 'stdout.txt').write_text(process.stdout)
                (out / 'stderr.txt').write_text(process.stderr)
                data = json.loads((out / check.get('report', 'report.json')).read_text())
                if not isinstance(data, dict):
                    raise ValueError('checker verdict must be an object')
                value = data.get(check.get('passed_field', 'passed'))
                if check.get('total_field'):
                    total = data.get(check['total_field'])
                    valid = type(value) is int and type(total) is int and total > 0
                    passed = valid and value == total
                else:
                    valid, passed = type(value) is bool, value is True
                if not valid or process.returncode not in (0, 1):
                    raise ValueError('checker did not return a valid verdict')
                results.append({'name': check['name'], 'passed': passed and process.returncode == 0,
                                'failure_class': check.get('failure_class', 'implementation'), 'report': data})
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                results.append({'name': check['name'], 'passed': False, 'failure_class': 'infrastructure',
                                'error': f'{type(error).__name__}: {error}'[:500]})
        if not self.intact():
            return {'passed': False, 'failure_class': 'acceptance_tampered', 'checks': results}
        failed = [r for r in results if not r['passed']]
        terminal = next((r['failure_class'] for r in failed if r['failure_class'] != 'implementation'), None)
        return {'passed': not failed, 'failure_class': terminal or ('implementation' if failed else None),
                'checks': results}


REPAIR = '''Repair only env.py using acceptance_feedback.json and the public sources in context/.
Read context/platform_rules.json before editing; public Python source must be adapted to these runtime rules.
The complete rule set applies on every attempt, including restrictions not mentioned in the last failure.
Use ordinary local variables, explicit function arguments/return values, or instance-local state when the
runtime forbids global/nonlocal declarations. Do not substitute unrestricted Python tests for platform checks.
The independent verifier, request, policy, schema, source files and test expectations are immutable.
Do not change them, remove cases, weaken assertions, or invent a domain rule to make a test pass.
Do not read outside this isolated folder. Do not hard-code fixture/customer IDs or return expected
values without implementing the general public behavior. Preserve unrelated tool behavior.
If the finding requires a specification decision or changing the verifier, leave env.py unchanged
and explain the ambiguity in repair_notes.txt. Infrastructure or teacher protocol failures must
never be repaired by changing business behavior. Run the exact self_check command in platform_rules.json
from this folder after editing. This trusted platform command is allowed to load its installed dependencies;
do not read other candidate folders or replace/stub platform modules. If the check fails, fix env.py and
repeat the check within this session before finishing. Preserve the resulting check-env.json as evidence.
'''


def repair_bundle(source, output, *, runner, suite, repairs=2, spend=None, refund=None, timeout=600):
    """Check a generated candidate and repair implementation defects against fixed truth.

Returns accepted=false on exhausted budget, ambiguity, tampering or infrastructure;
never promotes a merely certified bundle after an external acceptance failure.
"""
    suite = suite if isinstance(suite, Suite) else Suite(suite)
    if repairs < 0 or (spend is not None and refund is None):
        raise ValueError('repairs must be nonnegative; shared budget needs a refund callback')
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = spec.load(source / 'request.json')
    schema = answers.load_schema(source / 'answer_schema.json')
    names = [n for n in ('env.py', 'verifier.py', 'rubric.json') if (source / n).exists()]
    if 'verifier.py' not in names:
        raise ValueError('external implementation repair currently requires an independent business verifier')
    history, used, previous = [], 0, source
    identity = bundle.platform_identity()
    status = {'accepted': False, 'bundle': None, 'repairs_used': 0, 'rounds': history,
              'platform_identity': identity, 'protected_inputs': suite.frozen}

    def finish(failure=None, target=None):
        status.update(accepted=target is not None, bundle=str(target) if target else None,
                      failure_class=failure, repairs_used=used)
        write(output / 'acceptance.json', status)
        return status

    while True:
        folder = output / f'round-{used:02d}'
        spec.materialize(request, folder, schema)
        for name in names:
            shutil.copyfile(previous / name, folder / name)
        context = folder / 'context'
        context.mkdir()
        for index, path in enumerate(suite.context):
            shutil.copyfile(path, context / f'{index:02d}_{path.name}')
        from env_factory.core import sandbox
        write(context / 'platform_rules.json', {
            'allowed_imports': sorted(sandbox.ALLOWED_MODULES),
            'forbidden_names': sorted(sandbox.FORBIDDEN_NAMES),
            'allowed_dunder_attributes': sorted(sandbox.ALLOWED_DUNDERS),
            'global_and_nonlocal_statements_allowed': False,
            'self_check': f'{shlex.quote(sys.executable)} -m env_factory.checks env',
            'requirement': 'Passing raw Python or public API checks alone is insufficient; sandbox checks must pass.',
        })
        if history:
            # Only local evidence belongs in the isolated repair session. Host paths
            # are provenance, not permission to read sibling implementations.
            feedback = copy.deepcopy(history[-1])
            feedback.pop('candidate', None)
            for item in feedback['external']['checks']:
                item.get('report', {}).pop('bundle', None)
            write(folder / 'acceptance_feedback.json', feedback)
            frozen_paths = [folder / n for n in ('request.json', 'answer_schema.json', 'verifier.py',
                                                'brief.md', 'data.json', 'facts.json')]
            frozen = hashes([*frozen_paths, *context.iterdir()])
            before = bundle.sha256(folder / 'env.py')
            try:
                record = runner(folder, REPAIR + '\nIf self_check cannot execute, report the exact error; '
                                'do not claim qualification.\n', timeout=timeout, role='acceptance_repair')
                write(folder / 'repair_session.json', record)
            except InfrastructureError as error:
                used -= 1
                if refund is not None:
                    refund()
                status['error'] = str(error)[:500]
                return finish('infrastructure')
            try:
                intact = hashes(frozen) == frozen
            except OSError:
                intact = False
            if not intact or not suite.intact() or bundle.platform_identity() != identity:
                return finish('acceptance_tampered')
            if not (folder / 'env.py').is_file():
                return finish('invalid_candidate')
            if bundle.sha256(folder / 'env.py') == before:
                return finish('no_progress')
        static = {kind: checks.run(kind, folder) for kind in ('env', 'verifier', 'diff')}
        report = None
        if all(r['passed'] for r in static.values()):
            report = gates.qualify(load_module(folder / 'env.py', role='env'),
                                   load_module(folder / 'verifier.py', role='verifier'), schema, request)
        candidate_hashes = hashes(folder / n for n in ('env.py', 'verifier.py', 'request.json', 'answer_schema.json'))
        external = suite.evaluate(folder, folder / 'evidence')
        try:
            candidate_intact = hashes(candidate_hashes) == candidate_hashes
        except OSError:
            candidate_intact = False
        if not candidate_intact:
            return finish('acceptance_tampered')
        gate_ok = bool(report and report['passed'])
        current = {'round': used, 'candidate': str(folder), 'env_sha256': bundle.sha256(folder / 'env.py'),
                   'static': static, 'gates_passed': gate_ok,
                   'gates': {k: v for k, v in (report or {}).items() if k != 'worlds'}, 'external': external}
        history.append(current)
        write(folder / 'acceptance_result.json', current)
        if not suite.intact() or bundle.platform_identity() != identity:
            return finish('acceptance_tampered')
        if gate_ok and external['passed']:
            target = output / 'bundle'
            bundle.export(folder, folder / 'request.json', report, target,
                          status={'training_ready': True, 'external_acceptance': 'passed', 'policy_gate': 'not_run'})
            return finish(target=target)
        if external['failure_class'] not in (None, 'implementation'):
            return finish(external['failure_class'])
        if not static['verifier']['passed']:
            return finish('verifier_requires_independent_rebuild')
        if used >= repairs or (spend is not None and not spend()):
            return finish('repair_budget_exhausted')
        previous = folder
        used += 1


def main():
    import argparse
    from env_factory.authoring.codex import CodexRunner
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repairs', type=int, default=2)
    args = parser.parse_args()
    result = repair_bundle(args.bundle, args.output, runner=CodexRunner(), suite=args.suite, repairs=args.repairs)
    print(json.dumps({k: result[k] for k in ('accepted', 'bundle', 'failure_class', 'repairs_used')}))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
