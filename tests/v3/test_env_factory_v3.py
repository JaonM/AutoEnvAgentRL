"""EnvFactory V3 tests. No model calls: role sessions are replaced by a fake runner."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from env_factory_v3 import checks
from env_factory_v3.authoring import pipeline
from env_factory_v3.authoring.codex import InfrastructureError
from env_factory_v3.certify import bundle, gates
from env_factory_v3.core import answers, reward, spec
from env_factory_v3.core.sandbox import SandboxError, check_source, load_module
from env_factory_v3.runtime import BusinessError, Episode, run_reference, write
from env_factory_v3.serving import server

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'v3'
LIBRARY = FIXTURES / 'library'
PERTURBED_ONLY = ("    loan['due_day'] += book['renewal_days']\n",
                  "    loan['due_day'] += book['renewal_days'] + (1 if loan['renewals'] > 0 else 0)\n")
# Both sides stop after the first refusal: a shared guess the differential check cannot see.
CASCADE_ENV = ("            renewed.append(loan_id)\n    return",
               "            renewed.append(loan_id)\n        else:\n            break\n    return")
CASCADE_ORACLE = ("            renewed.append(loan_id)\n    return",
                  "            renewed.append(loan_id)\n        else:\n            break\n    return")
# Both sides ignore whether a loan is active: agreement everywhere, wrong on perturbed worlds.
IGNORE_ACTIVE_ENV = ("    if not loan['active']:\n        reasons.append('inactive_loan')\n", "")
IGNORE_ACTIVE_ORACLE = ("loan['reader_id'] == reader and loan['active'] and", "loan['reader_id'] == reader and")
BUGGY = ("if any(h['active'] and h['book_id'] == book['id'] and h['reader_id'] != reader_id for h in db['holds']):",
         "if False:")


def library_request():
    return spec.from_v2_frozen(FIXTURES / 'frozen', 4)


def library_modules(env_source=None):
    folder = Path(tempfile.mkdtemp())
    (folder / 'env.py').write_text(env_source or (LIBRARY / 'env.py').read_text())
    shutil.copy(LIBRARY / 'oracle.py', folder / 'oracle.py')
    schema = answers.load_schema(LIBRARY / 'answer_schema.json')
    return load_module(folder / 'env.py', role='env'), load_module(folder / 'oracle.py', role='oracle'), schema


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        @write
        def pay(db, now, account, cents):
            row = next(r for r in db['accounts'] if r['id'] == account)
            row['balance'] -= cents
            if row['balance'] < 0:
                raise BusinessError('insufficient funds')
            return {'balance': row['balance']}

        def balance(db, now, account):
            return next(r['balance'] for r in db['accounts'] if r['id'] == account)

        self.tools = {'pay': pay, 'balance': balance}
        self.data = {'accounts': [{'id': 'A', 'balance': 100}]}

    def episode(self, approve=True):
        return Episode(self.tools, self.data, {'approve_pay': approve, 'who': 'A'}, start=5)

    def test_consent_is_exact_and_single_use(self):
        ep = self.episode()
        token = ep.confirm('approve_pay', {'tool': 'pay', 'args': {'account': 'A', 'cents': 30}})
        with self.assertRaises(BusinessError):  # right token, wrong arguments: refused, not a violation
            ep.call('pay', consent=token, account='A', cents=31)
        self.assertEqual(ep.safety_violations, 0)
        self.assertEqual(ep.trace[-1]['status'], 'consent_mismatch')
        self.assertEqual(ep.db['accounts'][0]['balance'], 100)
        self.assertEqual(ep.call('pay', consent=token, account='A', cents=30), {'balance': 70})  # still usable
        with self.assertRaises(BusinessError):  # reuse: no consent for a second write
            ep.call('pay', consent=token, account='A', cents=30)
        self.assertEqual(ep.safety_violations, 1)
        with self.assertRaises(BusinessError):  # forged
            ep.call('pay', consent='consent-99', account='A', cents=1)
        self.assertEqual(ep.safety_violations, 2)
        self.assertEqual(ep.db['accounts'][0]['balance'], 70)

    def test_consent_for_another_tool_is_a_violation(self):
        @write
        def refund(db, now, account, cents):
            return {'refunded': cents}

        ep = Episode({**self.tools, 'refund': refund}, self.data, {'approve_pay': True}, start=5)
        token = ep.confirm('approve_pay', {'tool': 'pay', 'args': {'account': 'A', 'cents': 30}})
        with self.assertRaises(BusinessError):
            ep.call('refund', consent=token, account='A', cents=30)
        self.assertEqual(ep.safety_violations, 1)

    def test_declined_consent_and_rollback(self):
        self.assertIsNone(self.episode(approve=False).confirm('approve_pay', {'tool': 'pay', 'args': {}}))
        ep = self.episode()
        token = ep.confirm('approve_pay', {'tool': 'pay', 'args': {'account': 'A', 'cents': 500}})
        with self.assertRaises(BusinessError):
            ep.call('pay', consent=token, account='A', cents=500)
        self.assertEqual(ep.db['accounts'][0]['balance'], 100)
        self.assertEqual(ep.safety_violations, 0)

    def test_reads_cannot_mutate_and_clock(self):
        ep = Episode(self.tools, {**self.data, 'scheduled_events': []}, {}, start=5, tick=2)
        ep.call('balance', account='A')
        ep.wait(10)
        self.assertEqual(ep.now, 17)
        ep.submit({'x': 1})
        with self.assertRaises(Exception):
            ep.call('balance', account='A')


class SandboxTests(unittest.TestCase):
    def test_rejects_io_and_reflection(self):
        for source in ('import os', 'open("x")', 'x = ().__class__.__subclasses__()', 'eval("1")',
                       'from subprocess import run', 'global y'):
            self.assertTrue(check_source(source), source)

    def test_accepts_fixtures(self):
        for name in ('env.py', 'oracle.py'):
            self.assertEqual(check_source((LIBRARY / name).read_text()), [])

    def test_load_reports_missing_functions(self):
        folder = Path(tempfile.mkdtemp())
        (folder / 'env.py').write_text('def make_tools():\n    return {}\n')
        with self.assertRaises(SandboxError):
            load_module(folder / 'env.py', role='env')


class AnswerAndRewardTests(unittest.TestCase):
    def test_schema_and_comparison(self):
        schema = answers.load_schema(LIBRARY / 'answer_schema.json')
        expected = {'renewed_loan_ids': ['L1', 'L2'], 'blocked': []}
        self.assertTrue(all(answers.compare(schema, expected, {'renewed_loan_ids': ['L2', 'L1'], 'blocked': []}).values()))
        self.assertFalse(answers.compare(schema, expected, {'renewed_loan_ids': ['L1'], 'blocked': []})['renewed_loan_ids'])
        self.assertTrue(answers.check_schema({'fields': {'x': {'type': 'integer'}}}))
        number = {'fields': {'n': {'type': 'number', 'description': 'n'}}}
        self.assertTrue(answers.compare(number, {'n': 3}, {'n': 3.0})['n'])

    def test_prose_fields_must_be_ungraded_or_enumerated(self):
        prose = {'fields': {'ok': {'type': 'boolean', 'description': 'ok'},
                            'failure_reason': {'type': 'string', 'nullable': True, 'description': 'why'}}}
        self.assertTrue(answers.check_schema(prose))
        prose['fields']['failure_reason']['graded'] = False
        self.assertEqual(answers.check_schema(prose), [])
        self.assertEqual(list(answers.compare(prose, {'ok': True, 'failure_reason': 'a'},
                                              {'ok': True, 'failure_reason': 'b'})), ['ok'])
        enum = {'fields': {'failure': {'type': 'string', 'enum': ['none', 'ineligible'], 'description': 'kind'}}}
        self.assertEqual(answers.check_schema(enum), [])
        self.assertTrue(answers.answer_problems(enum, {'failure': 'other'}))
        self.assertTrue(answers.check_schema({'fields': {'note': {'type': 'string', 'graded': False, 'description': 'x'}}}))
        nested = {'fields': {'purchases': {'type': 'array', 'description': 'p', 'items': {'type': 'object', 'properties': {
            'order_id': {'type': 'string'}, 'reason': {'type': 'string', 'nullable': True}}}}}}
        self.assertTrue(answers.check_schema(nested))
        nested['fields']['purchases']['items']['properties']['reason']['graded'] = False
        self.assertEqual(answers.check_schema(nested), [])
        self.assertTrue(answers.compare(nested, {'purchases': [{'order_id': 'O1', 'reason': 'x'}]},
                                        {'purchases': [{'order_id': 'O1', 'reason': 'y'}]})['purchases'])
        self.assertFalse(answers.compare(nested, {'purchases': [{'order_id': 'O1', 'reason': 'x'}]},
                                         {'purchases': [{'order_id': 'O2', 'reason': 'x'}]})['purchases'])

    def test_reward_weights_and_safety(self):
        schema = {'fields': {'a': {'type': 'integer', 'description': 'a'}, 'b': {'type': 'integer', 'description': 'b'}}}
        initial = {'t': [{'id': 1, 'v': 0}]}
        expected = {'answer': {'a': 1, 'b': 2}, 'final_state': {'t': [{'id': 1, 'v': 1}]}}
        result = {'answer': {'a': 1, 'b': 0}, 'final_state': {'t': [{'id': 1, 'v': 1}]},
                  'safety_violations': 0, 'tool_calls': 2}
        grade = reward.grade(result, expected, schema, initial=initial, reference_calls=2)
        self.assertAlmostEqual(grade['reward'], 0.475 + 0.2375)
        self.assertFalse(grade['success'])
        read_only = {'answer': {'a': 1, 'b': 2}, 'final_state': initial}
        empty = {'answer': {}, 'final_state': initial, 'safety_violations': 0, 'tool_calls': 0}
        self.assertEqual(reward.grade(empty, read_only, schema, initial=initial, reference_calls=2)['reward'], 0.0)
        unsafe = dict(result, answer={'a': 1, 'b': 2}, safety_violations=1)
        self.assertEqual(reward.grade(unsafe, expected, schema, initial=initial, reference_calls=2)['reward'], 0.0)


class SpecTests(unittest.TestCase):
    def test_import_v2_frozen(self):
        request = library_request()
        self.assertEqual(request['category'], 'multi_step_agentic')
        self.assertEqual(sorted(request['tables']), ['books', 'holds', 'loans'])
        self.assertEqual(spec.validate(request), [])
        text = spec.brief(request)
        self.assertIn('approve_first', text)
        self.assertNotIn('"R1"', text)

    def test_validation(self):
        request = library_request()
        request['time'] = {'start': 0, 'tick': 1}
        self.assertTrue(spec.validate(request))


class GateTests(unittest.TestCase):
    def test_untyped_tool_parameters_fail_static(self):
        source = (LIBRARY / 'env.py').read_text().replace('book_id: str)', 'book_id)')
        env, oracle, schema = library_modules(source)
        report = gates.qualify(env, oracle, schema, library_request(), perturbed=5)
        self.assertFalse(report['static']['passed'])
        self.assertTrue(any('book_id' in p for p in report['static']['tool_problems']))

    def test_correct_environment_qualifies(self):
        env, oracle, schema = library_modules()
        report = gates.qualify(env, oracle, schema, library_request(), perturbed=20)
        self.assertTrue(report['passed'], {g: report[g] for g in ('differential', 'negatives', 'category', 'http')})
        self.assertTrue(report['negatives']['checks']['stale_answer']['applicable'])
        self.assertTrue(report['negatives']['checks']['forged_consent']['passed'])

    def test_buggy_environment_is_caught_by_the_oracle(self):
        env, oracle, schema = library_modules((LIBRARY / 'env.py').read_text().replace(*BUGGY))
        report = gates.qualify(env, oracle, schema, library_request(), perturbed=20)
        self.assertFalse(report['differential']['passed'])
        self.assertFalse(report['passed'])

    def test_trace_structure(self):
        env, _, _ = library_modules()
        request = library_request()
        result = run_reference(env, request['tables'], request['user_facts'], start=10)
        structure = gates.trace_structure(result['trace'], request['user_facts'])
        self.assertGreaterEqual(structure['tool_calls'], 3)
        self.assertTrue(structure['result_dependency'])


class ServerTests(unittest.TestCase):
    def test_http_episode(self):
        env, oracle, schema = library_modules()
        request = library_request()
        service = server.Service(env, oracle, schema, request, valid_worlds=['original', 'flip:approve_first'])
        httpd = server.start(service)
        try:
            api = server.HttpAPI(f'http://127.0.0.1:{httpd.server_address[1]}', None)
            reset = api._post('/reset', {'seed': 1})
            self.assertEqual(reset['world'], 'flip:approve_first')
            self.assertNotIn('approve_first": true', json.dumps(reset['instructions']))
            api.episode = reset['episode']
            bad = api._post('/step', {'episode': api.episode, 'action': {'type': 'tool', 'name': 'renew_loan',
                                                                          'args': {'loan_id': 'L1', 'reader_id': 'R1'}}})
            self.assertEqual(bad['kind'], 'business')
            api._post('/step', {'episode': api.episode, 'action': {'type': 'answer', 'answer': {}}})
            grade = api._post('/reward', {'episode': api.episode})
            self.assertEqual(grade['reward'], 0.0)
        finally:
            httpd.shutdown()
            httpd.server_close()


class FakeRunner:
    """Writes fixture files instead of calling a model; `env_source` lets a test plant bugs."""

    def __init__(self, env_source=None, arbiter=None, fail_role=None, oracle_source=None, auditor=None, repair=None,
                 perturbation_auditor=None):
        self.env_source, self.arbiter, self.fail_role, self.roles = env_source, arbiter, fail_role, []
        self.oracle_source, self.auditor, self.repair = oracle_source, auditor, repair
        self.perturbation_auditor = perturbation_auditor or self.all_consistent

    @staticmethod
    def all_consistent(folder):
        found = json.loads((folder / 'perturbations.json').read_text())
        (folder / 'perturbation_audit.json').write_text(json.dumps({'worlds': [
            {'world': f['world'], 'verdict': 'consistent', 'reason': 'irrelevant'} for f in found]}))

    def __call__(self, folder, prompt, *, timeout, role):
        self.roles.append(role)
        folder = Path(folder)
        if role == self.fail_role:
            raise InfrastructureError(f'{role}: 403 Forbidden')
        if role == 'schema':
            shutil.copy(LIBRARY / 'answer_schema.json', folder / 'answer_schema.json')
        elif role == 'author':
            (folder / 'env.py').write_text(self.env_source or (LIBRARY / 'env.py').read_text())
        elif role == 'oracle':
            (folder / 'oracle.py').write_text(self.oracle_source or (LIBRARY / 'oracle.py').read_text())
        elif role == 'arbiter' and self.arbiter:
            self.arbiter(folder)
        elif role == 'audit' and self.auditor:
            self.auditor(folder)
        elif role == 'repair' and self.repair:
            self.repair(folder)
        elif role == 'perturbation_audit':
            self.perturbation_auditor(folder)
        return {'role': role, 'status': 'exited', 'seconds': 0.0, 'usage': {}, 'attempts': 1}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def build(self, runner, request=None, **options):
        return pipeline.build(request or library_request(), self.root / 'build', runner=runner, perturbed=20, **options)

    @staticmethod
    def both_renewable():
        """No competing holds, so the reference asks both approvals in the original world."""
        request = library_request()
        for hold in request['tables']['holds']:
            hold['active'] = False
        return request

    def test_successful_build_exports_a_servable_bundle(self):
        status = self.build(FakeRunner())
        self.assertTrue(status['training_ready'], status)
        loaded = bundle.load_bundle(status['bundle'])
        self.assertIn('original', loaded['valid_worlds'])
        (Path(status['bundle']) / 'env.py').write_text('tampered')
        with self.assertRaises(ValueError):
            bundle.load_bundle(status['bundle'])

    def test_arbiter_fix_with_verbatim_quote(self):
        def arbiter(folder):
            shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'competing holds', 'wrong_side': 'env', 'reason': 'env ignored other readers',
                'brief_quote': '次数未达实际关联书目上限且无其他读者有效预约'}]}, ensure_ascii=False))
        status = self.build(FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*BUGGY), arbiter=arbiter))
        self.assertTrue(status['training_ready'], status)
        self.assertEqual(status['repairs_used'], 1)

    def test_arbiter_quote_must_exist(self):
        def arbiter(folder):
            shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'x', 'wrong_side': 'env', 'reason': 'x', 'brief_quote': 'a rule that is not in the brief'}]}))
        status = self.build(FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*BUGGY), arbiter=arbiter))
        self.assertEqual(status['failure_class'], 'arbitration_invalid')

    def test_arbiter_cannot_quote_the_answer_schema(self):
        def arbiter(folder):
            shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'x', 'wrong_side': 'env', 'reason': 'x',
                'brief_quote': "Ids of the user's loans renewed in this conversation."}]}))
        status = self.build(FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*BUGGY), arbiter=arbiter))
        self.assertEqual(status['failure_class'], 'arbitration_invalid')

    def test_ambiguity_is_reported_not_decided(self):
        def arbiter(folder):
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'x', 'wrong_side': 'ambiguous', 'reason': 'two readings', 'brief_quote': '允许部分成功'}]},
                ensure_ascii=False))
        status = self.build(FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*BUGGY), arbiter=arbiter))
        self.assertEqual(status['failure_class'], 'spec_ambiguous')

    def test_ambiguity_only_in_perturbed_worlds_excludes_them(self):
        def arbiter(folder):
            worlds = [m['world'] for m in json.loads((folder / 'mismatches.json').read_text())]
            self.assertTrue(worlds and all(w.startswith('perturb') for w in worlds))
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'x', 'wrong_side': 'ambiguous', 'worlds': worlds, 'reason': 'two readings',
                'brief_quote': '允许部分成功'}]}, ensure_ascii=False))
        env_source = (LIBRARY / 'env.py').read_text().replace(*PERTURBED_ONLY)
        status = self.build(FakeRunner(env_source=env_source, arbiter=arbiter))
        self.assertTrue(status['training_ready'], status)
        self.assertTrue(status['spec_findings'])
        loaded = bundle.load_bundle(status['bundle'])
        self.assertFalse(set(loaded['gates']['excluded_worlds']) & set(loaded['valid_worlds']))

    def test_ambiguity_on_the_original_world_fails(self):
        def arbiter(folder):
            worlds = [m['world'] for m in json.loads((folder / 'mismatches.json').read_text())]
            (folder / 'verdict.json').write_text(json.dumps({'decisions': [{
                'issue': 'x', 'wrong_side': 'ambiguous', 'worlds': worlds, 'reason': 'two readings',
                'brief_quote': '允许部分成功'}]}, ensure_ascii=False))
        status = self.build(FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*BUGGY), arbiter=arbiter))
        self.assertEqual(status['failure_class'], 'spec_ambiguous')

    def cascade_runner(self, verdict, quote='允许部分成功', repair=None):
        def auditor(folder):
            found = json.loads((folder / 'branches.json').read_text())
            self.assertFalse((folder / 'env.py').exists() or (folder / 'oracle.py').exists())
            (folder / 'audit.json').write_text(json.dumps({'branches': [
                {'world': c['world'], 'verdict': verdict, 'brief_quote': quote, 'reason': 'x'} for c in found]},
                ensure_ascii=False))
        return FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*CASCADE_ENV),
                          oracle_source=(LIBRARY / 'oracle.py').read_text().replace(*CASCADE_ORACLE), auditor=auditor,
                          repair=repair)

    def test_no_refusal_cascade_skips_the_audit(self):
        runner = FakeRunner()
        status = self.build(runner)
        self.assertTrue(status['training_ready'], status)
        self.assertNotIn('audit', runner.roles)
        self.assertEqual(status['stages']['branch_audit']['cascades'], [])

    def test_refusal_cascade_is_detected_and_audited(self):
        runner = self.cascade_runner('supported')
        status = self.build(runner, self.both_renewable())
        self.assertEqual(status['stages']['differential']['passed'], True)
        self.assertEqual([c['world'] for c in status['stages']['branch_audit']['cascades']], ['flip:approve_first'])
        self.assertIn('confirm approve_second', status['stages']['branch_audit']['cascades'][0]['skipped_steps'])
        self.assertIn('audit', runner.roles)
        self.assertTrue(status['training_ready'], status)
        self.assertEqual(status['repairs_used'], 0)

    def test_ambiguous_refusal_cascade_fails(self):
        status = self.build(self.cascade_runner('ambiguous'), self.both_renewable())
        self.assertEqual(status['failure_class'], 'spec_ambiguous')
        self.assertEqual(status['spec_findings'][0]['world'], 'flip:approve_first')

    def test_unsupported_refusal_cascade_is_repaired_on_both_sides(self):
        def repair(folder):
            self.assertTrue((folder / 'audit.json').exists())
            shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
            shutil.copy(LIBRARY / 'oracle.py', folder / 'oracle.py')
        runner = self.cascade_runner('unsupported', repair=repair)
        status = self.build(runner, self.both_renewable())
        self.assertTrue(status['training_ready'], status)
        self.assertIn('repair', runner.roles)
        self.assertEqual(status['repairs_used'], 1)
        self.assertTrue(status['stages']['branch_audit']['repaired'])

    def test_a_repaired_refusal_world_may_use_different_arguments(self):
        from env_factory_v3.authoring import branches
        env, _, _ = library_modules()
        cascade = {'world': 'flip:approve_first', 'denied': 'approve_first',
                   'skipped_steps': ['confirm approve_second',
                                     'renew_loan {"loan_id":"L2","reader_id":"an argument the repair changed"}']}
        self.assertEqual(branches.unrepaired(env, self.both_renewable(), [cascade]), [])
        broken, _, _ = library_modules((LIBRARY / 'env.py').read_text().replace(*CASCADE_ENV))
        self.assertEqual(branches.unrepaired(broken, self.both_renewable(), [cascade]), ['flip:approve_first'])

    def test_unsupported_refusal_cascade_left_unrepaired_fails(self):
        status = self.build(self.cascade_runner('unsupported', repair=lambda folder: None), self.both_renewable())
        self.assertEqual(status['failure_class'], 'branch_unsupported')
        self.assertEqual(status['stages']['branch_audit']['still_cascading'], ['flip:approve_first'])

    def test_unsupported_refusal_cascade_without_budget_fails(self):
        runner = self.cascade_runner('unsupported')
        status = self.build(runner, self.both_renewable(), repairs=0)
        self.assertEqual(status['failure_class'], 'branch_unsupported')
        self.assertNotIn('repair', runner.roles)

    def test_audit_cannot_quote_the_answer_schema(self):
        quote = "Ids of the user's loans renewed in this conversation."
        self.assertIn(quote, spec.brief(library_request(),
                      json.loads((LIBRARY / 'answer_schema.json').read_text())))
        status = self.build(self.cascade_runner('supported', quote=quote), self.both_renewable())
        self.assertEqual(status['failure_class'], 'audit_invalid')

    def test_audit_quote_must_exist(self):
        status = self.build(self.cascade_runner('supported', quote='a rule that is not in the brief'), self.both_renewable())
        self.assertEqual(status['failure_class'], 'audit_invalid')

    def ignore_active_runner(self, verdict, repair=None, quote='仅本人有效借阅可续借'):
        def auditor(folder):
            found = json.loads((folder / 'perturbations.json').read_text())
            self.assertFalse((folder / 'env.py').exists() or (folder / 'oracle.py').exists())
            (folder / 'perturbation_audit.json').write_text(json.dumps({'worlds': [
                {'world': f['world'], 'reason': 'x', 'brief_quote': quote,
                 'verdict': verdict if any(c['field'] == 'active' and c['table'] == 'loans'
                                           for c in f['changed_cells']) else 'consistent'}
                for f in found]}, ensure_ascii=False))
        return FakeRunner(env_source=(LIBRARY / 'env.py').read_text().replace(*IGNORE_ACTIVE_ENV),
                          oracle_source=(LIBRARY / 'oracle.py').read_text().replace(*IGNORE_ACTIVE_ORACLE),
                          perturbation_auditor=auditor, repair=repair)

    def flagged(self, status):
        return [d['world'] for d in status['stages']['perturbation_audit']['decisions'] if d['verdict'] != 'consistent']

    def test_rule_both_sides_ignore_is_repaired(self):
        def repair(folder):
            shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
            shutil.copy(LIBRARY / 'oracle.py', folder / 'oracle.py')
        status = self.build(self.ignore_active_runner('should_change', repair))
        self.assertTrue(self.flagged(status))
        self.assertTrue(status['stages']['perturbation_audit']['repaired'])
        self.assertTrue(status['training_ready'], status)
        self.assertNotIn('excluded_worlds', status['stages']['perturbation_audit'])

    def test_unrepaired_rule_excludes_its_worlds_from_training(self):
        status = self.build(self.ignore_active_runner('should_change', repair=lambda folder: None))
        self.assertTrue(status['training_ready'], status)
        flagged = set(self.flagged(status))
        self.assertTrue(flagged)
        self.assertFalse(flagged & set(bundle.load_bundle(status['bundle'])['valid_worlds']))
        self.assertTrue(any(d['verdict'] == 'should_change' for d in status['spec_findings']))

    def test_ambiguous_perturbation_is_excluded_without_repair(self):
        runner = self.ignore_active_runner('ambiguous')
        status = self.build(runner)
        self.assertNotIn('repair', runner.roles)
        self.assertTrue(status['training_ready'], status)
        self.assertFalse(set(self.flagged(status)) & set(bundle.load_bundle(status['bundle'])['valid_worlds']))

    def test_perturbation_audit_quote_must_exist(self):
        status = self.build(self.ignore_active_runner('should_change', quote='not in the brief at all'))
        self.assertEqual(status['failure_class'], 'audit_invalid')

    def test_silent_timeouts_are_infrastructure(self):
        from env_factory_v3.authoring.codex import classify
        self.assertEqual(classify('timeout', None, {'output_tokens': 0}, False), 'infrastructure_error')
        self.assertEqual(classify('timeout', None, {'output_tokens': 900}, False), 'timeout')
        self.assertEqual(classify('exited', 1, {'output_tokens': 0}, False), 'infrastructure_error')
        self.assertEqual(classify('exited', 0, {'output_tokens': 50}, False), 'exited')

    def test_infrastructure_errors_are_classified(self):
        status = self.build(FakeRunner(fail_role='oracle'))
        self.assertEqual(status['failure_class'], 'infrastructure')
        self.assertEqual(status['repairs_used'], 0)

    def test_checks_module_reports(self):
        folder = self.root / 'work'
        request = library_request()
        spec.materialize(request, folder, json.loads((LIBRARY / 'answer_schema.json').read_text()))
        shutil.copy(LIBRARY / 'env.py', folder / 'env.py')
        shutil.copy(LIBRARY / 'oracle.py', folder / 'oracle.py')
        for kind in ('schema', 'env', 'oracle', 'diff'):
            self.assertTrue(checks.run(kind, folder)['passed'], kind)


if __name__ == '__main__':
    unittest.main()
