"""Versioned task replay with caller-declared contract amendments and full requalification.

Original tasks, openings, facts, implementations and reference outcomes are preserved.
Amendments may change interaction policy and clarify outcome descriptions, not answer values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
from env_factory.core import spec as request_spec
import shutil

from env_factory.certify import bundle, policy_gate, sufficiency
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import contracts
from env_factory.runtime import run_reference
from recheck_world_tasks import save
from sample_world_tasks import natural


def amend(request, changes):
    request = copy.deepcopy(request)
    marker = '\n\nPUBLIC BUSINESS CONTRACT (authoritative; preserve policy and API units):\n'
    prefix, sep, embedded = request['business_rules'].partition(marker)
    contract = copy.deepcopy(request['public_contract'])
    if not sep:
        raise ValueError('embedded public contract is missing')
    embedded_contract, end = json.JSONDecoder().raw_decode(embedded)
    suffix = embedded[end:]
    if embedded_contract != contract:
        raise ValueError('embedded public contract does not match its declared source')
    for replacement in changes.get('policy_replacements', []):
        old, new = replacement['old'], replacement['new']
        if contract['policy'].count(old) != 1:
            raise ValueError('policy replacement must match exactly once')
        contract['policy'] = contract['policy'].replace(old, new)
        prefix = prefix.replace(old, new)
    for key, value in changes.get('constraints', {}).items():
        if value is None:
            contract.setdefault('constraints', {}).pop(key, None)
        else:
            contract.setdefault('constraints', {})[key] = value
    request['business_rules'] = prefix
    contracts.attach(request, contract)
    request['business_rules'] += suffix
    for change in changes.get('outcome_descriptions', []):
        for process in change['processes']:
            field = request['goal_schema'][process]['outcome']['properties'][change['field']]
            field['description'] = change['description']
    return request


def prepare_bundle(source, destination, changes):
    source, destination = Path(source), Path(destination)
    original = bundle.load_bundle(source, require_current_platform=False)
    identity = {'source_files': original['manifest']['files'], 'changes': changes,
                'platform': bundle.platform_identity(), 'preparer': bundle.sha256(__file__)}
    receipt = destination.parent / 'preparation.json'
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached['identity'] != identity:
            raise ValueError('preparation inputs changed; use a new output')
        if cached['passed']:
            bundle.load_bundle(destination)
        return cached
    request = amend(original['request'], changes)
    # Task schemas can deliberately omit constant leaves. Preserve those omissions,
    # all enums, types and grading metadata; only amend existing descriptions.
    schema = copy.deepcopy(original['schema'])
    for index, goal in enumerate(request['user_facts']['goals'], 1):
        properties = schema['fields'][f'goal_{index}']['properties']
        for change in changes.get('outcome_descriptions', []):
            if goal['process'] in change['processes'] and change['field'] in properties:
                properties[change['field']]['description'] = change['description']
    candidate = {**original, 'request': request, 'schema': schema}
    problems = contracts.interface_problems(candidate['env'].make_tools(), request)
    if problems:
        raise ValueError(str(problems))
    report = bundle.qualify_loaded(candidate)
    before = run_reference(original['env'], original['request']['tables'], original['request']['user_facts'],
                           start=original['request']['time']['start'])
    after = run_reference(candidate['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    if before != after:
        raise ValueError('contract amendments changed reference execution')
    record = {'identity': identity, 'passed': report['passed'], 'reference_unchanged': True,
              'source_bundle': str(source.resolve()), 'training_ready': False}
    save(destination.parent / 'qualification.json', report)
    if report['passed']:
        stage = destination.parent / 'prepared_source'
        stage.mkdir(parents=True, exist_ok=True)
        for name in ('env.py', 'verifier.py'):
            shutil.copyfile(source / name, stage / name)
        request_spec.write(request, stage / 'request.json')
        save(stage / 'answer_schema.json', schema)
        bundle.export(stage, stage / 'request.json', report, destination, status='qualified')
    save(receipt, record)
    return record


def model_identity():
    return {role: {'base_url': client.base_url, 'model': client.model}
            for role in ('LLM', 'USER_SIM', 'JUDGE') for client in [ChatClient(role)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--amendments', required=True)
    parser.add_argument('--stage', choices=('prepare', 'evaluate'), required=True)
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents:
        parser.error('use a separate versioned output')
    load_env_file()
    plan = json.loads((source / 'plan.json').read_text())
    amendments = json.loads(Path(args.amendments).read_text())
    identity = {'source_plan': plan, 'amendments': amendments, 'platform': bundle.platform_identity(),
                'evaluator': policy_gate.evaluation_identity(), 'sufficiency': bundle.sha256(sufficiency.__file__),
                'runner': bundle.sha256(__file__), 'natural_runner': bundle.sha256(natural.__code__.co_filename),
                'models': model_identity()}
    frozen = output / 'identity.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('replay configuration changed; use a new output')
    save(frozen, identity)
    if args.stage == 'prepare':
        for domain in plan['domains']:
            target = output / domain['name'] / 'world' / 'bundle'
            record = prepare_bundle(domain['bundle'], target, amendments[domain['name']])
            if not record['passed']:
                raise ValueError(f"world requalification failed: {domain['name']}")
            domain['bundle'] = str(target)
    else:
        plan = json.loads((output / 'plan.json').read_text())
    jobs = [(d, a) for d in plan['domains'] for a in d['assignments']]

    def work(job):
        domain, assignment = job
        index = assignment['index']
        folder = output / domain['name'] / 'tasks' / f't{index:03d}'
        if args.stage == 'prepare':
            old = json.loads((source / domain['name'] / 'tasks' / f't{index:03d}' / 'pilot_result.json').read_text())
            record = prepare_bundle(old['bundle'], folder / 'bundle', amendments[domain['name']])
            row = {**old, 'bundle': str(folder / 'bundle'), 'passed': record['passed'],
                   'stage': 'structural_only', 'original_passed': old['passed'],
                   'successful_demonstration_eligible': False}
            save(folder / 'pilot_result.json', row)
            return {'domain': domain['name'], 'index': index, 'passed': record['passed']}
        verdict = folder / 'requalification.json'
        if not verdict.exists():
            row = json.loads((folder / 'pilot_result.json').read_text())
            if not row['passed']:
                return {'domain': domain['name'], 'index': index, 'passed': False, 'stage': 'structural'}
            loaded = bundle.load_bundle(row['bundle'])
            check = sufficiency.check(loaded, ChatClient('LLM'))
            save(folder / 'sufficiency.json', check)
            row.update(passed=check['passed'], stage='gates' if check['passed'] else 'sufficiency')
            save(verdict, row)
        row = json.loads(verdict.read_text())
        result = natural(domain, assignment, output, diagnostic=not row['passed'])
        return {k: v for k, v in result.items() if k not in ('raw_episodes', 'bounded_episode', 'bounded_generation')}

    rows = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            rows.append(future.result())
            report = {'completed': len(rows), 'total': len(jobs), 'stage': args.stage, 'results': rows,
                      'training_ready': False}
            save(output / f'{args.stage}_summary.json', report)
            print(json.dumps({**{k: report[k] for k in ('completed', 'total', 'stage')},
                              'last': {k: v for k, v in rows[-1].items() if k in (
                                  'domain', 'index', 'passed', 'raw_successes', 'raw_policy_successes')}}), flush=True)
    if args.stage == 'prepare':
        save(output / 'plan.json', plan)


if __name__ == '__main__':
    main()
