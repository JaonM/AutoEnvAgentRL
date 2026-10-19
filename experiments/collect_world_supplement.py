"""Bounded assisted collection for fixed pilot tasks lacking an admitted demonstration.

Never edits original grades, reference answers or executed transcripts. Teacher
coaching is task-independent; all attempts and independent audits are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from env_factory.core import spec as request_spec

from env_factory.certify import bundle, policy_gate, trajectory
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import delivery_teacher, dialogue_audit
from env_factory.runtime import is_write, tool_specs
from env_factory.serving import server
from recheck_world_tasks import save
from trajectory_dedup import deduplicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, help='reuse already admitted tasks from an earlier bounded collection')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('attempt budget must be positive')
    load_env_file()
    audit = json.loads((args.audit / 'summary.json').read_text())
    if audit['completed'] != audit['total'] or audit['infrastructure_errors']:
        raise ValueError('finish source quality audit before selecting supplementation tasks')
    missing = [r for r in audit['results'] if not r.get('selected_sources')]
    total = len(missing)
    previous = json.loads((args.previous / 'summary.json').read_text()) if args.previous else None
    carried = [{**r, 'carried_from': str(args.previous.resolve())} for r in previous['results'] if r['passed']] if previous else []
    covered = {(r['domain'], r['index']) for r in carried}
    missing = [r for r in missing if (r['domain'], r['index']) not in covered]
    plan = json.loads((args.source / 'plan.json').read_text())
    assignments = {(d['name'], a['index']): a for d in plan['domains'] for a in d['assignments']}
    clients = {role: ChatClient(role) for role in ('LLM', 'USER_SIM', 'JUDGE')}
    identity = {'source_audit_sha256': bundle.sha256(args.audit / 'summary.json'),
                'previous_summary_sha256': bundle.sha256(args.previous / 'summary.json') if args.previous else None,
                'platform': bundle.platform_identity(), 'evaluator': policy_gate.evaluation_identity(),
                'code': {name: bundle.sha256(path) for name, path in (
                    ('runner', __file__), ('coaching', delivery_teacher.__file__), ('audit', dialogue_audit.__file__))},
                'attempts': args.attempts,
                'models': {role: {'model': c.model, 'endpoint': c.base_url} for role, c in clients.items()}}
    frozen = args.output / 'identity.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('collection configuration changed; use a new output')
    save(frozen, identity)

    def one(task):
        domain, index = task['domain'], task['index']
        assignment = assignments[domain, index]
        source = args.source / domain / 'tasks' / f't{index:03d}'
        if not json.loads((source / 'requalification.json').read_text())['passed']:
            raise ValueError('source task lacks policy sufficiency')
        folder = args.output / domain / f't{index:03d}'
        final = folder / 'selection.json'
        if final.exists():
            return json.loads(final.read_text())
        target = folder / 'bundle'
        if not target.exists():
            loaded = bundle.load_bundle(source / 'bundle', require_current_platform=False)
            report = bundle.qualify_loaded(loaded)
            save(folder / 'qualification.json', report)
            if not report['passed']:
                return {'domain': domain, 'index': index, 'passed': False, 'stage': 'requalification'}
            bundle.export(source / 'bundle', source / 'bundle/request.json', report, target, status='qualified')
        loaded = bundle.load_bundle(target)
        # Freeze all business inputs; only platform qualification evidence can differ.
        if loaded['request'] != request_spec.load(source / 'bundle/request.json'):
            raise ValueError('business request changed during supplementation')
        for name in ('answer_schema.json', 'env.py', 'verifier.py'):
            if (target / name).read_bytes() != (source / 'bundle' / name).read_bytes():
                raise ValueError('business input changed during supplementation')
        rules = trajectory.constraints(loaded['request'])
        service = bundle.service(loaded, user_mode='natural', user_client=ChatClient('USER_SIM'), perturbed=0)
        httpd = server.start(service)
        functions = loaded['env'].make_tools()
        write_names = {name for name, fn in functions.items() if is_write(fn)}
        rows, candidates = [], []
        try:
            for attempt in range(args.attempts):
                path = folder / f'attempt-{attempt:02d}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                else:
                    teacher = delivery_teacher.DeliveryTeacher(ChatClient('LLM'))
                    row = {'accepted': False}
                    try:
                        episode = trajectory.qualify(policy_gate.play(
                            f'http://127.0.0.1:{httpd.server_address[1]}', teacher, 'original',
                            loaded['request']['category'], max_turns=30, answer_tool=False,
                            trajectory_constraints=rules, consent='plain'), rules)
                        successful = set()
                        for message in episode.get('messages', []):
                            if message.get('role') == 'tool':
                                try:
                                    if json.loads(message['content']).get('ok') is True:
                                        successful.add(message['name'])
                                except (ValueError, AttributeError):
                                    pass
                        required = set(assignment['requirements'].get('required_tools', [])) <= successful
                        row.update(episode=episode, required_actions_passed=required)
                        reviewed = dialogue_audit.audit(ChatClient('JUDGE'), loaded['request'], episode,
                                                         tool_specs(functions, consent=False), write_names)
                        row.update(audit=reviewed, accepted=bool(
                            episode['policy_aware_success'] and reviewed['passed'] and required))
                    except Exception as exc:
                        row.update(infrastructure_error=f'{type(exc).__name__}: {exc}'[:500])
                    row.update(attempt=attempt, generation=teacher.report(), training_ready=False)
                    save(path, row)
                rows.append({'attempt': attempt, 'accepted': row['accepted'],
                             'business_success': bool(row.get('episode', {}).get('success')),
                             'audit_errors': row.get('audit', {}).get('errors', []),
                             'infrastructure_error': row.get('infrastructure_error')})
                if row['accepted']:
                    candidates.append({**row['episode'], 'id': loaded['request']['id'], 'domain': domain,
                                       'world': 'original', 'consent': 'plain', 'source_episode': attempt,
                                       'platform_identity': bundle.platform_identity(),
                                       'evaluation_identity': policy_gate.evaluation_identity()})
                    break  # one demonstration per previously uncovered task is the target
        finally:
            httpd.shutdown()
            httpd.server_close()
        selected, dedup = deduplicate(candidates, .85)
        result = {'domain': domain, 'index': index, 'attempts': rows, 'passed': bool(selected),
                  'selected_attempts': [e['source_episode'] for e in selected], 'dedup': dedup,
                  'training_ready': False, 'scope': 'assisted demonstration collection only'}
        save(final, result)
        return result

    rows = carried[:]
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(one, task) for task in missing]):
            rows.append(future.result())
            report = {'completed': len(rows), 'total': total, 'tasks_with_candidate': sum(r['passed'] for r in rows),
                      'results': rows, 'training_ready': False, 'raw_evaluation_unchanged': True}
            save(args.output / 'summary.json', report)
            print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
