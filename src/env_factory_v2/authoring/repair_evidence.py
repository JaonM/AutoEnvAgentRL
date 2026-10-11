"""Bounded repair index over immutable full evidence; no semantic verdict rewriting."""
import copy
import hashlib
from env_factory_v2.contracts.expressions import canonical

PROMPT_LIMIT = 262144


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def repair_context(request, feedback, previous_source):
    feedback = copy.deepcopy(feedback or {})
    issues, seen = [], set()
    # Only current report fields carry verdicts. Audit/recovery archives and
    # execution payloads are data, not new repair obligations.
    children = {'records', 'checks', 'requirement_review', 'reviews',
                'obligation_reviews', 'communication_coverage', 'independent_review'}
    def visit(value, path):
        if isinstance(value, dict):
            if value.get('passed') is True:
                return
            finding = (value.get('consistent') is False or value.get('business_consistent') is False or
                       value.get('passed') is False and any(k in value for k in ('case', 'failure_evidence', 'kind')))
            if finding:
                key = identity(value)
                if key not in seen:
                    seen.add(key)
                    case = value.get('case')
                    reason = value.get('reason') or value.get('error') or (case.get('reason') if isinstance(case, dict) else None) or 'Failed execution evidence'
                    issue = {'id': key, 'evidence_path': path, 'reason': str(reason)[:800]}
                    if isinstance(case, dict):
                        issue['case_id'] = case.get('id')
                        issue['expect_success'] = case.get('expect_success')
                    first = value.get('first')
                    if isinstance(first, dict):
                        issue['execution_error'] = str(first.get('error') or '')[:800]
                        outcome = first.get('outcome', {})
                        issue['observed_success'] = outcome.get('success') if isinstance(outcome, dict) else None
                    issues.append(issue)
            for k in sorted(children & value.keys()):
                if k == 'independent_review' and 'independent_challenge' in feedback:
                    continue  # The top-level current challenge is authoritative.
                visit(value[k], path + '/' + k)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + '/' + str(i))
    for name in ('verification', 'independent_challenge', 'recipe_preflight'):
        if name in feedback:
            visit(feedback[name], '/feedback/' + name)
    # Also accept a standalone current report (used by direct author callers).
    visit(feedback, '/feedback')
    recovery = feedback.get('author_recovery', {})
    resume = bool(recovery) and feedback.get('failed_phase', 'generation') == 'generation'
    if not issues and feedback.get('error') and not resume:
        issues.append({'id': identity({'error': feedback['error'], 'phase': feedback.get('failed_phase')}),
                       'evidence_path': '/feedback', 'reason': str(feedback['error'])[:800]})
    lineage = recovery.get('repair_state') if resume and recovery.get('source_identity') == identity(previous_source) else None
    if lineage and lineage.get('mode') == 'repair':
        return {'request': copy.deepcopy(request), 'feedback': feedback, 'repair': copy.deepcopy(lineage)}
    mode = 'repair' if previous_source is not None and issues else 'resume' if previous_source is not None else 'create'
    return {'request': copy.deepcopy(request), 'feedback': feedback,
            'repair': {'mode': mode, 'issues': issues, 'initial_identity': identity(previous_source),
                       'scope': 'proposed fixes require subsequent independent validation'}}


def prompt_context(context):
    request = context['request']
    # Full original input is available to read_context, never silently truncated.
    return {'original_request': request if len(canonical(request)) <= 65536 else {
                'read_context_path': '/request', 'characters': len(canonical(request)),
                'sha256': identity(request), 'keys': sorted(request)},
            'repair': repair_index(context),
            'evidence': {'read_context_path': '/feedback', 'sha256': identity(context['feedback']),
                         'characters': len(canonical(context['feedback'])),
                         'keys': sorted(context['feedback'])}}


def repair_index(context):
    repair = context['repair']
    return {**repair, 'issues': repair['issues'][:16], 'issue_count': len(repair['issues']),
            'all_issues_path': '/repair/issues'}
