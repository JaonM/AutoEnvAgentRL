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
    def visit(value, path):
        if isinstance(value, dict):
            finding = (value.get('consistent') is False or value.get('business_consistent') is False or
                       value.get('passed') is False and any(k in value for k in ('case', 'failure_evidence', 'kind')))
            if finding:
                # Deduplicate the same finding nested in verification and audit.
                key = identity(value)
                if key not in seen:
                    seen.add(key)
                    case = value.get('case')
                    reason = value.get('reason') or value.get('error') or (case.get('reason') if isinstance(case, dict) else None) or 'Failed execution evidence'
                    issues.append({'id': key, 'evidence_path': path, 'reason': str(reason)[:800]})
            for k, child in value.items():
                if k not in {'original_request', 'previous_source', 'source', 'contract', 'literal'}:
                    visit(child, path + '/' + k.replace('~','~0').replace('/','~1'))
        elif isinstance(value, list):
            for i, child in enumerate(value):visit(child, path + '/' + str(i))
    visit(feedback, '/feedback')
    recovery = feedback.get('author_recovery', {})
    resume = bool(recovery) and feedback.get('failed_phase', 'generation') == 'generation'
    if not issues and feedback.get('error') and not resume:
        issues.append({'id': identity({'error': feedback['error'], 'phase': feedback.get('failed_phase')}),
                       'evidence_path': '/feedback', 'reason': str(feedback['error'])[:800]})
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
