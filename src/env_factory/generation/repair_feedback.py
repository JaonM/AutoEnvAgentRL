"""Evidence-scoped repair instructions; never override an acceptance verdict."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _location(bundle, path):
    """Resolve only concrete dotted/indexed data paths, without executing text."""
    result = {'path': path, 'resolved': False}
    if not isinstance(path, str):
        return result
    normalized = path.removeprefix('$.')
    if not re.fullmatch(r'[^.\[\]]+(?:\.[^.\[\]]+|\[(?:\d+|\*)\])*', normalized):
        return result
    tokens = re.findall(r'(?:^|\.)([^.\[\]]+)|\[(\d+|\*)\]', normalized)
    def resolve(current, remaining):
        if not remaining:
            return current
        key, index = remaining[0]
        if key:
            if isinstance(current, list) and re.fullmatch(r'0|[1-9][0-9]*', key):
                return resolve(current[int(key)], remaining[1:])
            if not isinstance(current, dict) or key not in current:
                raise KeyError(key)
            return resolve(current[key], remaining[1:])
        if not isinstance(current, list):
            raise KeyError(index)
        if index == '*':
            if not current:
                raise KeyError('empty wildcard')
            return [resolve(child, remaining[1:]) for child in current]
        return resolve(current[int(index)], remaining[1:])
    try:
        current = resolve(bundle, tokens)
    except (KeyError, IndexError):
        return result
    result.update(resolved=True, value=current)
    return result


def repair_feedback(error: Exception, *, root: Path, attempt_root: Path) -> dict:
    message = str(error)
    prefix, separator, tail = message.partition(':')
    try:
        parsed = json.loads(tail) if separator else None
    except ValueError:
        parsed = None
    candidate = _load(attempt_root / 'semantic_review/candidate.json') or {}
    source_path = attempt_root / 'source.json'
    if not source_path.is_file():
        source_path = root / 'authoring/source.json'
    source = _load(source_path)
    if not candidate:
        candidate = {'source': source}
    issues = []
    if prefix == 'SOURCE_SEMANTIC_REVIEW_FAILED' and isinstance(parsed, list):
        for finding in parsed:
            if not isinstance(finding, dict):
                continue
            from .review_replay import observe_counterexample
            locations = [_location(candidate, path) for path in finding.get('source_paths', [])]
            issues.append({
                'code': finding.get('code', prefix), 'evidence_level': 'model_review_claim',
                'claim': finding.get('reason', ''), 'locations': locations,
                'counterexample': finding.get('counterexample'),
                'execution_evidence': finding.get('execution_evidence'),
                'counterexample_observations': observe_counterexample(finding, candidate),
                'suggested_change': finding.get('repair', ''),
                'validate_before_edit': [
                    'Check the cited values against the compiled tool/reward implementation and mandatory interaction order.',
                    'Use actual declared final disclosures; an unimplemented alternative user reply is not an executable counterexample.',
                    'For a reward claim, evaluate the complete reward including prerequisites; a raw component alone is insufficient.',
                    'A missing cited path needs investigation; it does not automatically prove or disprove the business defect.',
                    'For disclosure issues, inspect related_review_obligations across public rubrics, criteria, schema and reference questions. Preserve legitimate unselected option lists; lexical hits alone do not prove leakage.'],
                'closure': 'Original goal preserved; focused counterexample resolved; full parent execution and independent review pass.'})
    elif prefix == 'AGENT_SCENARIO_FAILED' and isinstance(parsed, dict):
        diagnostic = parsed.get('diagnostic', {})
        issues.append({
            'code': parsed.get('code', prefix), 'evidence_level': 'execution_observation',
            'scenario_id': parsed.get('scenario_id'), 'claim': parsed.get('message'),
            'observed_failure': parsed.get('details', {}),
            'answer_targets': diagnostic.get('answer_targets', []),
            'state_predicates': diagnostic.get('state_predicates', []),
            'metric_state_predicates': diagnostic.get('metric_state_predicates', []),
            'gate_diagnostics': diagnostic.get('gate_diagnostics', {}),
            'preservation_violations': diagnostic.get('preservation_violations', []),
            'components': diagnostic.get('gated_components', {}),
            'last_evaluated_components': diagnostic.get('last_evaluated_components', {}),
            'model_evidence': diagnostic.get('model_evidence', []),
            'suggested_change': 'Trace the failing assertion to its query, binding or interaction prerequisite. Correct that contract without weakening the assertion.',
            'closure': 'The named scenario passes its original assertion; related negative cases still fail the policy as intended.'})
    elif prefix == 'REWARD_GOAL_CONFLICT' and isinstance(parsed, list):
        for conflict in parsed:
            if not isinstance(conflict, dict):
                continue
            issues.append({'code': prefix, 'evidence_level': 'validator_diagnostic',
                'metric_id': conflict.get('metric_id'), 'field': conflict.get('field'),
                'observed_failure': conflict,
                'locations': [_location(candidate, conflict[key])
                              for key in ('reward_path', 'goal_path') if key in conflict],
                'suggested_change': 'Resolve the intended value from the public requirement and query evidence. '
                    'Define the state target once in business_contract and derive both goal and reward from it; '
                    'do not choose either conflicting value merely to satisfy the validator.',
                'closure': 'Shared goal and reward agree; the complete reference and relevant counterexamples pass.'})
    elif prefix == 'AGENT_PREFLIGHT_FAILED' and isinstance(parsed, dict):
        for evidence in parsed.get('ablations', []):
            issues.append({'code': prefix, 'evidence_level': 'execution_observation',
                           'scenario_id': evidence.get('scenario_id'),
                           'check': evidence.get('check'), 'observed_failure': evidence,
                           'claim': evidence.get('message'),
                           'suggested_change': 'Repair the redundant trajectory or missing business dependency; preserve required recovery mechanics and the reward threshold.',
                           'closure': 'Complete reference still succeeds and removing the required logical request loses reward.'})
    elif prefix == 'SEMANTIC_CALIBRATION_FAILED' and isinstance(parsed, dict):
        for case in parsed.get('cases', []):
            if case.get('passed'):
                continue
            issues.append({
                'code': prefix, 'evidence_level': 'observed_judge_disagreement',
                'metric_id': case.get('metric_id'), 'case_index': case.get('index'),
                'reference_id': case.get('reference_id'),
                'failed_case_evidence': case.get('failed_case_evidence', {}),
                'expected_label': case.get('expected_label'), 'actual_score': case.get('actual_score'),
                'judgment_context_hashes': case.get('judgment_context_hashes', []),
                'suggested_change': 'Check case facts and rubric against tool evidence. A judge label alone does not explain the cause; do not relabel a case merely to match it.',
                'closure': 'Correct positives and genuine negatives pass calibration under an unchanged business requirement.'})
    if not issues:
        issues.append({'code': prefix if separator else type(error).__name__,
                       'evidence_level': 'validator_diagnostic', 'claim': message,
                       'suggested_change': 'Locate the violated contract from this diagnostic and reproduce it before editing.',
                       'closure': 'Original validator passes and dependent execution checks remain valid.'})
    for issue in issues:
        identity = {key:issue[key] for key in ('code','scenario_id','metric_id','case_index','check') if key in issue}
        identity['paths'] = [loc['path'] for loc in issue.get('locations', [])]
        issue['issue_id'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    return {'version':'1.0', 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else None,
            'issues':issues,
            'related_review_obligations': {'evidence_level':'compiler_lexical_findings_require_semantic_review',
                'items':candidate.get('privacy_review_obligations', []),
                'policy':'Use these to check other surfaces affected by the same fact. Do not assume every lexical occurrence leaks the selected value.'},
            'preserve':['original public goal and explanations','user decisions and mandatory interactions',
                        'business data and distractors','reward assertions and independent acceptance gates'],
            'repair_workflow':['Resolve the cited evidence and reproduce the defect.',
                               'Make the smallest contract change that fixes the underlying cause.',
                               'Run structural preview; let the parent rerun execution, calibration and independent review.'],
            'acceptance_policy':'Diagnostic guidance only. No issue classification changes a verdict or grants readiness.'}
