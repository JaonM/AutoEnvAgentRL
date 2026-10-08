"""Read-only deterministic observations for a reviewer's proposed data changes.

These observations do not adjudicate semantic claims or override reward gates.
"""
import copy

from .report_values import report_value

from ..sandbox_runtime import DeclarativeMetricEvaluator, SandboxError


def observe_counterexample(finding, bundle):
    from .review_scope import validate_review_scope
    example = finding.get('counterexample')
    if not isinstance(example, dict) or example.get('basis') != 'business_data':
        return {'status': 'not_applicable'}
    from .semantic_review import SemanticReviewUnavailable
    try:
        validate_review_scope({'findings': [finding]}, bundle)
    except SemanticReviewUnavailable as exc:
        return {'status': 'invalid_counterexample', 'reason': str(exc)}
    import json
    initial = {table['table_name']: copy.deepcopy(table['rows'])
               for table in bundle['source']['tables']}
    mutated = copy.deepcopy(initial)
    for change in example['data_mutations']:
        mutated[change['table']][change['row_index']][change['column']] = json.loads(change['value_json'])
    observations = []
    for rule in bundle.get('compiled_reward_rules', []):
        if rule.get('operator') != 'value_targets':
            continue
        for target in rule.get('expected', {}).get('targets', []):
            item = {'metric_id': rule.get('metric_id'), 'field': target['key']}
            try:
                for label, state in [('original_target', initial), ('mutated_target', mutated)]:
                    item[label] = DeclarativeMetricEvaluator._value_expression(target['expression'], state)
                item['resolved'] = item['original_target'] is not None and item['mutated_target'] is not None
            except SandboxError as exc:
                item.update(resolved=False, error=str(exc))
            observations.append(item)
    return {'status': 'observed', 'scope': 'value_targets only; no trajectory, semantic metric or full reward replay',
            'verdict_override': False, 'targets': report_value(observations)}


def baseline_reward_observations(bundle):
    """Expose executed lookup selectors, without claiming post-action success."""
    from ..sandbox_runtime import ExpressionBusinessState
    rows = {table['table_name']: copy.deepcopy(table['rows'])
            for table in bundle.get('source', {}).get('tables', [])}
    state = ExpressionBusinessState(rows, rows)
    observations = []
    def walk(node, path, current, found):
        if isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f'{path}[{index}]', current, found)
        elif isinstance(node, dict):
            if set(node) == {'literal'}:
                return
            if set(node) == {'at_tool'}:
                found.append({'path': path, 'tool': node['at_tool']['name'],
                              'reason': 'reward_checkpoint_missing_before_execution'})
                return
            if set(node) == {'initial'}:
                walk(node['initial'], path+'.initial', ExpressionBusinessState(rows, rows), found)
                return
            for key, value in node.items():
                walk(value, path+'.'+key, current, found)
            if set(node) == {'lookup'}:
                lookup = node['lookup']
                where = DeclarativeMetricEvaluator._numeric_where(lookup['where'], current, depth=0)
                matches = [row for row in current.get(lookup['table'], [])
                           if DeclarativeMetricEvaluator._where_matches(row, where)]
                found.append({'path':path, 'table':lookup['table'], 'field':lookup['field'],
                              'resolved_where':where, 'matched_row_count':len(matches),
                              'matched_values':[row.get(lookup['field']) for row in matches[:8]]})
    for rule in bundle.get('compiled_reward_rules', []):
        if rule.get('operator') != 'value_targets':
            continue
        for target in rule.get('expected', {}).get('targets', []):
            item = {'metric_id':rule.get('metric_id'), 'field':target['key'], 'lookups':[]}
            try:
                item['value'] = DeclarativeMetricEvaluator._value_expression(target['expression'], state)
                walk(target['expression'], '$', state, item['lookups'])
            except SandboxError as exc:
                item['error'] = str(exc)
            observations.append(item)
    return {'scope':'Initial business snapshot only. Writes may change these values; this is not a success verdict.',
            'targets':report_value(observations)}


def replay_review_findings(report, *, source, artifacts, request, root):
    """Execute bounded data counterexamples; never equate rejection with proof.

    The original reference may contain a stale answer after mutation. Report the
    observed result and retain pending adjudication rather than inventing a new
    correct answer or silently overriding the independent review.
    """
    import hashlib
    import json
    from .agent_authoring import compile_source
    root.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    semantic = any(metric.get('type') != 'rule-based' for metric in artifacts.get('metrics', []))
    for index, finding in enumerate(report.get('findings', [])):
        evidence = {'status': 'pending_adjudication', 'source_sha256': source_hash,
                    'rule': finding.get('code'), 'source_paths': finding.get('source_paths', []),
                    'verdict_override': False,
                    'replay_mode': 'recompiled_source_counterexample',
                    'contract_frozen': False}
        example = finding.get('counterexample', {})
        changes = example.get('data_mutations', [])
        if index >= 8 or not changes or semantic:
            evidence['reason'] = ('replay_limit' if index >= 8 else 'semantic_judge_required' if semantic
                                  else 'requires_authored_trajectory_or_public_contract_check')
            finding['execution_evidence'] = evidence
            continue
        candidate = copy.deepcopy(source)
        tables = {table['table_name']: table for table in candidate.get('tables', [])}
        case_root = root / f'finding-{index + 1}'
        try:
            for change in changes:
                tables[change['table']]['rows'][change['row_index']][change['column']] = json.loads(change['value_json'])
            result = compile_source(candidate, root=case_root, request=request, defer_privacy_review=True)
            evidence.update(observation='mutated_reference_and_ablations_passed',
                            verification=result['generation_pipeline']['verification'])
        except (ValueError, TypeError, KeyError, IndexError, SandboxError) as exc:
            evidence.update(observation='mutated_validation_rejected', error=str(exc))
            diagnostic = case_root / 'preflight_failure.json'
            if diagnostic.is_file():
                evidence['diagnostic'] = json.loads(diagnostic.read_text())
        evidence['replay_root'] = str(case_root)
        evidence['limitation'] = 'Source is recompiled after mutation; this is not a frozen-contract exploit replay. Reference and final answer are unchanged; acceptance/rejection alone does not establish semantic correctness.'
        finding['execution_evidence'] = report_value(evidence)
    (root / 'evidence.json').write_text(json.dumps(report_value(report.get('findings', [])), ensure_ascii=False, indent=2) + '\n')
    return report
