"""Grade facts against current authoritative targets; judge only explanations."""
from __future__ import annotations

import copy
from collections import Counter

from .contract import require, validate_value
from .expressions import ContractError, canonical, evaluate, typed_equal


def equivalent(actual, expected, comparison):
    if comparison == 'unordered':
        if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
            return False
        return Counter(_key(item) for item in actual) == Counter(_key(item) for item in expected)
    return typed_equal(actual, expected)


def _key(value):
    if type(value) in {int, float}:
        return ('number', value.as_integer_ratio())
    if isinstance(value, dict):
        return ('object', tuple((key, _key(item)) for key, item in sorted(value.items())))
    if isinstance(value, list):
        return ('list', tuple(_key(item) for item in value))
    return (type(value).__name__, value)


def complete_retrieval(sandbox, tool_id):
    records = [record for record in sandbox.trace if record.get('tool') == tool_id and record.get('status') == 'ok']
    groups = {}
    for record in records:
        page = record['result']
        groups.setdefault((page['query_identity'], page['state_version']), []).append(page)
    for pages in groups.values():
        # An explicit traversal of one query; repeated pages never fill gaps.
        by_offset = {page['offset']: page for page in pages}
        offset = 0
        total = by_offset.get(0, {}).get('total')
        if total is None:
            continue
        while offset in by_offset:
            page = by_offset[offset]
            if page['total'] != total:
                break
            if page['next_cursor'] is None:
                if offset + len(page['items']) == total:
                    return True
                break
            if not page['items']:
                break
            offset += len(page['items'])
    return False


def grade(sandbox, answer, *, judge=None):
    require(not sandbox.closed, 'episode already finalized')
    require(isinstance(answer, dict), 'final answer must be an object')
    fields = sandbox.source['answers']
    explanations = sandbox.source.get('explanations', [])
    require(set(answer) == {field['id'] for field in [*fields, *explanations]}, 'final answer fields mismatch')
    context = sandbox.context(private=True)
    components, targets = {}, {}
    for field in fields:
        try:
            validate_value(answer[field['id']], field['schema'], field['id'])
            expected = evaluate(field['expected'], context)
            validate_value(expected, field['schema'], 'target')
            targets[field['id']] = expected
            passed = equivalent(answer[field['id']], expected, field['comparison'])
            components[field['id']] = {'passed': passed, 'weight': field['weight']}
        except ContractError as exc:
            components[field['id']] = {'passed': False, 'weight': field['weight'], 'error': str(exc)}
    semantic_evidence = {}
    for explanation in explanations:
        require(isinstance(answer[explanation['id']], str), 'explanation must be a string')
        require(judge is not None, 'native semantic judge required; no mock or fallback in qualification')
        payload = {'contract_identity': sandbox.compiled['identity'], 'task': sandbox.compiled['public'], 'field': explanation,
                   'text': answer[explanation['id']],
                   'authoritative_facts': {key: targets[key] for key in explanation['context_fields'] if key in targets},
                   'unresolved_fields': [key for key in explanation['context_fields'] if key not in targets],
                   'user_turns': [turn for turn in sandbox.trace if turn['kind'] == 'user']}
        result = judge(payload)
        require(set(result) == {'criteria', 'provenance'}, 'invalid semantic judgment')
        require(set(result['criteria']) == {c['id'] for c in explanation['criteria']}
                and all(type(v) is bool for v in result['criteria'].values()), 'criterion identity mismatch')
        components[explanation['id']] = {'passed': all(result['criteria'].values()), 'weight': explanation['weight']}
        semantic_evidence[explanation['id']] = result
    used = {turn['tool'] for turn in sandbox.trace if turn.get('status') == 'ok'}
    branch_checks = []
    for branch in sandbox.source['success']:
        try:
            active = evaluate(branch['when'], context)
            require(type(active) is bool, 'branch condition must be boolean')
        except ContractError:
            active = False
        passed = active and set(branch['required_tools']) <= used and set(branch['required_stages']) <= set(sandbox.user.completed)
        if passed:
            passed = all(not sandbox.tools[tool].get('complete_retrieval') or complete_retrieval(sandbox, tool)
                         for tool in branch['required_tools'])
        branch_checks.append({'id': branch['id'], 'active': active, 'passed': bool(passed)})
    safety = not sandbox.violations
    invariant_checks = [evaluate(item['predicate'], context) is True for item in sandbox.source.get('invariants', [])]
    goal_checks = []
    for goal in sandbox.source.get('goals', []):
        try:
            passed = evaluate(goal['predicate'], context) is True
        except ContractError:
            passed = False
        goal_checks.append({'id': goal['id'], 'passed': passed})
    process = any(branch['passed'] for branch in branch_checks) and all(invariant_checks) and all(g['passed'] for g in goal_checks)
    category_valid = (sandbox.source['category'] != 'simple_agentic' or sandbox.business_attempts == 1)
    process = process and category_valid
    field_reward = sum(item['weight'] for item in components.values() if item['passed'])
    # Evidence and safety are mandatory; correct guesses cannot earn outcome credit.
    reward = field_reward if process and safety else 0.0
    success = process and safety and all(item['passed'] for item in components.values())
    sandbox.closed = True
    return {'reward': reward, 'success': success, 'components': components,
            'process': branch_checks, 'safety': safety, 'violations': copy.deepcopy(sandbox.violations),
            'invariants': invariant_checks, 'goals': goal_checks, 'category_valid': category_valid,
            'business_attempts': sandbox.business_attempts, 'semantic_evidence': semantic_evidence}
