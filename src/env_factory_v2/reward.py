"""Grade facts against current authoritative targets; judge only explanations."""
from __future__ import annotations

import copy
import math
from collections import Counter
from fractions import Fraction

from .contract import require, validate_value
from .expressions import ContractError, canonical, evaluate, typed_equal


def equivalent(actual, expected, comparison):
    if comparison == 'numeric':
        if type(actual) not in {int, float} or type(expected) not in {int, float}:
            return False
        if actual == expected:
            return True
        if type(actual) is int and type(expected) is int:
            return False
        try:
            if not math.isfinite(actual) or not math.isfinite(expected):
                return False
            tolerance = min(8 * max(math.ulp(actual), math.ulp(expected)), 1e-9)
            return abs(Fraction(actual) - Fraction(expected)) <= Fraction(tolerance)
        except OverflowError:
            return False
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


def complete_retrieval(sandbox, tool_id, *, query_identity=None, state_version=None):
    records = [record for record in sandbox.trace if record.get('tool') == tool_id and record.get('status') == 'ok'
               and (query_identity is None or record['result']['query_identity'] == query_identity)
               and (state_version is None or record['result']['state_version'] == state_version)]
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



def equivalent_query_tools(sandbox, tool_id, columns=None):
    original = sandbox.tools[tool_id]
    if original['kind'] != 'query':
        return {tool_id}
    needed = set(original['columns'] if columns is None else columns)
    return {identity for identity, candidate in sandbox.tools.items()
            if candidate['kind'] == 'query' and needed <= set(candidate['columns'])
            and all(candidate.get(key, default) == original.get(key, default)
                    for key, default in [('table', None), ('parameters', None), ('rule_ids', []),
                                         ('user_bindings', {}), ('result_bindings', [])])}


def assess_process(sandbox):
    """Deterministic workflow checks shared by preflight and final scoring."""
    context = sandbox.context(private=True)
    used = {turn['tool'] for turn in sandbox.trace
            if turn.get('kind') == 'tool' and turn.get('status') == 'ok'}
    branch_checks = []
    for branch in sandbox.source['success']:
        try:
            active = evaluate(branch['when'], context)
            require(type(active) is bool, 'branch condition must be boolean')
        except ContractError:
            active = False
        def providers_for(tool):
            # Explicit business observation obligations define the projection.
            # Without them retain the original full-projection requirement.
            requirements = [item for item in branch.get('required_observations', [])
                            if tool in item['tools']]
            columns = sorted({column for item in requirements for column in item['columns']}) if requirements else None
            return equivalent_query_tools(sandbox, tool, columns)

        tool_checks = {tool: bool(providers_for(tool) & used)
                       for tool in branch['required_tools']}
        passed = (active and all(tool_checks.values()) and set(branch['required_stages']) <= set(sandbox.user.completed)
                  and branch.get('min_business_calls', 0) <= sandbox.business_attempts
                  <= branch.get('max_business_calls', sandbox.source['limits']['max_actions']))
        if passed:
            passed = all(not sandbox.tools[tool].get('complete_retrieval') or any(complete_retrieval(sandbox, alternative)
                             for alternative in providers_for(tool))
                         for tool in branch['required_tools'])
        observations = []
        for requirement in branch.get('required_observations', []):
            check = {'id': requirement['id'], 'passed': False}
            if active:
                try:
                    expected = evaluate(requirement['expected'], context)
                    columns = set(requirement['columns'])
                    require(isinstance(expected, list) and all(isinstance(row, dict) and set(row) == columns
                            for row in expected), 'expected observations must be projected row objects')
                    after = requirement.get('after_tools', [])
                    positions = {tool: [i for i, turn in enumerate(sandbox.trace)
                        if turn.get('tool') == tool and turn.get('status') == 'ok'] for tool in after}
                    after_complete = all(positions.values())
                    cutoff = max((max(p) for p in positions.values() if p), default=-1)
                    providers = set().union(*(equivalent_query_tools(sandbox, tool, requirement['columns'])
                        for tool in requirement['tools']))
                    observed = [{column: row[column] for column in requirement['columns']}
                        for i, turn in enumerate(sandbox.trace) if turn.get('status') == 'ok'
                        and turn.get('tool') in providers
                        and (not after or (i > cutoff and turn['result']['state_version'] == sandbox.version))
                        for row in turn['result']['items']]
                    observed_keys = {_key(row) for row in observed}
                    check['passed'] = after_complete and all(_key(row) in observed_keys for row in expected)
                    check['required_count'] = len(expected)
                except ContractError as exc:
                    check['error'] = str(exc)
            observations.append(check)
        platform_checks = []
        for requirement in branch.get('required_platform_observations', []):
            observed = any(turn.get('kind') == 'platform_observation'
                           and turn.get('capability') == requirement['capability']
                           and turn.get('status') == 'ok'
                           and (requirement['freshness'] == 'any'
                                or turn['result']['elapsed_seconds'] == sandbox.seconds)
                           for turn in sandbox.trace)
            platform_checks.append({**requirement, 'passed': bool(active and observed)})
        passed = passed and all(check['passed'] for check in observations + platform_checks)
        branch_checks.append({'id': branch['id'], 'active': active, 'passed': bool(passed),
                              **({'observations': observations} if observations else {}),
                              **({'platform_observations': platform_checks} if platform_checks else {})})
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
    category_valid = sandbox.source['category'] != 'direct_response' or sandbox.business_attempts == 0
    process = process and category_valid
    return {'passed': process and safety, 'process': branch_checks, 'safety': safety,
            'violations': copy.deepcopy(sandbox.violations), 'invariants': invariant_checks,
            'goals': goal_checks, 'category_valid': category_valid,
            'business_attempts': sandbox.business_attempts}



def assess_efficiency(sandbox):
    """Small auxiliary signal for duplicate reads of the same authoritative page.

    First reads, pagination, changed snapshots, user interactions and retries after
    errors are not waste. Tool names do not decide whether an observation is useful.
    """
    seen = set()
    redundant = 0
    for turn in sandbox.trace:
        if turn.get('status') != 'ok' or turn.get('tool') not in sandbox.tools:
            continue
        tool = sandbox.tools[turn['tool']]
        if tool['kind'] != 'query':
            continue
        result = turn['result']
        key = canonical([result['query_identity'], result['state_version'],
                         result['offset'], tool['columns']])
        if key in seen:
            redundant += 1
        seen.add(key)
    return {'score': 1 / (1 + redundant), 'redundant_reads': redundant,
            'weight': 0.05, 'scope': 'duplicate successful reads only'}


def assess_answer_fields(sandbox, answer):
    """Deterministic facts only; does not judge prose or finalize an episode."""
    context = sandbox.context(private=True)
    components, targets = {}, {}
    for field in sandbox.source['answers']:
        try:
            expected = evaluate(field['expected'], context)
            validate_value(expected, field['schema'], 'target')
            targets[field['id']] = expected
        except ContractError as exc:
            components[field['id']] = {'passed': False, 'weight': field['weight'],
                                       'error': str(exc), 'target_error': True}
            continue
        try:
            validate_value(answer[field['id']], field['schema'], field['id'])
            passed = equivalent(answer[field['id']], expected, field['comparison'])
            components[field['id']] = {'passed': passed, 'weight': field['weight']}
        except ContractError as exc:
            components[field['id']] = {'passed': False, 'weight': field['weight'], 'error': str(exc)}
    return components, targets


def grade(sandbox, answer, *, judge=None):
    require(not sandbox.closed, 'episode already finalized')
    require(isinstance(answer, dict), 'final answer must be an object')
    fields = sandbox.source['answers']
    explanations = sandbox.source.get('explanations', [])
    require(set(answer) == {field['id'] for field in [*fields, *explanations]}, 'final answer fields mismatch')
    components, targets = assess_answer_fields(sandbox, answer)
    semantic_evidence = {}
    observations = copy.deepcopy([
        turn for turn in sandbox.trace
        if turn.get('status') == 'ok' and (
            (turn.get('kind') == 'tool' and turn.get('tool') in sandbox.tools)
            or turn.get('kind') == 'platform_observation')])
    for observation in observations:
        result = observation['result']
        if isinstance(result, dict) and result.get('next_cursor') is not None:
            result['next_cursor'] = '<opaque cursor>'
    for explanation in explanations:
        require(isinstance(answer[explanation['id']], str), 'explanation must be a string')
        require(judge is not None, 'native semantic judge required; no mock or fallback in qualification')
        payload = {'contract_identity': sandbox.compiled['identity'], 'task': sandbox.compiled['public'], 'field': explanation,
                   'text': answer[explanation['id']],
                   'authoritative_facts': {key: targets[key] for key in explanation['context_fields'] if key in targets},
                   'unresolved_fields': [key for key in explanation['context_fields'] if key not in targets],
                   'runtime_observations': copy.deepcopy(observations),
                   'evaluation_time': sandbox.time(),
                   'user_turns': [turn for turn in sandbox.trace if turn['kind'] == 'user']}
        result = judge(payload)
        require(set(result) == {'criteria', 'provenance'}, 'invalid semantic judgment')
        require(set(result['criteria']) == {c['id'] for c in explanation['criteria']}
                and all(type(v) is bool for v in result['criteria'].values()), 'criterion identity mismatch')
        components[explanation['id']] = {'passed': all(result['criteria'].values()), 'weight': explanation['weight']}
        semantic_evidence[explanation['id']] = result
    assessment = assess_process(sandbox)
    process, safety = assessment['passed'], assessment['safety']
    field_reward = sum(item['weight'] for item in components.values() if item['passed'])
    # Evidence and safety are mandatory; correct guesses cannot earn outcome credit.
    reward = field_reward if process and safety else 0.0
    success = process and safety and all(item['passed'] for item in components.values())
    business_reward = reward
    efficiency = assess_efficiency(sandbox)
    # Efficiency never creates credit for a wrong or unverified business outcome.
    reward = 0.95 * business_reward
    if success:
        reward += 0.05 * efficiency['score']
    sandbox.closed = True
    return {'reward': reward, 'business_reward': business_reward, 'efficiency': efficiency, 'success': success, 'components': components,
            **{key: value for key, value in assessment.items() if key != 'passed'},
            'semantic_evidence': semantic_evidence}
