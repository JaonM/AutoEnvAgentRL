"""Identify retries of one idempotent request for causal counterfactuals.

Grouping is derived from the immutable authored trace, so both the validator and
the independent certifier can reproduce it without trusting a report's claims.
Different payloads, episodes or capture bindings remain separate operations.
"""
import json
import copy
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache


def bind_success_steps(steps, history):
    """Resolve requests at their original capture positions in a successful run.

    Final capture values cannot reconstruct a trace that reuses capture names.
    Consumers and certifiers derive grouping from this evidence, never from an
    unverified list of claimed idempotent groups.
    """
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator
    if not isinstance(history, list) or len(history) != len(steps):
        raise ValueError('success history must cover every authored step')
    bound, variables = copy.deepcopy(steps), {}
    for step, resolved, response in zip(steps, bound, history):
        if (not isinstance(response, Mapping)
                or response.get('operation') != step.get('operation')
                or response.get('status') != step.get('expected_status', 200)):
            raise ValueError('success history does not match authored operation/status')
        if step.get('operation') == 'tool_call':
            resolved['arguments'] = Evaluator.resolve_capture_argument(step.get('arguments', {}), variables)
            if 'idempotency_key' in step:
                resolved['idempotency_key'] = Evaluator.resolve_capture_argument(step['idempotency_key'], variables)
        for name, path in step.get('capture', {}).items():
            variables[name] = Evaluator._resolve(response.get('body'), path)
    return bound


def restore_pagination_refs(bound, original, implementations):
    """Retain same-episode cursor captures while freezing business decisions.

    A cursor from the baseline episode is invalid after reset. Missing a cursor
    producer still fails normally; this never invents or skips an omitted page.
    """
    cursors = {item['tool_name']: item['pagination']['cursor_argument']
               for item in implementations if item.get('pagination')}
    for step, authored in zip(bound, original, strict=True):
        cursor = cursors.get(step.get('tool_name'))
        if step.get('operation') != 'tool_call' or cursor is None:
            continue
        value = authored.get('arguments', {}).get(cursor)
        if isinstance(value, Mapping) and set(value) == {'$ref'}:
            step['arguments'][cursor] = copy.deepcopy(value)
    return bound


def idempotent_call_groups(steps):
    groups, identities, versions = [], {}, {}
    episode = 0

    def references(value):
        if isinstance(value, Mapping):
            if set(value) == {'$ref'} and isinstance(value['$ref'], str):
                reference = value['$ref']
                candidates = [name for name in versions if reference == name
                              or reference.startswith(name + '.') or reference.startswith(name + '[')]
                name = max(candidates, key=len) if candidates else reference
                yield name, versions.get(name, 0)
            else:
                for child in value.values():
                    yield from references(child)
        elif isinstance(value, list):
            for child in value:
                yield from references(child)

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        if step.get('operation') == 'reset':
            episode += 1
            versions.clear()
        if step.get('operation') != 'tool_call':
            continue
        key = step.get('idempotency_key')
        payload = [step.get('tool_name'), key, step.get('arguments', {})]
        identity = (episode, json.dumps(payload, sort_keys=True, ensure_ascii=False),
                    tuple(sorted(set(references(payload))))) if key else None
        if identity is not None and identity in identities:
            group_index = identities[identity]
            groups[group_index].append(index)
        else:
            group_index = len(groups)
            if identity is not None:
                identities[identity] = group_index
            groups.append([index])
        for name, path in step.get('capture', {}).items():
            # Re-capturing the same cached response/path preserves the binding.
            # A fresh request or another result field establishes a new binding.
            versions[name] = (group_index, json.dumps(path, sort_keys=True))
    return groups


def logical_trace_matches(steps, observed_names, lifecycle=None):
    """Each committed request needs one ordered event, including delayed retries."""
    groups = idempotent_call_groups(steps)
    if len(groups) != len(observed_names) or len(groups) > 128:
        return False
    names = [steps[group[0]].get('tool_name') for group in groups]
    if any(not isinstance(name, str) or not name for name in [*names, *observed_names]):
        return False
    if Counter(names) != Counter(observed_names):
        return False
    losses = {(fault['tool'], fault['status']) for fault in (lifecycle or {}).get('dynamics', {}).get('faults', [])
              if fault.get('phase') == 'after_commit'}
    def can_commit(position):
        status = steps[position].get('expected_status', 200)
        return type(status) is int and (200 <= status < 300 or
                                       (steps[position].get('tool_name'), status) in losses)
    candidates = [[position for position in group if can_commit(position)] for group in groups]
    visited = 0

    @lru_cache(maxsize=20000)
    def match(index, used, previous):
        nonlocal visited
        visited += 1
        if visited > 20000:
            return False  # bounded proof search; never accept uncertain coverage
        if index == len(observed_names):
            return True
        for group_index, name in enumerate(names):
            if name != observed_names[index] or used & (1 << group_index):
                continue
            position = next((p for p in candidates[group_index] if p > previous), None)
            if position is not None and match(index + 1, used | (1 << group_index), position):
                return True
        return False

    return match(0, 0, -1)


def swap_logical_calls(steps, first, second, *, grouping_steps=None):
    """Invert whole requests so a retained retry cannot repair the negative probe."""
    groups = idempotent_call_groups(steps if grouping_steps is None else grouping_steps)
    left = next(group for group in groups if first in group)
    right = next(group for group in groups if second in group)
    if left == right:
        raise ValueError('cannot reorder two attempts of the same idempotent request')
    positions = sorted(left + right)
    replacement = [steps[index] for index in right + left]
    for index, step in zip(positions, replacement):
        steps[index] = step


def optional_observation_positions(steps, implementations):
    """Optional repeat reads must still pass an executable omission probe.

    A declaration cannot exempt writes, first reads or arbitrary tool calls.
    The caller must verify that removing the observation retains full reward.
    """
    operations = {item['tool_name']: item['operation'] for item in implementations}
    seen, optional = set(), set()
    for index, step in enumerate(steps):
        if step.get('operation') == 'reset':
            seen.clear()
        if 'optional_observation' in step and type(step['optional_observation']) is not bool:
            raise ValueError('OPTIONAL_OBSERVATION_INVALID: expected boolean')
        if step.get('optional_observation'):
            name = step.get('tool_name')
            if (step.get('operation') != 'tool_call' or name not in seen
                    or operations.get(name) not in {'select', 'aggregate_count'}):
                raise ValueError('OPTIONAL_OBSERVATION_INVALID: only repeated read-only observations may be optional')
            optional.add(index)
        if step.get('operation') == 'tool_call':
            seen.add(step.get('tool_name'))
    return optional
