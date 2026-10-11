"""Expand declarative scenario reuse before any business validation."""
from __future__ import annotations

import copy

from env_factory_v2.contracts.expressions import ContractError, canonical


def expand_scenarios(scenarios):
    if not isinstance(scenarios, list):
        raise ContractError('scenarios must be a list')
    index = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get('id'), str):
            raise ContractError('scenario identity required')
        if scenario['id'] in index:
            raise ContractError('duplicate scenario identity')
        index[scenario['id']] = scenario
    expanded = {}
    depths = {}

    def resolve(identity, ancestors=()):
        if identity in ancestors or len(ancestors) > 32:
            raise ContractError('scenario inheritance cycle or depth limit')
        if identity not in index:
            raise ContractError('unknown parent scenario')
        if identity in expanded:
            if len(ancestors) + depths[identity] > 32:
                raise ContractError('scenario inheritance depth limit')
            return copy.deepcopy(expanded[identity])
        child = copy.deepcopy(index[identity])
        if 'extends' in child:
            parent = child.pop('extends')
            if not isinstance(parent, str) or 'expect_success' not in child:
                raise ContractError('inherited scenario requires parent and explicit expectation')
            extra = child.pop('append_steps', [])
            if not isinstance(extra, list):
                raise ContractError('append_steps must be a list')
            result = {**resolve(parent, (*ancestors, identity)), **child}
            depths[identity] = depths[parent] + 1
            if depths[identity] > 32:
                raise ContractError('scenario inheritance depth limit')
            if extra:
                if not isinstance(result.get('steps'), list):
                    raise ContractError('inherited steps must be a list')
                result['steps'] = result['steps'] + extra
        else:
            if 'append_steps' in child:
                raise ContractError('append_steps requires scenario inheritance')
            result = child
            depths[identity] = 0
        if len(canonical(result).encode()) > 4 * 1024 * 1024:
            raise ContractError('expanded scenario size limit')
        expanded[identity] = result
        return copy.deepcopy(result)

    result = [resolve(scenario['id']) for scenario in scenarios]
    if len(canonical(result).encode()) > 4 * 1024 * 1024:
        raise ContractError('expanded scenarios size limit')
    return result
