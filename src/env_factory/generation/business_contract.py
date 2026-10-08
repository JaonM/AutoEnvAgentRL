"""Lower shared business targets into goal and reward contracts.

Targets are defined once. Reward components select target IDs rather than
restating their predicates. Supported writes are generated from those targets.
"""
from __future__ import annotations

import copy


def expand_business_contract(source: dict) -> dict:
    contract = source.get('business_contract')
    if contract is None:
        return source
    if (not isinstance(contract, dict)
            or not {'version', 'state_targets', 'outcomes'} <= set(contract)
            or set(contract) - {'version', 'state_targets', 'outcomes', 'operations', 'report_outcomes', 'facts'}):
        raise ValueError('BUSINESS_CONTRACT_INVALID: requires version/state_targets/outcomes')
    facts = contract.get('facts', {})
    if not isinstance(facts, dict) or len(facts) > 64:
        raise ValueError('BUSINESS_CONTRACT_INVALID: facts must be a bounded mapping')
    remaining_nodes = 20000
    def resolve(value, stack=(), depth=0):
        nonlocal remaining_nodes
        remaining_nodes -= 1
        if depth > 40 or remaining_nodes < 0:
            raise ValueError('BUSINESS_CONTRACT_INVALID: fact expansion too deep')
        if isinstance(value, dict):
            if set(value) == {'literal'}:
                return copy.deepcopy(value)
            if set(value) == {'fact'}:
                name = value['fact']
                if not isinstance(name, str) or name not in facts or name in stack:
                    raise ValueError('BUSINESS_CONTRACT_INVALID: unknown or cyclic fact')
                return resolve(facts[name], (*stack, name), depth + 1)
            return {key: resolve(child, stack, depth + 1) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, stack, depth + 1) for child in value]
        return copy.deepcopy(value)
    contract = resolve({key: value for key, value in contract.items() if key != 'facts'})
    if contract['version'] != '1.0':
        raise ValueError('BUSINESS_CONTRACT_INVALID: unsupported version')
    if 'semantic_goal' in source or 'outcomes' in source:
        raise ValueError('BUSINESS_CONTRACT_DUPLICATE: goal and outcomes are generated from business_contract')
    targets = copy.deepcopy(contract['state_targets'])
    if (not isinstance(targets, dict) or not targets
            or any(not isinstance(key, str) or not key or not isinstance(value, dict)
                   for key, value in targets.items())):
        raise ValueError('BUSINESS_CONTRACT_INVALID: state_targets must map IDs to predicates')
    identities = {}
    for key, target in targets.items():
        identity = target.pop('identity', {})
        if not isinstance(identity, dict) or any(not isinstance(field, str) or not field for field in identity):
            raise ValueError('BUSINESS_CONTRACT_INVALID: identity must map columns to observed expressions')
        if identity:
            from ..sandbox_runtime import RewardStateSnapshots
            for field, expression in identity.items():
                if field in target.get('where', {}) or field in target.get('values', {}) or field in target.get('value_expressions', {}):
                    raise ValueError('BUSINESS_CONTRACT_DUPLICATE: identity field ' + field)
                if not RewardStateSnapshots.requested(expression):
                    raise ValueError('BUSINESS_CONTRACT_INVALID: dynamic identity requires observed at_tool evidence')
            target.setdefault('value_expressions', {}).update(copy.deepcopy(identity))
        identities[key] = identity
    components = contract['outcomes']
    if not isinstance(components, list) or not components:
        raise ValueError('BUSINESS_CONTRACT_INVALID: outcomes must select state target IDs')
    outcomes, covered, ids = [], set(), set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {'id', 'rubric', 'targets'}:
            raise ValueError('BUSINESS_CONTRACT_INVALID: outcome requires id/rubric/targets')
        identity, selected = component['id'], component['targets']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('BUSINESS_CONTRACT_INVALID: unique outcome IDs required')
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(key, str) or key not in targets for key in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('BUSINESS_CONTRACT_INVALID: unknown or duplicate state target reference')
        if not isinstance(component['rubric'], str) or not component['rubric'].strip():
            raise ValueError('BUSINESS_CONTRACT_INVALID: outcome rubric required')
        ids.add(identity)
        covered.update(selected)
        outcomes.append({'id': identity, 'rubric': component['rubric'], 'rule': {
            'source': 'business_state', 'path': '$', 'operator': 'state_predicates',
            'expected': [copy.deepcopy(targets[key]) for key in selected]}})
    if covered != set(targets):
        raise ValueError('BUSINESS_CONTRACT_UNSCORED: ' + ', '.join(sorted(set(targets) - covered)))
    result = copy.deepcopy(source)
    result.pop('business_contract')
    # Branch decisions and shared targets must resolve the same fact definitions.
    for route in result.get('execution_routes', []):
        route['when'] = resolve(route['when'])
    result['semantic_goal'] = {'row_predicates': copy.deepcopy(list(targets.values()))}
    result['outcomes'] = outcomes
    reports = contract.get('report_outcomes', [])
    if not isinstance(reports, list):
        raise ValueError('BUSINESS_CONTRACT_INVALID: report_outcomes must be a list')
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError('BUSINESS_CONTRACT_INVALID: report outcome must be an object')
        identity, rubric = report.get('id'), report.get('rubric')
        rule = report.get('rule')
        semantic = report.get('semantic')
        valid_rule = (set(report) == {'id', 'rubric', 'rule'}
                      and isinstance(rule, dict)
                      and rule.get('source') == 'final_agent_response'
                      and rule.get('operator') in {'value_targets', 'numeric_targets'})
        valid_semantic = (set(report) == {'id', 'rubric', 'semantic'}
                          and isinstance(semantic, dict))
        valid_fields = set(report) == {'id', 'rubric', 'fields'} and isinstance(report['fields'], list)
        if (not isinstance(identity, str) or not identity.strip() or identity in ids
                or not isinstance(rubric, str) or not rubric.strip()
                or not (valid_rule or valid_semantic or valid_fields)):
            raise ValueError('BUSINESS_CONTRACT_INVALID: report outcomes must evaluate answers, not duplicate state targets')
        ids.add(report['id'])
        outcomes.append(copy.deepcopy(report))
    tools = {tool['name']: tool for tool in result.get('business_tools', [])}
    operations = contract.get('operations', [])
    if not isinstance(operations, list):
        raise ValueError('BUSINESS_CONTRACT_INVALID: operations must be a list')
    seen = set()
    for operation in operations:
        if (not isinstance(operation, dict)
                or not {'tool', 'target', 'selector', 'bindings'} <= set(operation)
                or set(operation) - {'tool', 'target', 'selector', 'bindings', 'policy', 'derive', 'observed'}):
            raise ValueError('BUSINESS_CONTRACT_INVALID: operation requires tool/target/selector/bindings')
        name, target_id = operation['tool'], operation['target']
        if name not in tools or name in seen or target_id not in targets:
            raise ValueError('BUSINESS_CONTRACT_INVALID: unknown target/tool or duplicate operation')
        seen.add(name)
        tool, target = tools[name], targets[target_id]
        if 'implementation' in tool:
            raise ValueError('BUSINESS_CONTRACT_DUPLICATE: operation implementation is generated: ' + name)
        fields = (set(target.get('values', {})) | set(target.get('value_expressions', {}))) - set(identities[target_id])
        derived = operation.get('derive', [])
        if (not isinstance(derived, list) or any(not isinstance(field, str) or field not in fields for field in derived)
                or len(set(derived)) != len(derived)):
            raise ValueError('BUSINESS_CONTRACT_INVALID: derive must uniquely select target fields')
        bindings, selector = operation['bindings'], operation['selector']
        parameters = tool.get('parameters', {})
        declared = parameters.get('properties', {})
        required = set(parameters.get('required', []))
        if (not isinstance(bindings, dict) or set(bindings) != fields - set(derived) or not fields
                or any(not isinstance(arg, str) or arg not in declared or arg not in required
                       for arg in bindings.values())
                or len(set(bindings.values())) != len(bindings)):
            raise ValueError('BUSINESS_CONTRACT_INVALID: bind every target field to a distinct required argument')
        table = next((item for item in result.get('tables', [])
                      if item.get('table_name') == target.get('table')), None)
        columns = ({item['name'] for item in table.get('columns', [])}
                   if table is not None else set(target.get('where', {})) | set(identities[target_id]))
        if (not isinstance(selector, dict) or not selector
                or not isinstance(target.get('where'), dict) or not (target['where'] or identities[target_id])
                or not set(identities[target_id]) <= set(selector.values())
                or target.get('count', 1) != 1
                or any(arg not in required or not isinstance(column, str) or column not in columns
                       for arg, column in selector.items())
                or len(set(selector.values())) != len(selector)):
            raise ValueError('BUSINESS_CONTRACT_INVALID: selector must bind declared fields for one target row')
        for field, value in target.get('values', {}).items():
            if field in target.get('value_expressions', {}) or field in derived:
                continue
            parameter = declared[bindings[field]]
            if ('enum' in parameter and value not in parameter['enum']
                    or 'const' in parameter and value != parameter['const']):
                raise ValueError('BUSINESS_CONTRACT_CONFLICT: argument schema contradicts target ' + field)
            parameter['enum'] = [copy.deepcopy(value)]
        tool['implementation'] = {'operation': 'update', 'table': target['table'],
            'selector': copy.deepcopy(selector), 'changes': {arg: field for field, arg in bindings.items()}}
        if derived:
            tool['implementation']['derived_changes'] = {field: copy.deepcopy(
                target.get('value_expressions', {}).get(field, target.get('values', {}).get(field)))
                for field in derived}
        lifecycle = result.setdefault('business_lifecycle', {'version': '1.0'})
        policies = lifecycle.setdefault('tools', {})
        if name in policies:
            raise ValueError('BUSINESS_CONTRACT_DUPLICATE: operation policy is generated: ' + name)
        policy = copy.deepcopy(operation.get('policy', {}))
        if not isinstance(policy, dict):
            raise ValueError('BUSINESS_CONTRACT_INVALID: policy must be an object')
        policy.setdefault('require_idempotency', True)
        if not isinstance(policy.get('guards', []), list):
            raise ValueError('BUSINESS_CONTRACT_INVALID: policy guards must be a list')
        identity_guard = {'id': target_id + ':identity', 'table': target['table'],
            'selector': {column: {'argument': arg} for arg, column in selector.items()},
            'conditions': {**{column: {'eq': value} for column, value in target['where'].items()},
                           **{column: {'eq': {'expression': expression}}
                              for column, expression in identities[target_id].items()}}}
        observed = operation.get('observed', {})
        if (not isinstance(observed, dict) or set(observed) - columns):
            raise ValueError('BUSINESS_CONTRACT_INVALID: observed must map known fields to decision snapshots')
        if observed:
            from ..sandbox_runtime import RewardStateSnapshots
            if any(not RewardStateSnapshots.requested(expr) for expr in observed.values()):
                raise ValueError('BUSINESS_CONTRACT_INVALID: observed requires at_tool snapshots')
            policy.setdefault('guards', []).insert(0, {
                'id': target_id + ':freshness', 'table': target['table'],
                'selector': copy.deepcopy(identity_guard['selector']),
                'conditions': {field: {'eq': {'expression': expr}} for field, expr in observed.items()}})
        policy.setdefault('guards', []).insert(0, identity_guard)
        policies[name] = policy
    return result
