"""Derived obligation/evidence identities, not a second business specification."""
import hashlib

from .expressions import canonical
from .demand import cited_requirements


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def mandatory_obligations(source):
    """Reverse-review catalog of authored outputs and enforced success conditions."""
    result = []
    def add(path, kind, value):
        result.append({'path': path, 'kind': kind, 'definition': value})
    def fields(schema, path):
        if schema.get('type') == 'array':
            fields(schema['items'], path + '/items')
        elif schema.get('type') == 'object':
            for key, child in schema['properties'].items():
                escaped = key.replace('~', '~0').replace('/', '~1')
                child_path = path + '/properties/' + escaped
                add(child_path, 'answer_property', child)
                fields(child, child_path)
    for index, answer in enumerate(source['answers']):
        path = f'/answers/{index}'
        add(path, 'answer_field', {key: answer[key] for key in ('id', 'meaning', 'weight', 'comparison')})
        fields(answer['schema'], path + '/schema')
    for index, explanation in enumerate(source.get('explanations', [])):
        for position, criterion in enumerate(explanation['criteria']):
            add(f'/explanations/{index}/criteria/{position}', 'communication', criterion)
    for root in ('goals', 'invariants'):
        for index, item in enumerate(source.get(root, [])):
            add(f'/{root}/{index}', 'world_constraint', item)
    for index, branch in enumerate(source.get('success', [])):
        prefix = f'/success/{index}'
        add(prefix + '/when', 'completion_branch', {'public': branch['public'], 'when': branch['when']})
        for key in ('required_tools', 'required_stages', 'required_platform_observations'):
            for position, item in enumerate(branch.get(key, [])):
                add(f'{prefix}/{key}/{position}', key, item)
        for position, observation in enumerate(branch.get('required_observations', [])):
            for column_index, column in enumerate(observation['columns']):
                add(f'{prefix}/required_observations/{position}/columns/{column_index}',
                    'required_observation_column', {'column': column, 'tools': observation['tools']})
        for key in ('min_business_calls', 'max_business_calls'):
            if key in branch:
                add(prefix + '/' + key, 'call_constraint', branch[key])
    return result


def implementation_paths(source):
    """A finite catalog prevents response paths from referring to payload wrappers."""
    paths = []
    for root in ('rules', 'tools', 'answers', 'goals', 'invariants', 'explanations', 'success'):
        if root not in source:
            continue
        value = source[root]
        if not value or root == 'success':
            paths.append('/' + root)
        if isinstance(value, list):
            for index, item in enumerate(value):
                prefix = f'/{root}/{index}'
                paths.append(prefix)
                if isinstance(item, dict):
                    paths.extend(prefix + '/' + key.replace('~', '~0').replace('/', '~1') for key in item)
    stages = source.get('user', {}).get('stages', [])
    if not stages:
        paths.append('/user/stages')
    for index, stage in enumerate(stages):
        prefix = f'/user/stages/{index}'
        paths.append(prefix)
        paths.extend(prefix + '/' + key.replace('~', '~0').replace('/', '~1') for key in stage)
    # Declared integrity constraints are executable evidence. Fixture rows and
    # generators are deliberately excluded: example values cannot prove rules.
    for index, table in enumerate(source.get('tables', [])):
        paths.extend(f'/tables/{index}/{key}' for key in ('schema', 'primary_key', 'foreign_keys')
                     if key in table)
    return paths


def obligation_index(compiled):
    from .verification import authored_cases
    source = compiled['source']
    # Conservative dependency boundary: rules, tools, user, time, rewards and
    # public context all affect execution. Recipes have separate identities.
    execution = {k: v for k, v in source.items() if k not in {'scenarios', 'probes'}}
    execution_identity = identity(execution)
    context = source.get('demand', {}).get('agent_context', {'user_request': source['task']})
    entries = mandatory_obligations(source)
    return {'version': '1.0', 'contract_identity': compiled['identity'],
        'scope': 'derived_dependency_index_not_business_verdict_or_score_cache',
        'requirements': cited_requirements(context),
        'obligations': [{**entry, 'identity': identity(entry)} for entry in entries],
        'execution_identity': execution_identity,
        'references': [{'id': scenario['id'], 'kind': kind,
            'identity': identity({'execution': execution_identity, 'scenario': scenario, 'rule_ids': rules})}
            for kind, scenario, rules in authored_cases(compiled)]}


def reference_changes(previous, current):
    def refs(index):
        return {(x['kind'], x['id']): x['identity'] for x in index['references']}
    before, after = refs(previous), refs(current)
    old_obligations = {x['path']: x['identity'] for x in previous['obligations']}
    new_obligations = {x['path']: x['identity'] for x in current['obligations']}
    return {'scope': 'evidence_invalidation_not_failure_or_repair_proof',
        'previous_contract_identity': previous['contract_identity'],
        'current_contract_identity': current['contract_identity'],
        'changed_obligation_paths': sorted(p for p in old_obligations.keys() | new_obligations.keys()
                                           if old_obligations.get(p) != new_obligations.get(p)),
        'references_requiring_validation': [{'kind': k, 'id': i} for k, i in sorted(after)
                                           if before.get((k, i)) != after[(k, i)]],
        'removed_references': [{'kind': k, 'id': i} for k, i in sorted(before.keys() - after.keys())],
        'instruction': 'Revalidate affected references against the current original-grounded obligations. '
                       'Changing a criterion does not update old reference wording or prove it correct. '
                       'Do not weaken requirements, delete coverage or relabel expected outcomes to silence findings.'}
