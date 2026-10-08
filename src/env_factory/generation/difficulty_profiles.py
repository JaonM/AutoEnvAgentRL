"""Immutable runner-owned task difficulty requirements, separate from prose labels."""
import copy
import math


PROFILES = {
    'standard': {},
    'long_horizon_planning': {'min_dependency_depth': 8, 'requires_recovery': True},
    'large_scale_retrieval': {'min_dataset_rows': 10000, 'max_page_size': 100,
                            'min_complete_pages': 3, 'min_read_tables': 2},
    'complex_write_transactions': {'min_changed_tables': 2, 'min_changed_objects': 3,
                                   'requires_authorization': True, 'requires_idempotency': True,
                                   'requires_recovery': True},
}


def requirements(profile='standard'):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('DIFFICULTY_PROFILE_INVALID: unknown profile')
    return {'version': '1.0', 'profile': profile, **copy.deepcopy(PROFILES[profile])}


def parse_mix(text):
    result = {}
    for part in text.split(','):
        try:
            name, raw = (value.strip() for value in part.split('='))
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError('DIFFICULTY_MIX_INVALID: use profile=weight pairs') from exc
        if name not in PROFILES or name in result or not math.isfinite(value) or value < 0:
            raise ValueError('DIFFICULTY_MIX_INVALID: unknown/duplicate profile or invalid weight')
        result[name] = value
    total = sum(result.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('DIFFICULTY_MIX_INVALID: positive finite total required')
    return result


def allocate(categories, mix, *, rng, available_modes, task_intent=None):
    weights = parse_mix(mix) if isinstance(mix, str) else parse_mix(','.join(f'{k}={v}' for k, v in mix.items()))
    labels = list(weights)
    raw = [len(categories) * weights[name] / sum(weights.values()) for name in labels]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(len(labels)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[:len(categories)-sum(counts)]:
        counts[index] += 1
    hard = [name for name, count in zip(labels, counts) if name != 'standard' for _ in range(count)]
    slots = [i for i, category in enumerate(categories) if category == 'multi_step_agentic']
    if len(hard) > len(slots):
        raise ValueError('DIFFICULTY_MIX_INCOMPATIBLE: hard profiles require enough multi_step_agentic slots')
    if not hard:
        return ['standard'] * len(categories)
    rng.shuffle(slots)
    rng.shuffle(hard)
    result = ['standard'] * len(categories)
    for index, name in zip(slots, hard):
        validate_request({'difficulty_profile': name, 'training_category': categories[index],
                          'available_environment_modes': list(available_modes), 'task_intent': task_intent})
        result[index] = name
    return result


def validate_request(request):
    required = requirements(request.get('difficulty_profile', 'standard'))
    if 'difficulty_requirements' in request and request['difficulty_requirements'] != required:
        raise ValueError('DIFFICULTY_PROFILE_INVALID: frozen requirements differ from the supported profile version')
    profile = required['profile']
    if profile == 'standard':
        return required
    if request.get('training_category') != 'multi_step_agentic':
        raise ValueError('DIFFICULTY_PROFILE_INCOMPATIBLE: hard profiles require multi_step_agentic')
    modes = set(request.get('available_environment_modes', ['reference_data', 'stateful']))
    allowed = {'stateful'} if profile == 'complex_write_transactions' else {'reference_data', 'stateful'}
    if not modes & allowed or profile == 'complex_write_transactions' and request.get('task_intent') == 'query':
        raise ValueError('DIFFICULTY_PROFILE_INCOMPATIBLE: required business mode conflicts with request')
    return required


def augment_realism(request):
    """Merge assigned mechanisms without replacing the invocation's requirements."""
    required = validate_request(request)
    value = copy.deepcopy(request.get('realism_requirements', {'version': '1.0', 'mechanisms': []}))
    extra = []
    if required.get('requires_recovery'):
        extra.append('fault_recovery')
    if required.get('requires_authorization'):
        extra.extend(['authorization', 'idempotency'])
    value['mechanisms'] = sorted(set(value.get('mechanisms', [])) | set(extra))
    return value


def process_bindings(metric_implementations):
    bindings = {}
    for rule in metric_implementations:
        if rule.get('operator') == 'contains_tool_call':
            name = rule['expected'].get('tool_name')
            if name:
                bindings.setdefault(name, []).append(copy.deepcopy(rule['expected']))
    return bindings


def compile_requirements(source, request, spec, metric_implementations):
    required = validate_request(request)
    if 'difficulty_requirements' in source:
        raise ValueError('DIFFICULTY_PROFILE_INVALID: difficulty_requirements is runner-owned')
    profile = required['profile']
    spec['difficulty_requirements'] = required
    mode = source['environment_plan']['mode']
    if profile != 'standard' and mode not in ({'stateful'} if profile == 'complex_write_transactions' else {'reference_data', 'stateful'}):
        raise ValueError('DIFFICULTY_PROFILE_MISSING: incompatible actual environment mode')
    tools = source['tool_implementations']
    if profile == 'complex_write_transactions':
        writes = [tool for tool in tools if tool['operation'] in {'insert', 'update', 'delete'}]
        compensation = sorted(set(source.get('realism_requirements', {}).get('compensation_tools', []))
                              | set(request.get('realism_requirements', {}).get('compensation_tools', [])))
        rules = source.get('business_lifecycle', {}).get('tools', {})
        if len({tool['table'] for tool in writes}) < required['min_changed_tables']:
            raise ValueError('DIFFICULTY_PROFILE_MISSING: complex transactions need writes across at least two tables')
        if not compensation or set(compensation) - {tool['tool_name'] for tool in writes}:
            raise ValueError('DIFFICULTY_PROFILE_MISSING: declare actual compensation write tools in realism_requirements')
        if any(not rules.get(tool['tool_name'], {}).get('requires_grants')
               or not rules.get(tool['tool_name'], {}).get('require_idempotency') for tool in writes):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: all transaction writes require bound authorization and idempotency')
        terms = {(stage.get('before_tool'), grant)
                 for variant in source.get('interaction_contract', {}).get('variants', [])
                 for original in variant.get('stages', [])
                 for stage in [original, *[{**original, **branch} for branch in original.get('branches', [])]]
                 for grant, term in stage.get('approval_terms', {}).items() if term.get('scope') == 'terms'}
        if any((tool['tool_name'], grant) not in terms for tool in writes
               for grant in rules[tool['tool_name']]['requires_grants']):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: each transaction grant needs tool-bound approval terms')
        primary_keys = {table['table_name']: table.get('primary_key', []) for table in source['tables']}
        if any(not keys for keys in primary_keys.values()):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: transaction audit requires primary keys on all tables')
        spec['transaction_contract'] = {'write_tools': sorted(tool['tool_name'] for tool in writes),
            'compensation_tools': sorted(compensation), 'table_primary_keys': primary_keys}
    paging = [tool for tool in tools if tool.get('pagination')]
    if profile == 'large_scale_retrieval':
        if sum(len(table['rows']) for table in source['tables']) < required['min_dataset_rows']:
            raise ValueError('DIFFICULTY_PROFILE_MISSING: retrieval needs at least 10000 actual rows')
        if not paging or len({tool['table'] for tool in tools if tool['operation'] in {'select', 'aggregate_count'}}) < 2:
            raise ValueError('DIFFICULTY_PROFILE_MISSING: bounded pagination and cross-table reads required')
        if any(metric.get('semantic_fields') and metric.get('semantic_evidence_mode') != 'fields'
               for metric in source['metrics']):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: large retrieval explanations require fields evidence mode')
        large_tables = {table['table_name'] for table in source['tables'] if len(table['rows']) >= 10000}
        if any(tool['operation'] == 'select' and tool['table'] in large_tables and not tool.get('pagination') for tool in tools):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: selects on large tables must be paginated')
    names = source.get('complete_retrieval_tools', [])
    if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or set(names) - {tool['tool_name'] for tool in paging}):
        raise ValueError('RETRIEVAL_CONTRACT_INVALID: uniquely name declared paginated tools')
    if profile != 'large_scale_retrieval':
        paging = [tool for tool in paging if tool['tool_name'] in names]
    if paging:
        bindings = process_bindings(metric_implementations)
        if any(tool['tool_name'] not in bindings for tool in paging):
            raise ValueError('DIFFICULTY_PROFILE_MISSING: every paginated query requires a process binding')
        spec['retrieval_contract'] = {'tools': [
            {'tool_name': tool['tool_name'], 'table': tool['table'],
             'result_field': tool.get('result_field', 'records'), **tool['pagination']}
            for tool in paging], 'process_bindings': bindings}
        public = source['description']['public_input']
        public['initial_user_message'] += (
            '\n分页检索要求：对下列任务查询，从第一页开始沿返回游标读取到 has_more=false；'
            '重试同一页不算新页，其他查询的完整结果不能替代目标查询。即使已知最终答案，也必须完成必要检索：'
            + ', '.join(tool['tool_name'] for tool in paging))


def require_coverage(artifacts, observations):
    required = artifacts['task_spec'].get('difficulty_requirements', requirements())
    profile = required['profile']
    if profile == 'standard':
        return {'passed': True, 'profile': profile}
    if profile == 'large_scale_retrieval':
        valid = [row for row in observations if row.get('retrieval_complete')
                 and max((item['pages'] for item in row.get('retrieval_chains', [])), default=0) >= required['min_complete_pages']
                 and len(row.get('read_tables', [])) >= required['min_read_tables']]
    elif profile == 'long_horizon_planning':
        valid = [row for row in observations if row.get('dependency_depth', 0) >= required['min_dependency_depth']
                 and row.get('recovered_fault_ids')]
    elif profile == 'complex_write_transactions':
        valid = [row for row in observations if transaction_covered(required, row.get('transaction_witness', {}))
                 and row.get('authorization_completed')]
    else:
        valid = []
    if not valid:
        raise ValueError('DIFFICULTY_EXECUTION_UNCOVERED: ' + profile + ' lacks verified executed coverage')
    return {'passed': True, 'profile': profile, 'observations': len(valid)}


def transaction_covered(required, witness):
    return (witness.get('audit_valid') is True and witness.get('unrelated_state_preserved') is True
            and len(witness.get('changed_tables', [])) >= required['min_changed_tables']
            and witness.get('changed_objects', 0) >= required['min_changed_objects']
            and bool(witness.get('compensated_fault_ids')) and bool(witness.get('idempotent_replay_tools')))


def observed_dependency_depth(diagnostics):
    """Count distinct business tools linked by consumed executed result values."""
    edges = [item['edge'] for item in diagnostics.get('dependency_edges', [])
             if item.get('established') is True and item['edge'].get('result_path')
             and item['edge']['result_path'] != '$.next_cursor' and item['edge'].get('argument_path')
             and item['edge']['from_tool'] != item['edge']['to_tool']]
    parents = {}
    for edge in edges:
        parents.setdefault(edge['from_tool'], set())
        parents.setdefault(edge['to_tool'], set()).add(edge['from_tool'])
    depth = {}
    while parents:
        ready = [name for name, deps in parents.items() if deps <= depth.keys()]
        if not ready:
            return 0  # A cycle is not additional planning depth.
        for name in ready:
            depth[name] = 1 + max((depth[parent] for parent in parents.pop(name)), default=0)
    return max(depth.values(), default=0)
