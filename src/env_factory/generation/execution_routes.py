"""Compile branch applicability from declared business decisions, never scores."""
import copy
import re

from env_factory.sandbox_runtime import ExecutionRoutes, RewardStateSnapshots


def compile_execution_routes(source, spec, metrics, rules, scripts):
    routes = source.get('execution_routes', [])
    if not routes:
        return
    if source['environment_plan']['mode'] not in {'stateful', 'reference_data'}:
        raise ValueError('EXECUTION_ROUTE_INVALID: business routes require business data')
    spec['execution_routes'] = copy.deepcopy(routes)
    process_ids = {m['id'] for m in metrics if m['category'] == 'process'}
    spec['process_metric_tools'] = {
        rule['metric_id']: rule['expected']['tool_name'] for rule in rules
        if rule['metric_id'] in process_ids and rule['operator'] == 'contains_tool_call'
    }
    plan = ExecutionRoutes(spec)
    plan.validate_user_branches(scripts or [])
    RewardStateSnapshots.validate_tools(source['tool_implementations'], [r['when'] for r in routes])
    all_stages = [script['interaction_protocol']['stages'] for script in scripts or []]
    edges = spec['capability_dag']['edges']
    for route in plan.routes:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', route['id']):
            raise ValueError('EXECUTION_ROUTE_INVALID: route id must be a bounded identifier')
        tools, stages = set(route['required_tools']), set(route['required_stages'])
        rejection = route['when'].get('user_rejection') if route['id'] in plan.dialogue_stop_ids else None
        if stages and not all_stages:
            raise ValueError('EXECUTION_ROUTE_INVALID: route requires stages without user scripts')
        for variant in all_stages:
            if not stages <= {stage['id'] for stage in variant}:
                raise ValueError('EXECUTION_ROUTE_INVALID: required stages must exist in every user variant')
            for stage in variant:
                protected = {stage['before_tool'], *(b['tool'] for b in stage.get('reuse_bindings', []))}
                if protected & tools and stage['id'] not in stages:
                    raise ValueError('EXECUTION_ROUTE_INVALID: cannot omit an interaction protecting a required tool')
                if stage['id'] in stages and not set(stage.get('requires_tools', [])) <= tools:
                    raise ValueError('EXECUTION_ROUTE_INVALID: required interaction depends on an inactive tool')
            if rejection:
                stage = next(stage for stage in variant if stage['id'] == rejection['stage'])
                ancestors = set()
                frontier = {stage['before_tool']}
                while frontier:
                    parents = {edge['from_tool'] for edge in edges if edge['to_tool'] in frontier}
                    frontier = parents - ancestors
                    ancestors |= parents
                if not tools <= ancestors or tools & plan.writes:
                    raise ValueError('EXECUTION_ROUTE_INVALID: refusal tools must be a read-only prerequisite prefix')
        if (spec['training_contract']['category'] == 'multi_step_agentic'
                and route['id'] not in plan.early_stop_ids | plan.dialogue_stop_ids):
            if len(tools) < 2 or not any(e['from_tool'] in tools and e['to_tool'] in tools for e in edges):
                raise ValueError('EXECUTION_ROUTE_INVALID: every multi-step route needs a genuine dependency')


def compile_route_probes(source, spec, scripts):
    """Each alternate valid branch has a concrete reference on valid changed data."""
    routes = {r['id']: r for r in spec.get('execution_routes', [])}
    definitions = source.get('route_probes', [])
    if not isinstance(definitions, list) or len(definitions) > 8:
        raise ValueError('EXECUTION_ROUTE_INVALID: route_probes must be a bounded list')
    if not routes:
        if definitions:
            raise ValueError('EXECUTION_ROUTE_INVALID: probes require routes')
        return []
    successes = [case for case in source['scenarios'] if case.get('kind') == 'goal_success']
    covered = {case.get('execution_route') for case in successes}
    if not covered <= routes.keys():
        raise ValueError('EXECUTION_ROUTE_INVALID: each success reference needs an execution_route')
    for case in successes:
        _validate_reference_route_tools(routes[case['execution_route']], case['steps'], case['scenario_id'])
    probes, seen = [], set()
    from .agent_authoring import expand_capture_fields
    from .interaction_contract import interleave_scenarios, scenario_script_indices
    for definition in definitions:
        if (not isinstance(definition, dict)
                or not {'route_id', 'initial_changes', 'reference'} <= set(definition)
                or set(definition) - {'route_id', 'initial_changes', 'reference', 'reference_variants', 'user_variables'}
                or definition['route_id'] not in routes or definition['route_id'] in seen):
            raise ValueError('EXECUTION_ROUTE_INVALID: probe requires unique route_id, initial_changes and reference')
        identity = definition['route_id']
        seen.add(identity)
        variants = definition.get('reference_variants', [])
        if not isinstance(variants, list) or len(variants) > 8:
            raise ValueError('EXECUTION_ROUTE_INVALID: probe supports at most eight reference variants')
        reset = {'operation': 'reset', 'body': {'seed': 17}}
        scenarios, covered_scripts, variant_ids = [], set(), set()
        for index, original in enumerate([definition['reference'], *variants]):
            reference = copy.deepcopy(original)
            allowed = {'calls', 'answer', 'user_script_indices'} | ({'id'} if index else set())
            required_fields = {'calls', 'answer'} | ({'id', 'user_script_indices'} if index else set())
            if (not isinstance(reference, dict) or set(reference) - allowed
                    or not required_fields <= set(reference) or not isinstance(reference['answer'], str)):
                raise ValueError('EXECUTION_ROUTE_INVALID: probe reference requires calls, answer and valid script scope')
            suffix = ''
            if index:
                variant_id = reference['id']
                if (not isinstance(variant_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', variant_id)
                        or variant_id in variant_ids):
                    raise ValueError('EXECUTION_ROUTE_INVALID: probe reference variant ids must be unique identifiers')
                variant_ids.add(variant_id)
                suffix = '__' + variant_id
            if scripts:
                indices = set(scenario_script_indices(reference, len(scripts)))
                if indices & covered_scripts:
                    raise ValueError('EXECUTION_ROUTE_INVALID: probe references overlap user scripts')
                covered_scripts.update(indices)
            elif variants or 'user_script_indices' in reference:
                raise ValueError('EXECUTION_ROUTE_INVALID: scoped probe references require user scripts')
            calls = reference['calls']
            minimum = 0 if (not routes[identity]['required_tools'] and set(routes[identity]['when']) == {'user_rejection'}) else 1
            if (not isinstance(calls, list) or not minimum <= len(calls) <= 32
                    or any(not isinstance(call, dict) or not isinstance(call.get('tool_name'), str) for call in calls)):
                raise ValueError('EXECUTION_ROUTE_INVALID: probe needs bounded reference calls')
            expand_capture_fields(calls)
            if not set(routes[identity]['required_tools']) <= {call['tool_name'] for call in calls}:
                raise ValueError('EXECUTION_ROUTE_INVALID: probe omits required tools')
            _validate_reference_route_tools(routes[identity], calls, 'route_' + identity + suffix)
            scenarios.append({'scenario_id': 'route_' + identity + suffix, 'kind': 'goal_success',
                'execution_route': identity,
                **({'user_script_indices': reference.get('user_script_indices', list(range(len(scripts))))} if scripts else {}),
                'steps': [copy.deepcopy(reset), *[{**call, 'operation': 'tool_call',
                    'expected_status': call.get('expected_status', 200)} for call in calls],
                    {'operation': 'agent_response', 'content': reference['answer']},
                    {'operation': 'reward', 'step_id': 'reward'}],
                'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'gte', 'expected': 1}]})
        if scripts and covered_scripts != set(range(len(scripts))):
            raise ValueError('EXECUTION_ROUTE_UNCOVERED: every route probe must cover every user script')
        scenarios.append({'scenario_id': 'empty_' + identity, 'kind': 'goal_failure',
             'steps': [copy.deepcopy(reset), {'operation': 'reward', 'step_id': 'reward'}],
             'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'lte', 'expected': 0}]})
        process_rules = []
        if source.get('metrics') and source.get('tool_bindings'):
            from .contract_compiler import ContractCompiler
            from .interaction_contract import bind_captured_interaction_arguments, stage_bindings
            from .input_grounding import validate_query_inputs
            validate_query_inputs({**source, 'scenarios': scenarios})
            active = {name for name, tool in spec.get('process_metric_tools', {}).items()
                      if tool in routes[identity]['required_tools']}
            process_rules = ContractCompiler._compile_process_metric_implementations(
                metrics=[metric for metric in source['metrics'] if metric['id'] in active],
                metric_implementations=[], business_scenarios=scenarios, tool_bindings=source['tool_bindings'])
            bindings = [{'script_id': script['script_id'], **binding}
                for script in scripts or [] for stage in script['interaction_protocol']['stages']
                for binding in stage_bindings(stage) if binding['tool'] in routes[identity]['required_tools']]
            bind_captured_interaction_arguments(process_rules, scripts or [], bindings)
        if scripts:
            scenarios = interleave_scenarios(scenarios, scripts, execution_routes=list(routes.values()))
        changes = definition['initial_changes']
        if not isinstance(changes, list) or len(changes) > 16:
            raise ValueError('EXECUTION_ROUTE_INVALID: initial_changes must be a bounded list')
        variables = definition.get('user_variables', {})
        validate_probe_user_variables(variables, scripts or [])
        probes.append({'route_id': identity, 'initial_changes': copy.deepcopy(changes),
                       'user_variables': copy.deepcopy(variables), 'process_rules': process_rules, 'scenarios': scenarios})
    if covered | seen != routes.keys():
        raise ValueError('EXECUTION_ROUTE_UNCOVERED: every route needs a primary reference or route probe')
    return probes


def _validate_reference_route_tools(route, steps, reference_id):
    """An authored success cannot depend on tools its route makes inactive."""
    called = {step['tool_name'] for step in steps
              if step.get('operation', 'tool_call') == 'tool_call' and 'tool_name' in step}
    undeclared = sorted(called - set(route['required_tools']))
    if undeclared:
        raise ValueError(
            f'EXECUTION_ROUTE_REFERENCE_MISMATCH: reference {reference_id} calls '
            f'{undeclared}, omitted from route {route["id"]} required_tools. '
            'Declare the full necessary query chain, or split success and early-stop '
            'routes using actual business conditions. Do not remove a necessary query '
            'or relax the omission reward check.')


def validate_probe_user_variables(variables, scripts):
    import math
    if (not isinstance(variables, dict) or len(variables) > 16
            or variables and not scripts
            or any(not isinstance(value, (str, bool, int, float))
                   or isinstance(value, float) and not math.isfinite(value)
                   or isinstance(value, str) and len(value) > 512 for value in variables.values())):
        raise ValueError('EXECUTION_ROUTE_INVALID: user-variable fixture must contain bounded declared preferences')
    for script in scripts:
        if any(key not in script.get('variables', {}) or type(value) is not type(script['variables'][key])
               for key, value in variables.items()):
            raise ValueError('EXECUTION_ROUTE_INVALID: user-variable fixture changes an undeclared preference or its type')


def apply_initial_changes(data, changes):
    """Validate alternate fixtures; invalid data is never branch success evidence."""
    baseline = copy.deepcopy(data.baseline)
    for change in changes:
        if not isinstance(change, dict) or set(change) != {'table', 'where', 'values'}:
            raise ValueError('EXECUTION_ROUTE_INVALID: initial change requires table, where and values')
        table, selector, values = change['table'], change['where'], change['values']
        schema = data.schemas.get(table, {})
        columns = {c['name'] for c in schema.get('columns', [])}
        if (table not in baseline or not isinstance(selector, dict) or not selector
                or not isinstance(values, dict) or not values or set(selector) - columns
                or set(values) - columns or set(values) & set(schema.get('primary_key', []))):
            raise ValueError('EXECUTION_ROUTE_INVALID: invalid alternate fixture columns or identity')
        matches = [row for row in baseline[table] if all(row.get(k) == v for k, v in selector.items())]
        if len(matches) != 1:
            raise ValueError('EXECUTION_ROUTE_INVALID: alternate fixture selector must match exactly one row')
        matches[0].update(copy.deepcopy(values))
    data._validate_tables(baseline)
    data.baseline = baseline
