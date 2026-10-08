"""Compile task-authored, executable multi-turn user protocols."""
import copy
import json
import re
import math

KINDS = {'information_required', 'option_selection', 'constraint_update', 'execution_confirmation', 'correction'}


def stage_bindings(stage):
    """A single disclosure can constrain later uses without another user turn."""
    primary = stage.get('bind_to')
    return ([primary] if isinstance(primary, dict) else []) + stage.get('reuse_bindings', [])


def scenario_script_indices(scenario, count):
    """An omitted selector preserves legacy expansion to every script."""
    indices = scenario.get('user_script_indices')
    if indices is None:
        return list(range(count))
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('INTERACTION_REFERENCE_SCOPE: user_script_indices must be unique valid zero-based indices')
    return indices


def compile_interactions(contract, description, tools, count, *, implementations=(), scenarios=(), privacy_findings=None, structured_requests=False, capture_bindings=None, execution_routes=()):
    if not isinstance(contract, dict) or contract.get('version') != '1.0':
        raise ValueError('interaction_contract requires version 1.0')
    variants = contract.get('variants', [])
    if len(variants) != count:
        raise ValueError('interaction variants must match user_script_count')
    writes = {tool['tool_name'] for tool in implementations if tool['operation'] in {'insert','update','delete'}}
    successes = [scenario for scenario in scenarios if scenario.get('kind') == 'goal_success']
    for scenario in scenarios:
        scenario_script_indices(scenario, count)
    names = {tool['function']['name'] for tool in tools}
    scripts, signatures, kinds = [], set(), set()
    public = description.get('public_input', {}).get('initial_user_message', '')
    routes = {route['id']: route for route in execution_routes}
    for index, variant in enumerate(variants):
        variables = variant.get('variables', {})
        if (not isinstance(variables, dict) or len(variables) > 16
                or any(not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,63}', key)
                       or not isinstance(value, (str, bool, int, float))
                       or isinstance(value, str) and len(value) > 512
                       or isinstance(value, float) and not math.isfinite(value)
                       for key, value in variables.items())):
            raise ValueError('INTERACTION_USER_STATE: variables must be bounded private scalar preferences')
        applicable = [scenario for scenario in successes if index in scenario_script_indices(scenario, count)]
        def inactive_or_refused(stage_id, tool):
            if not applicable:
                return False
            for scenario in applicable:
                route = routes.get(scenario.get('execution_route'), {})
                terminal = route.get('when', {}).get('user_rejection', {})
                if not route or tool in route.get('required_tools', []):
                    return False
                from env_factory.sandbox_runtime import RewardStateSnapshots
                observed = RewardStateSnapshots.user_values(route.get('when', {}))
                if (stage_id in route.get('required_stages', []) and terminal.get('stage') != stage_id
                        and stage_id not in observed):
                    return False
            return True
        references = []
        for scenario in successes:
            if index not in scenario_script_indices(scenario, count):
                continue
            available = set()
            for step in scenario['steps']:
                if step.get('operation') == 'reset':
                    available.clear()
                if step.get('operation') == 'tool_call':
                    references.append((step, available.copy()))
                    available.update(step.get('capture', {}))
        if successes and not references and not (applicable and all(
                routes.get(case.get('execution_route'), {}).get('when', {}).get('user_rejection')
                and not routes[case['execution_route']]['required_tools'] for case in applicable)):
            raise ValueError(f'INTERACTION_REFERENCE_SCOPE: variant {index} has no successful reference')
        stages = variant.get('stages', [])
        if not 2 <= len(stages) <= 5:
            raise ValueError('each interaction variant requires 2..5 necessary user turns')
        signature = json.dumps({'stages': stages, 'variables': variables}, sort_keys=True, ensure_ascii=False)
        if signature in signatures:
            raise ValueError('interaction variants must differ beyond script identifiers')
        signatures.add(signature)
        seen, stages = set(), copy.deepcopy(stages)
        bindings = set()
        disclosed_text = public
        for stage in stages:
            key, kind = stage.get('id'), stage.get('kind')
            if not isinstance(key, str) or not key or key in seen or kind not in KINDS:
                raise ValueError('invalid interaction stage identity/kind')
            seen.add(key); kinds.add(kind)
            for field in ('user_reply', 'reference_response', 'purpose', 'retry_reply'):
                if not isinstance(stage.get(field), str) or not stage[field].strip():
                    raise ValueError(f'interaction stage requires {field}')
            reference_tokens = re.findall(r'\{\{([^{}]+)\}\}', stage['reference_response'])
            observed_reference = bool(reference_tokens)
            if observed_reference and (not structured_requests
                    or set(reference_tokens) - {'observed_options', 'observed_result', 'observed_context'}
                    or (set(reference_tokens) & {'observed_options', 'observed_result'} and (
                        'options_from' not in stage or 'observed_options' not in reference_tokens))
                    or ('observed_context' in reference_tokens and 'context_from' not in stage)):
                raise ValueError('INTERACTION_REFERENCE_TEMPLATE: observed menu needs options_from '
                                 'and {{observed_options}}; observed context needs context_from')
            if 'context_from' in stage and ('observed_context' not in reference_tokens or not structured_requests):
                raise ValueError('INTERACTION_REFERENCE_TEMPLATE: context_from needs structured {{observed_context}}')
            required = stage.get('assistant_contains_all')
            if not structured_requests and (not isinstance(required, list) or not required or any(not isinstance(x, str) or not x.strip() for x in required)):
                raise ValueError('interaction stage requires grounded assistant_contains_all')
            if not structured_requests and not all(x in stage['reference_response'] for x in required):
                raise ValueError('reference response must satisfy interaction trigger')
            before = stage.get('before_tool', '__answer__')
            if before != '__answer__' and before not in names:
                raise ValueError('interaction stage references unknown before_tool')
            from env_factory.sandbox_runtime import ContractUserSimulator
            ContractUserSimulator.validate_branches(stage.get('branches', []))
            for branch in stage.get('branches', []):
                if (branch.get('grant_permissions') or branch.get('revoke_permissions')) and kind != 'execution_confirmation':
                    raise ValueError('branch permission changes require execution confirmation')
            for permission_action in ('grant_permissions', 'revoke_permissions'):
                permissions = stage.get(permission_action, [])
                if (not isinstance(permissions, list) or any(not isinstance(x, str) or not x for x in permissions)
                        or permissions and kind != 'execution_confirmation'):
                    raise ValueError('permission changes require an execution confirmation stage')
            if set(stage.get('grant_permissions', [])) & set(stage.get('revoke_permissions', [])):
                raise ValueError('a stage cannot grant and revoke the same permission')
            if kind == 'execution_confirmation' and before not in writes:
                raise ValueError('execution confirmation must protect an actual tool')
            reuse = stage.get('reuse_bindings', [])
            if (not isinstance(reuse, list) or len(reuse) > 8
                    or reuse and kind == 'execution_confirmation'
                    or any(not isinstance(item, dict) or set(item) != {'tool', 'argument'}
                           or not isinstance(item.get('tool'), str)
                           or item.get('tool') not in names or item.get('tool') == before
                           or not isinstance(item.get('argument'), str) for item in reuse)
                    or len({(item['tool'], item['argument']) for item in reuse}) != len(reuse)):
                raise ValueError('INTERACTION_REUSE_INVALID: reuse_bindings require unique later tool arguments')
            evidence = stage.get('requires_tools', [])
            if not isinstance(evidence, list) or not set(evidence) <= names or before in evidence:
                raise ValueError('invalid interaction tool evidence dependencies')
            ContractUserSimulator.validate_evidence_requirements(stage)
            for requirement in stage.get('requires_evidence', []):
                if 'arguments' in requirement:
                    from env_factory.sandbox_runtime import validate_json_schema
                    tool = next(tool['function'] for tool in tools
                                if tool['function']['name'] == requirement['tool'])
                    validate_json_schema(tool['parameters'], requirement['arguments'],
                                         'interaction evidence arguments')
            if 'options_from' in stage:
                from env_factory.sandbox_runtime import observed_interaction_options
                if kind not in {'option_selection', 'correction', 'constraint_update'} or not structured_requests:
                    raise ValueError('options_from requires a structured choice, correction or constraint_update stage')
                observed_interaction_options(stage, [])  # Validate binding syntax without inventing observations.
                producer = stage['options_from']['tool']
                if producer not in evidence or producer == before:
                    raise ValueError('options_from producer must be a preceding requires_tools dependency')
            if 'context_from' in stage:
                from env_factory.sandbox_runtime import observed_interaction_options
                observed_interaction_options({'context_from': stage['context_from']}, [])
                if stage['context_from']['tool'] not in evidence or stage['context_from']['tool'] == before:
                    raise ValueError('context_from producer must be a preceding requires_tools dependency')
            hidden = stage.get('private_fact')
            bounded_choice = kind == 'option_selection' or 'options_from' in stage
            def privacy_issue(message, surface, evidence):
                if privacy_findings is None:
                    raise ValueError(message)
                privacy_findings.append({'variant_index': index, 'stage_id': key,
                    'private_fact': hidden, 'surface': surface, 'evidence': evidence,
                    'status': 'requires_independent_privacy_review'})
            if kind in {'information_required', 'option_selection', 'constraint_update', 'correction'}:
                if not isinstance(hidden, (str, int, float, bool)) or (isinstance(hidden, str) and not hidden) or (isinstance(hidden, float) and not math.isfinite(hidden)):
                    raise ValueError('private_fact must be a finite scalar')
                fact_text = hidden if isinstance(hidden, str) else json.dumps(hidden)
                pattern = (r'(?<![\d.])' + re.escape(fact_text) + r'(?![\d.])' if not isinstance(hidden, str)
                           else re.escape(fact_text))
                if not re.search(pattern, stage['user_reply']):
                    raise ValueError('interaction requires a concrete disclosed private_fact')
                if re.search(pattern, json.dumps(description.get('public_input', {}), ensure_ascii=False)):
                    matches = []
                    def locate(value, path):
                        if isinstance(value, dict):
                            for field, child in value.items():
                                locate(child, path + "." + str(field))
                        elif isinstance(value, list):
                            for position, child in enumerate(value):
                                locate(child, path + f"[{position}]")
                        else:
                            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            match = re.search(pattern, text)
                            if match and len(matches) < 3:
                                matches.append({"path": path,
                                    "excerpt": text[max(0, match.start()-50):match.end()+50]})
                    locate(description.get('public_input', {}), 'description.public_input')
                    privacy_issue('private interaction fact already appears in initial user input: '
                        + json.dumps({"variant_index": index, "stage_id": key, "matches": matches,
                            "interaction_kind": kind,
                            "options": stage.get('options', []) if bounded_choice else [],
                            "interpretation": (
                                "A candidate option appearing publicly does not prove the user's selection was disclosed. "
                                "Use structural preview and independent privacy review to distinguish a menu from "
                                "an already selected answer. Preserve legitimate public options; do not remove them "
                                "merely to avoid a lexical match."
                                if bounded_choice else
                                "Check whether the public text already supplies the information requested by this stage."),
                            "hint": "Compiled public input includes published answer schema and semantic criteria; inspect these as well as the original message."}, ensure_ascii=False),
                        'compiled_public_input', matches)
                if not bounded_choice and re.search(pattern, stage['reference_response']):
                    privacy_issue('reference question must not pre-disclose the private fact',
                                  'reference_response', stage['reference_response'])
                if re.search(pattern, stage['retry_reply']):
                    privacy_issue('retry reply must not leak undisclosed private fact',
                                  'retry_reply', stage['retry_reply'])
                binding = stage.get('bind_to', {})
                if not isinstance(binding, dict) or binding.get('tool') != before or not binding.get('argument'):
                    raise ValueError('private interaction fact requires bind_to tool/argument at disclosure boundary')
                bound_tool = next(tool['function'] for tool in tools if tool['function']['name'] == before)
                parameter = bound_tool.get('parameters', {}).get('properties', {}).get(binding['argument'], {})
                if ('const' in parameter and parameter['const'] == hidden
                        or parameter.get('enum') == [hidden]):
                    raise ValueError('INTERACTION_SINGLETON_DISCLOSURE: ' + json.dumps({
                        'variant_index': index, 'stage_id': key, 'tool': before,
                        'argument': binding['argument'],
                        'reason': 'Public parameter admits only the supposedly private user choice; publish its domain, then enforce the disclosed choice at runtime.'}, ensure_ascii=False))
                binding_key = (before, binding['argument'])
                if binding_key in bindings:
                    raise ValueError('redundant interaction asks for the same argument twice')
                bindings.add(binding_key)
                if kind in {'constraint_update', 'correction'}:
                    previous = stage.get('previous_fact')
                    previous_text = previous if isinstance(previous, str) else json.dumps(previous)
                    if previous is None or previous == hidden or not previous_text or previous_text not in disclosed_text:
                        raise ValueError('update/correction requires a previously disclosed different fact')
                from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
                direct = any(step['tool_name'] == before and DeclarativeMetricEvaluator._json_value_equal(
                    step.get('arguments', {}).get(binding['argument']), hidden) for step, _ in references)
                def captured(value, available):
                    if not isinstance(value, dict) or set(value) not in ({'$ref'}, {'$expr'}):
                        return False
                    names = []
                    def walk(item):
                        if isinstance(item, dict):
                            if set(item) == {'$ref'}:
                                names.append(item['$ref'])
                            else:
                                for child in item.values():
                                    walk(child)
                        elif isinstance(item, list):
                            for child in item:
                                walk(child)
                    walk(value)
                    return bool(names) and all(isinstance(name, str) and name in available for name in names)
                deferred = (not direct and capture_bindings is not None and any(
                    step['tool_name'] == before and captured(step.get('arguments', {}).get(binding['argument']), available)
                    for step, available in references))
                if not direct and not deferred and not inactive_or_refused(key, before):
                    raise ValueError('disclosed private fact does not match executable reference argument')
                if deferred:
                    capture_bindings.append({'script_id': f'script-{index + 1}', 'stage_id': key,
                                             'tool': before, 'argument': binding['argument'], 'value': hidden})
                for target in reuse:
                    schema = next(tool['function']['parameters'] for tool in tools
                                  if tool['function']['name'] == target['tool'])
                    pair = (target['tool'], target['argument'])
                    if pair in bindings or target['argument'] not in schema.get('required', []):
                        raise ValueError('INTERACTION_REUSE_INVALID: reused input must be unique and required')
                    bindings.add(pair)
                    matching = [(step, available) for step, available in references
                                if step['tool_name'] == target['tool']]
                    if not matching and inactive_or_refused(key, target['tool']):
                        continue
                    if not matching or capture_bindings is None:
                        raise ValueError('INTERACTION_REUSE_INVALID: reused input needs executable process enforcement')
                    for step, available in matching:
                        value = step.get('arguments', {}).get(target['argument'])
                        if not (DeclarativeMetricEvaluator._json_value_equal(value, hidden) or captured(value, available)):
                            raise ValueError('INTERACTION_REUSE_INVALID: reference changed the disclosed input')
                    for scenario in successes:
                        if index not in scenario_script_indices(scenario, count):
                            continue
                        called = set()
                        for step in scenario['steps']:
                            if step.get('operation') == 'reset':
                                called.clear()
                            if step.get('operation') == 'tool_call':
                                if step['tool_name'] == target['tool'] and before not in called:
                                    raise ValueError('INTERACTION_REUSE_INVALID: reuse precedes disclosure tool')
                                called.add(step['tool_name'])
                    capture_bindings.append({'script_id': f'script-{index + 1}', 'stage_id': key,
                                             **target, 'value': hidden})
            if bounded_choice:
                options = stage.get('options', [])
                if not isinstance(options, list) or len(set(options)) < 2 or hidden not in options:
                    raise ValueError('option selection requires distinct options and an actual selected option')
                missing_options = [option for option in options
                                   if str(option) not in stage['reference_response']]
                if missing_options and not observed_reference:
                    raise ValueError('selection reference must present every option: ' + json.dumps({
                        'path': f'interaction_contract.variants[{index}].stages',
                        'stage_id': key, 'field': 'reference_response',
                        'missing_options': missing_options,
                        'actual': stage['reference_response'],
                        'hint': 'Present the declared alternatives in the question before asking the user to choose.'
                    }, ensure_ascii=False))
            disclosed_text += '\n' + stage['user_reply']
            stage['before_tool'] = before
        if structured_requests:
            from env_factory.sandbox_runtime import structured_interaction_request
            for stage in stages:
                stage['reference_response'] = json.dumps({'question': stage['reference_response'],
                    'request': structured_interaction_request(stage)}, ensure_ascii=False)
        states = [{'state_id':f'stage-{i}', 'user_behavior':stage['purpose'], 'terminal':False} for i,stage in enumerate(stages)]
        states += [{'state_id':'review', 'user_behavior':'检查实际业务目标是否完成', 'terminal':False}, {'state_id':'done', 'user_behavior':'接受结果', 'terminal':True}]
        outcomes = {'correction':'user_correction', 'constraint_update':'user_correction', 'execution_confirmation':'information_required', 'option_selection':'information_required', 'information_required':'information_required'}
        transitions = [{'transition_id':stage['id'], 'from_state':f'stage-{i}', 'to_state':f'stage-{i+1}' if i+1<len(stages) else 'review', 'condition':stage['purpose'], 'outcome_category':outcomes[stage['kind']], 'should_end':False, 'updates':{}} for i,stage in enumerate(stages)]
        transitions.append({'transition_id':'complete', 'from_state':'review', 'to_state':'done', 'condition':'业务目标和所有必要交互均已满足', 'outcome_category':'goal_satisfied', 'should_end':True, 'updates':{}})
        scripts.append({'initial_state':'stage-0', 'states':states, 'transitions':transitions, 'variables':copy.deepcopy(variables),
                        'recovery_policy':{'max_recoveries':2, 'user_behavior':'请完成当前必要交互。', 'handled_outcomes':['agent_off_topic','agent_premature_completion','unrecognized']}, 'script_id':f'script-{index+1}', 'goal':description['task'],
                        'user_input':copy.deepcopy(description.get('public_input', {})),
                        'interaction_protocol':{'version':'1.0', 'request_mode':'structured_v1' if structured_requests else 'legacy_phrases', 'stages':stages}})
    if len(kinds) < 2:
        raise ValueError('interaction contract requires at least two interaction kinds')
    observed_domains = {(binding['tool'], binding['argument'])
        for script in scripts for stage in script['interaction_protocol']['stages']
        if 'options_from' in stage for binding in stage_bindings(stage)}
    for script in scripts:
        for stage in script['interaction_protocol']['stages']:
            if stage['kind'] not in {'correction', 'constraint_update'} or 'options_from' in stage:
                continue
            for binding in stage_bindings(stage):
                if (binding['tool'], binding['argument']) in observed_domains:
                    raise ValueError('INTERACTION_OPTION_DOMAIN: '
                        f"script={script['script_id']} stage={stage['id']} "
                        f"tool={binding['tool']} argument={binding['argument']}; "
                        'another variant constrains this choice using observed options. '
                        'Preserve an explicit options_from binding and baseline options on '
                        'corrections/updates; changing interaction kind cannot remove eligibility.')
    return scripts


def _select_disclosed_capture_rows(option):
    """Locate the chosen row by its disclosed key, retaining producer provenance."""
    selectors = {}
    captures = option.get('captures', [])
    for argument, selected in option.get('disclosed_arguments', {}).items():
        reference = option.get('arguments', {}).get(argument)
        if (not isinstance(reference, dict) or set(reference) != {'$ref'}
                or not isinstance(selected, (str, int, float, bool))
                or isinstance(selected, float) and not math.isfinite(selected)):
            continue
        capture = next((item for item in captures if item.get('name') == reference['$ref']), None)
        if capture is None:
            continue
        match = re.fullmatch(r'(\$\.[A-Za-z_][A-Za-z_0-9]*\[\d+\])\.([A-Za-z_][A-Za-z_0-9]*)',
                             capture.get('path', ''))
        if match is None:
            continue
        prefix, field = match.groups()
        key = (capture['tool_name'], prefix)
        selectors.setdefault(key, prefix[:prefix.rfind('[')] +
            '[?(@.' + field + '==' + json.dumps(selected, ensure_ascii=False, allow_nan=False) + ')]')
    for capture in captures:
        for (producer, prefix), selector in selectors.items():
            if capture.get('tool_name') == producer and capture.get('path', '').startswith(prefix + '.'):
                capture['path'] = selector + capture['path'][len(prefix):]
                break


def bind_captured_interaction_arguments(rules, scripts, capture_bindings):
    """Conjoin capture provenance with actual user choice on the SAME process call."""
    tools = {binding['tool'] for binding in capture_bindings}
    if not tools:
        return
    covered = set()
    for rule in rules:
        if rule.get('operator') != 'contains_tool_call':
            continue
        expected = rule['expected']
        if expected.get('tool_name') not in tools:
            continue
        original = expected.get('by_user_script')
        fallback = {key: value for key, value in expected.items() if key != 'by_user_script'}
        scoped = {}
        for script in scripts:
            identity = script['script_id']
            options = ([*original.get(identity, []), *original.get('*', [])]
                       if original is not None else [fallback])
            scoped[identity] = copy.deepcopy(options)
            for option in scoped[identity]:
                selected = {binding['argument']: stage['private_fact']
                    for stage in script['interaction_protocol']['stages']
                    for binding in stage_bindings(stage)
                    if binding.get('tool') == option.get('tool_name')}
                option['disclosed_arguments'] = selected
                _select_disclosed_capture_rows(option)
                covered.update((identity, option['tool_name'], argument) for argument in selected)
        expected['by_user_script'] = scoped
    if any((b['script_id'], b['tool'], b['argument']) not in covered for b in capture_bindings):
        raise ValueError('INTERACTION_CAPTURE_UNENFORCED: captured user choice needs a script-scoped process rule')


def interleave_scenarios(scenarios, scripts, *, execution_routes=()):
    """Expand business success cases into seed-selected executable dialogue cases."""
    import random
    result = []
    for scenario in scenarios:
        if scenario.get('kind') != 'goal_success' and not scenario.get('derive_interactions'):
            if 'user_script_indices' in scenario:
                raise ValueError('INTERACTION_REFERENCE_SCOPE: scoped negative scenarios require derive_interactions')
            result.append(scenario)
            continue
        for index, script in enumerate(scripts):
            if index not in scenario_script_indices(scenario, len(scripts)):
                continue
            seed = next(seed for seed in range(10000) if random.Random(seed).randrange(len(scripts)) == index)
            expanded = copy.deepcopy(scenario)
            expanded.pop('derive_interactions', None)
            expanded.pop('user_script_indices', None)
            expanded['scenario_id'] += f'-{script["script_id"]}'
            pending = list(script['interaction_protocol']['stages'])
            route = None
            if execution_routes:
                route = next((r for r in execution_routes if r['id'] == scenario.get('execution_route')), None)
                if route is None:
                    raise ValueError('EXECUTION_ROUTE_INVALID: reference needs a declared execution_route; '
                        f'scenario={scenario.get("scenario_id")} kind={scenario.get("kind")}; '
                        'manual negatives with derive_interactions must declare the route whose dialogue '
                        f'they exercise; available routes={[r["id"] for r in execution_routes]}')
                pending = [stage for stage in pending if stage['id'] in route['required_stages']]
            steps = [] if any(step.get('operation') == 'reset' for step in expanded['steps']) else [{'operation':'reset', 'body':{'seed':seed}}]
            for step in expanded['steps']:
                if step.get('operation') == 'reset':
                    step.setdefault('body', {})['seed'] = seed
                anchor = step.get('tool_name') if step.get('operation') == 'tool_call' else '__answer__' if step.get('operation') == 'agent_response' else None
                while pending and (pending[0]['before_tool'] == anchor or (
                        anchor == '__answer__' and route and route['writes'] == 'forbidden'
                        and pending[0]['before_tool'] not in route['required_tools'])):
                    # A refusal/no-write route can require a user decision while
                    # omitting the write protected by that decision. Execute the
                    # declared dialogue before the answer, never a dummy write.
                    stage = pending.pop(0)
                    refusal = (route or {}).get('when', {}).get('user_rejection', {})
                    expected = ({'expected_branch': refusal['branch'], 'expected_stage': None}
                                if refusal.get('stage') == stage['id'] else {'expected_stage':stage['id']})
                    observed_menu = ({'observed_options_from': copy.deepcopy(stage['options_from'])}
                                     if '{{observed_options}}' in stage['reference_response'] else {})
                    observed_context = ({'observed_context_from': copy.deepcopy(stage['context_from'])}
                                        if 'context_from' in stage else {})
                    steps.append({'operation':'dialogue_turn', 'content':stage['reference_response'],
                                  **observed_menu, **observed_context, **expected, 'expected_status':200})
                steps.append(step)
            if pending:
                raise ValueError('interaction stage ordering cannot be reached in reference tools; '
                    f'scenario={scenario.get("scenario_id")} script={script["script_id"]} '
                    f'pending_stages={[stage["id"] for stage in pending]}')
            expanded['steps'] = steps
            result.append(expanded)
    return result
