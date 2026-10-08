"""Validate business structure before spending on data and executable implementation."""
from __future__ import annotations

from .requirement_coverage import resolve_pointer


GUIDE = '''Write blueprint.json version 1.0 only. Do not implement source.json yet.
Read request.json. Design an original goal grounded in its graph, intent and language.
Schema:
- description: the complete eventual source description (task, goal, expected_result,
  task_intent, complexity, requirements input/output_modalities, public_input).
- environment_mode: one requested available mode.
- requirements: nonempty list of {id, statement}, one per actual business obligation.
- tools: list of {name, purpose, operation, inputs: {argument_name: origin}}.
  Optional result_field names a non-default read result envelope (default records).
  operation: select, aggregate_count, insert, update or delete.
  reference_data tools are read-only; stateful requires a write operation.
  origin is {kind:public, pointer:/description/public_input/...},
  {kind:user, stage:interaction_id}, {kind:tool, tool:producer_name, field:output_field},
  or {kind:derived, sources:[origin,...]}. Inputs must be discoverable, not guessed.
  Reuse an already disclosed parameter in a downstream dependent tool with the
  same {kind:user, stage:...} origin and argument name; do not invent a tool echo.
  In source interaction stages, declare reuse_bindings:[{tool,argument}] for these
  later consumers. A select returns table records and count, never root-level
  request arguments. If a producer field has the same name as one of its inputs,
  use an explicit path such as $.records[0].id for a real returned table column.
- interactions: 2..5 {id, kind, before_tool, argument, purpose} records.
  Kinds: information_required, option_selection, constraint_update, correction,
  execution_confirmation. argument is required except for execution_confirmation.
  Include at least two distinct interaction kinds, matching the implementation gate.
  Optional allowed_kinds lists acceptable variants of the same parameter disclosure.
  Include kind in that list. Never mix execution_confirmation with disclosure kinds.
  execution_confirmation must protect a write tool.
At least two necessary tools with a real tool-result dependency; no artificial splitting.
Dependencies must be acyclic. A user stage cannot depend on the tool it blocks.
Keep this blueprint concise. State business obligations, not fixture-specific answers.
Before committing the description, account for each valid terminal outcome: success,
no eligible or unique lookup result, and refusal when the business goal permits it.
Refusal must reflect a private user preference and an actual observed user reply,
not an assistant phrase that lets the policy force a rewarded early stop. Declare
the applicable read prefix and truthful incomplete answer for each refusal route.
Express those obligations in requirements. A downstream choice or write is conditional
on its prerequisite evidence; a truthful no-result outcome must not require choosing
from an empty list or completing an inapplicable write. Plan these alternatives before
authoring fixtures, rather than adding them only after independent review rejects the task.
Put decision rules needed by the policy into description.public_input now, using
natural-language materials when needed: this description is preserved during
implementation. State whether identity must be unique before eligibility filtering,
or whether uniqueness is tested over eligible candidates after filtering. For a
user-selected item, scope ambiguity and compatibility to that actual choice unless
the public goal explicitly requires a catalog-wide condition. Define current versus
historical eligibility and any required ordering of answer collections. Requirements,
public rules, tool predicates, terminal routes and explanation criteria must describe
the same decision. Do not publish hidden user preferences or fixture answers.
Distinguish fixed public policy from changeable business facts. A public eligibility
threshold/status must remain fixed when business rows change; do not design it as a
mutable policy-table lookup. Plan any required fixed-filter tool arguments with
single-value enums before freezing the tool interface. Business-dependent limits
may come from data only when the public task explicitly defines that dependency.
If a user corrects a choice drawn from a tool-provided domain, the correction still
uses that observed domain; changing interaction kind does not remove eligibility.
Use the same obligation IDs/statements, description, tool names/input names and stage
identities in the later implementation. Variants may change private values and branches.
Do not write business tables, reward code or detailed reference trajectories yet.
Do not modify request.json, AUTHORING.md, platform files or access credentials/network.
'''


def _unique(items, key, label):
    if not isinstance(items, list) or not items:
        raise ValueError(f'BLUEPRINT: {label} must be a nonempty list')
    result = {}
    for item in items:
        value = item.get(key) if isinstance(item, dict) else None
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f'BLUEPRINT: {label} require unique nonempty {key}')
        result[value] = item
    return result


def validate_blueprint(blueprint, request):
    if not isinstance(blueprint, dict) or blueprint.get('version') != '1.0':
        raise ValueError('BLUEPRINT: version 1.0 required')
    description = blueprint.get('description', {})
    if not isinstance(description, dict):
        raise ValueError('BLUEPRINT: description must be an object')
    for key in ('task', 'goal', 'expected_result', 'task_intent', 'complexity'):
        if not isinstance(description.get(key), str) or not description[key].strip():
            raise ValueError(f'BLUEPRINT: description.{key} is required')
    public = description.get('public_input', {})
    if not isinstance(public, dict):
        raise ValueError('BLUEPRINT: public_input must be an object')
    if not isinstance(public.get('initial_user_message'), str) or not public['initial_user_message'].strip():
        raise ValueError('BLUEPRINT: initial public input is required')
    from env_factory.tasks.task_routing import training_contract
    policy = training_contract(request['training_category'])
    if description['task_intent'] not in policy['allowed_intents']:
        raise ValueError('BLUEPRINT: intent is incompatible with the requested category')
    if request.get('task_intent') and request['task_intent'] != description['task_intent']:
        raise ValueError('BLUEPRINT: requested intent changed')
    mode = blueprint.get('environment_mode')
    if mode not in request['available_environment_modes'] or mode not in policy['allowed_environment_modes']:
        raise ValueError('BLUEPRINT: environment mode is unavailable or incompatible')
    requirements = _unique(blueprint.get('requirements'), 'id', 'requirements')
    tools = _unique(blueprint.get('tools'), 'name', 'tools')
    stages = _unique(blueprint.get('interactions'), 'id', 'interactions')
    limits = policy['business_tools']
    if len(tools) < limits['min'] or (limits['max'] is not None and len(tools) > limits['max']):
        raise ValueError('BLUEPRINT: business tool count is incompatible')
    if not 2 <= len(stages) <= 5:
        raise ValueError('BLUEPRINT: multi-step design needs 2..5 meaningful user interactions')
    for item in requirements.values():
        if not isinstance(item.get('statement'), str) or not item['statement'].strip():
            raise ValueError('BLUEPRINT: empty business obligation')
    from .interaction_contract import KINDS
    writes = set()
    for name, tool in tools.items():
        if tool.get('operation') not in ('select', 'aggregate_count', 'insert', 'update', 'delete'):
            raise ValueError('BLUEPRINT: unsupported tool operation')
        if not isinstance(tool.get('inputs'), dict):
            raise ValueError('BLUEPRINT: tool inputs must be an object')
        if 'result_field' in tool and (tool['operation'] not in {'select', 'aggregate_count'}
                or not isinstance(tool['result_field'], str) or not tool['result_field']):
            raise ValueError('BLUEPRINT: invalid read result_field')
        if tool['operation'] in ('insert', 'update', 'delete'):
            writes.add(name)
    if mode == 'reference_data' and writes:
        raise ValueError('BLUEPRINT: reference_data must be read-only')
    if mode == 'stateful' and not writes:
        raise ValueError('BLUEPRINT: stateful requires a write operation')
    for stage in stages.values():
        if (not isinstance(stage.get('kind'), str) or stage['kind'] not in KINDS
                or not isinstance(stage.get('before_tool'), str) or stage['before_tool'] not in tools
                or not isinstance(stage.get('purpose'), str) or not stage['purpose'].strip()):
            raise ValueError('BLUEPRINT: invalid user stage kind, tool or purpose')
        allowed = stage.get('allowed_kinds', [stage['kind']])
        if (not isinstance(allowed, list) or not allowed
                or any(not isinstance(kind, str) or kind not in KINDS for kind in allowed)
                or len(set(allowed)) != len(allowed) or stage['kind'] not in allowed
                or ('execution_confirmation' in allowed and allowed != ['execution_confirmation'])):
            raise ValueError('BLUEPRINT: invalid allowed interaction kinds')
        if stage['kind'] == 'execution_confirmation' and stage['before_tool'] not in writes:
            raise ValueError('BLUEPRINT: execution confirmation must protect a write tool')
        if stage['kind'] != 'execution_confirmation' and (not isinstance(stage.get('argument'), str)
                or stage['argument'] not in tools[stage['before_tool']]['inputs']):
            raise ValueError('BLUEPRINT: user disclosure must bind a real tool input')
    if len({stage['kind'] for stage in stages.values()}) < 2:
        raise ValueError('BLUEPRINT: at least two distinct interaction kinds are required')
    dependencies = {name: set() for name in tools}
    reused_inputs = []
    def origin(value, consumer, argument, depth=0):
        if depth > 8 or not isinstance(value, dict):
            raise ValueError('BLUEPRINT: invalid or excessively nested input provenance')
        kind = value.get('kind')
        if kind == 'public':
            pointer = value.get('pointer', '')
            if not isinstance(pointer, str) or not pointer.startswith('/description/public_input/'):
                raise ValueError('BLUEPRINT: public input must cite the public surface')
            resolve_pointer(blueprint, pointer)
        elif kind == 'user':
            if not isinstance(value.get('stage'), str):
                raise ValueError('BLUEPRINT: user input requires a stage identifier')
            stage = stages.get(value.get('stage'), {})
            # A disclosed choice can inform several derived arguments of the
            # same action (e.g. selected format -> offer ID and price). Direct
            # inputs still require an exact binding; derived inputs must cite
            # a real disclosure before this consumer, never an approval stage.
            if (stage.get('kind') == 'execution_confirmation'
                    or (depth == 0 and stage.get('argument') != argument)):
                raise ValueError('BLUEPRINT: user input provenance is not bound to this argument')
            if stage.get('before_tool') != consumer:
                if depth or not stage:
                    raise ValueError('BLUEPRINT: user input provenance is not bound to this argument')
                reused_inputs.append({'stage': stage['id'], 'producer': stage['before_tool'],
                                      'tool': consumer, 'argument': argument})
        elif kind == 'tool':
            producer = value.get('tool')
            if not isinstance(producer, str) or producer not in tools or producer == consumer or not isinstance(value.get('field'), str) or not value['field']:
                raise ValueError('BLUEPRINT: invalid producer tool/field')
            from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
            try:
                tokens = DeclarativeMetricEvaluator.path_tokens(value['field'])
            except SandboxError as exc:
                raise ValueError('BLUEPRINT: unsupported producer field path') from exc
            if (tools[producer]['operation'] in {'select', 'aggregate_count'}
                    and value['field'] in tools[producer]['inputs']):
                raise ValueError('BLUEPRINT: ambiguous producer field equals a request argument; '
                    'select does not echo request arguments. Use an explicit returned-column path '
                    'or reuse the original user disclosure with kind:user.')
            if (tools[producer]['operation'] == 'select' and value['field'].startswith('$.')
                    and tokens and tokens[0] not in [('field', tools[producer].get('result_field', 'records')), ('field', 'count')]):
                raise ValueError('BLUEPRINT: select output has records/count, not arbitrary root fields')
            dependencies[consumer].add(producer)
        elif kind == 'derived':
            if not isinstance(value.get('sources'), list) or not value['sources']:
                raise ValueError('BLUEPRINT: derived input requires provenance sources')
            for child in value['sources']:
                origin(child, consumer, argument, depth + 1)
        else:
            raise ValueError('BLUEPRINT: unknown input origin')
    for name, tool in tools.items():
        if not isinstance(tool.get('purpose'), str) or not tool['purpose'].strip() or not isinstance(tool.get('inputs'), dict):
            raise ValueError('BLUEPRINT: tools need a business purpose and inputs mapping')
        for argument, value in tool['inputs'].items():
            if not isinstance(argument, str) or not argument:
                raise ValueError('BLUEPRINT: invalid argument name')
            origin(value, name, argument)
    remaining = {name: set(deps) for name, deps in dependencies.items()}
    order = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError('BLUEPRINT: cyclic tool-result dependencies')
        order.extend(ready)
        remaining = {name: deps - set(ready) for name, deps in remaining.items() if name not in ready}
    if request['training_category'] == 'multi_step_agentic' and not any(dependencies.values()):
        raise ValueError('BLUEPRINT: multi-step tasks need a real tool-result dependency')
    def ancestors(name):
        return dependencies[name] | {ancestor for parent in dependencies[name] for ancestor in ancestors(parent)}
    for reuse in reused_inputs:
        if reuse['producer'] not in ancestors(reuse['tool']):
            raise ValueError('BLUEPRINT: reused user input must follow its disclosure tool through a real dependency')
    return {'version': '1.0', 'passed': True, 'scope': 'business_structure_not_executable_certification',
            'tool_order': order, 'dependencies': {name: sorted(deps) for name, deps in dependencies.items()},
            'requirement_ids': list(requirements), 'reused_user_inputs': reused_inputs}


def validate_implementation(blueprint, source, request):
    evidence = validate_blueprint(blueprint, request)
    if source.get('description') != blueprint['description']:
        raise ValueError('BLUEPRINT_DRIFT: original description changed')
    if source.get('environment_plan', {}).get('mode') != blueprint['environment_mode']:
        raise ValueError('BLUEPRINT_DRIFT: environment mode changed')
    expected = {item['id']: item['statement'] for item in blueprint['requirements']}
    actual = {item['id']: item['statement'] for item in source.get('business_requirements', [])}
    if actual != expected:
        raise ValueError('BLUEPRINT_DRIFT: business obligations changed')
    tools = {tool['function']['name']: tool['function']['parameters'] for tool in source['tools']}
    if set(tools) != {tool['name'] for tool in blueprint['tools']}:
        raise ValueError('BLUEPRINT_DRIFT: business tools changed')
    for tool in blueprint['tools']:
        if set(tools[tool['name']].get('properties', {})) != set(tool['inputs']):
            raise ValueError('BLUEPRINT_DRIFT: tool input names changed')
        implementation = next((item for item in source.get('tool_implementations', [])
                               if item['tool_name'] == tool['name']), {})
        if implementation.get('operation') != tool['operation']:
            raise ValueError('BLUEPRINT_DRIFT: tool operation changed')
        if 'result_field' in tool and implementation.get('result_field', 'records') != tool['result_field']:
            raise ValueError('BLUEPRINT_DRIFT: read result envelope changed')
    actual_deps = {item['action_name']: set(item.get('dependencies', [])) for item in source['capability_plan']}
    for name, deps in evidence['dependencies'].items():
        if not set(deps) <= actual_deps.get(name, set()):
            raise ValueError('BLUEPRINT_DRIFT: required tool-result dependency missing')
    variants = source.get('interaction_contract', {}).get('variants', [])
    if len(variants) != request['user_script_count']:
        raise ValueError('BLUEPRINT_DRIFT: missing interaction variants')
    for variant in variants:
        stages = {stage['id']: stage for stage in variant.get('stages', [])}
        for expected_stage in blueprint['interactions']:
            actual_stage = stages.get(expected_stage['id'], {})
            if (actual_stage.get('before_tool') != expected_stage['before_tool']
                    or actual_stage.get('kind') not in expected_stage.get('allowed_kinds', [expected_stage['kind']])):
                raise ValueError('BLUEPRINT_DRIFT: interaction obligation changed')
            if expected_stage['kind'] != 'execution_confirmation' and actual_stage.get('bind_to') != {
                    'tool': expected_stage['before_tool'], 'argument': expected_stage['argument']}:
                raise ValueError('BLUEPRINT_DRIFT: interaction parameter binding changed')
        for reuse in evidence['reused_user_inputs']:
            if {'tool': reuse['tool'], 'argument': reuse['argument']} not in stages[reuse['stage']].get('reuse_bindings', []):
                raise ValueError('BLUEPRINT_DRIFT: reused user disclosure needs an executable reuse_binding')
    def tool_origins(value, derived=False):
        if value['kind'] == 'tool':
            yield {**value, '_derived': derived}
        elif value['kind'] == 'derived':
            for child in value['sources']:
                yield from tool_origins(child, True)
    intended = {(tool['name'], argument): list(tool_origins(origin))
                for tool in blueprint['tools'] for argument, origin in tool['inputs'].items()
                if list(tool_origins(origin))}
    observed = set()
    def refs(value):
        if isinstance(value, dict):
            if set(value) == {'$ref'}:
                yield value['$ref']
            else:
                for child in value.values():
                    yield from refs(child)
        elif isinstance(value, list):
            for child in value:
                yield from refs(child)
    for scenario in source.get('scenarios', []):
        if scenario.get('kind') != 'goal_success':
            continue
        captures = {}
        for step in scenario.get('steps', []):
            if step.get('operation') != 'tool_call':
                continue
            for argument, value in step.get('arguments', {}).items():
                key = (step['tool_name'], argument)
                if key not in intended:
                    continue
                actual = []
                for reference in refs(value):
                    matches = [name for name in captures if reference == name or reference.startswith(name + '.')
                               or reference.startswith(name + '[')]
                    if matches:
                        alias = max(matches, key=len)
                        producer, path = captures[alias]
                        actual.append((producer, path + reference[len(alias):]))
                for origin in intended[key]:
                    field = origin['field']
                    def matches_field(path):
                        from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
                        expected = DeclarativeMetricEvaluator.path_tokens(field)
                        consumed = DeclarativeMetricEvaluator.path_tokens(path)
                        if not expected:
                            return bool(origin['_derived']) or not consumed
                        if origin['_derived']:
                            # Selecting a row/field from declared query results
                            # retains that provenance; sibling fields do not.
                            starts = [0] if field.startswith('$') else range(len(consumed))
                            return any(consumed[i:i + len(expected)] == expected for i in starts)
                        return consumed == expected if field.startswith('$') else consumed[-len(expected):] == expected
                    if not any(name == origin['tool'] and matches_field(path) for name, path in actual):
                        raise ValueError(f'BLUEPRINT_DRIFT: {key} does not consume the planned producer field')
                observed.add(key)
            captures.update({name: (step['tool_name'], path) for name, path in step.get('capture', {}).items()})
    if set(intended) - observed:
        raise ValueError('BLUEPRINT_DRIFT: missing executable tool-result provenance')
    return evidence
