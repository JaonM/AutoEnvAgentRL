"""Derive negative business cases without duplicating their successful prerequisites."""
import copy

def derive_negatives(expanded, definitions):
    result = list(expanded)
    references = [case for case in expanded if case.get('kind') == 'goal_success']
    for definition in definitions:
        if definition.get('kind') != 'goal_failure' or 'steps' in definition:
            raise ValueError('derived negative requires goal_failure and no hand-written steps')
        allowed = {'scenario_id', 'kind', 'from_reference', 'argument_mutations', 'assertions'}
        if set(definition) - allowed or not isinstance(definition.get('scenario_id'), str) or not definition['scenario_id']:
            raise ValueError('derived negative requires a scenario_id and supported fields')
        if not isinstance(definition.get('assertions'), list) or not definition['assertions']:
            raise ValueError('derived negative requires explicit assertions')
        changes = definition.get('argument_mutations')
        if not isinstance(changes, list) or not changes:
            raise ValueError('derived negative requires explicit argument mutations')
        if not references:
            raise ValueError('derived negative requires executable success reference')
        for reference in references:
            case = copy.deepcopy(reference)
            case['scenario_id'] = definition['scenario_id'] + '--' + reference['scenario_id']
            case['kind'] = 'goal_failure'
            case['assertions'] = copy.deepcopy(definition['assertions'])
            changed = False
            seen = set()
            for mutation in changes:
                if (not isinstance(mutation, dict) or not {'tool_name', 'arguments'} <= set(mutation)
                        or set(mutation) - {'tool_name', 'arguments', 'expected_status'}
                        or not isinstance(mutation['tool_name'], str)
                        or not isinstance(mutation['arguments'], dict) or not mutation['arguments']):
                    raise ValueError('negative mutation requires a tool name and nonempty argument mapping')
                if 'expected_status' in mutation and (type(mutation['expected_status']) is not int
                        or not 100 <= mutation['expected_status'] <= 599):
                    raise ValueError('negative mutation expected_status must be an integer HTTP status')
                matching = [step for step in case['steps'] if step.get('operation') == 'tool_call' and step.get('tool_name') == mutation['tool_name']]
                if len(matching) != 1:
                    raise ValueError('negative mutation must select exactly one tool call')
                step = matching[0]
                if 'expected_status' in mutation:
                    step['expected_status'] = mutation['expected_status']
                for argument, value in mutation['arguments'].items():
                    key = (mutation['tool_name'], argument)
                    if key in seen or argument not in step.get('arguments', {}):
                        raise ValueError('duplicate or unknown negative argument')
                    seen.add(key)
                    changed |= step['arguments'][argument] != value
                    step['arguments'][argument] = copy.deepcopy(value)
            if not changed:
                raise ValueError('negative mutation must change reference')
            # This is a counterfactual of an independently executed positive.
            # Its changed query may make the inherited downstream dialogue
            # impossible. The runner must still observe every mutated call and
            # evaluate the original reward assertions on the actual trajectory.
            case['counterfactual_argument_tools'] = sorted({m['tool_name'] for m in changes})
            # Later realism compilation can duplicate a logical request for a
            # retry. Preserve the actual changed arguments so the runner can
            # verify every physical call, rather than assuming one call/tool.
            by_tool = {}
            for mutation in changes:
                by_tool.setdefault(mutation['tool_name'], {}).update(copy.deepcopy(mutation['arguments']))
            case['counterfactual_argument_mutations'] = by_tool
            result.append(case)
    return result
