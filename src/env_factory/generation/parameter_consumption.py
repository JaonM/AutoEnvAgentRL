"""Parameter uses enforced outside a declarative database operation."""
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator


def disclosed_parameter_uses(scripts, metrics, rules):
    """Require both a user binding and enforced script-scoped call matching.

    A declaration alone is insufficient. Every alternative process call for
    every script must constrain the parameter to its actual disclosed value.
    Ordinary fixture arguments are not a use of an otherwise ignored input.
    """
    from .interaction_contract import stage_bindings
    process_ids = {m['id'] for m in metrics if m.get('category') == 'process'}
    process_rules = [r for r in rules if r.get('metric_id') in process_ids
                     and r.get('operator') == 'contains_tool_call']
    shared = None
    for script in scripts:
        disclosed = {}
        for stage in script.get('interaction_protocol', {}).get('stages', []):
            if stage.get('kind') not in {'information_required', 'option_selection', 'correction', 'constraint_update'}:
                continue
            if 'private_fact' not in stage:
                continue
            for binding in stage_bindings(stage):
                disclosed[(binding['tool'], binding['argument'])] = stage['private_fact']
        consumed = set()
        for rule in process_rules:
            expected = rule.get('expected', {})
            scoped = expected.get('by_user_script')
            if not isinstance(scoped, dict):
                continue
            options = [*scoped.get(script['script_id'], []), *scoped.get('*', [])]
            if not options:
                continue
            for (tool, argument), value in disclosed.items():
                def constrains(option):
                    if option.get('tool_name') != tool or argument not in option.get('arguments', {}):
                        return False
                    actual = option['arguments'][argument]
                    guarded = option.get('disclosed_arguments', {})
                    return (DeclarativeMetricEvaluator._json_value_equal(actual, value)
                            or argument in guarded and DeclarativeMetricEvaluator._json_value_equal(guarded[argument], value))
                if all(constrains(option) for option in options):
                    consumed.add((tool, argument))
        shared = consumed if shared is None else shared & consumed
    return shared or set()


def lifecycle_parameter_uses(policy):
    """Only executable selector/guard operands count, never descriptive metadata."""
    used = set()
    def walk(value):
        if isinstance(value, dict):
            if set(value) == {'argument'} and isinstance(value['argument'], str):
                used.add(value['argument'])
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for guard in policy.get('guards', []):
        walk(guard.get('selector', {}))
        walk(guard.get('conditions', {}))
    walk(policy.get('transition', {}).get('selector', {}))
    return used
