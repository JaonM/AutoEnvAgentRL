"""Reject unexecutable acceptance contracts before execution or judging."""
from env_factory.contracts.tool_chain_contract import captured_schema


def measured_component_failure(scenario, execution, metrics):
    """Recognize explicit low outcome assertions on the final observed reward.

    The runner has already enforced every authored assertion. Only a terminal
    reward observation can replace the default aggregate-failure check; an early
    empty episode or a process-only penalty is insufficient.
    """
    import math
    steps = scenario.get('steps', [])
    if not steps or steps[-1].get('operation') != 'reward':
        return None
    last = steps[-1]
    outcomes = {m['id'] for m in metrics if m.get('category') == 'outcome'}
    targets = set()
    for assertion in scenario.get('assertions', []):
        source = assertion.get('source')
        if source == 'variables' or (str(source).startswith('step:')
                and source != f"step:{last.get('step_id')}"):
            continue
        expected = assertion.get('expected')
        if (assertion.get('operator') not in {'eq', 'lte', 'lt'}
                or type(expected) not in {int, float} or not math.isfinite(expected)
                or not 0 <= expected <= .2):
            continue
        for metric_id in outcomes:
            if assertion.get('path') == f'$.components.{metric_id}':
                targets.add(metric_id)
    if not targets:
        return None
    observed = execution['history'][-1]['body']
    total = observed.get('reward')
    components = {name: observed.get('components', {}).get(name) for name in sorted(targets)}
    passed = (type(total) in {int, float} and math.isfinite(total) and 0 <= total < 1 - 1e-9
              and all(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= .2
                      and name not in observed.get('inactive_components', [])
                      for name, value in components.items()))
    return {'scenario_id': scenario.get('scenario_id'), 'scope': 'final_observed_outcome_components',
            'reward': total, 'components': components, 'passed': passed}


def validate_scenario_fixture_scope(scenarios):
    """Fixture changes belong to isolated probes, not portable HTTP scenarios."""
    for scenario in scenarios:
        identity = scenario.get('scenario_id')
        for field in ('user_variables', 'initial_changes'):
            if field in scenario:
                raise ValueError(f'SCENARIO_FIXTURE_SCOPE: scenario={identity} {field} '
                    'is not applied by acceptance execution. Use route_probes for isolated '
                    'fixture changes; use declared interaction variants and user_script_indices '
                    'to select actual user behavior. Do not rely on a seed or scenario field '
                    'to override private user state.')
        for step in scenario.get('steps', []):
            if step.get('operation') == 'mutate_business_state':
                raise ValueError(f'SCENARIO_FIXTURE_SCOPE: scenario={identity} mutate_business_state '
                    'is an internal diagnostic callback, unavailable over the production HTTP interface. '
                    'Use route_probes.initial_changes or business-condition probes to vary fixtures; '
                    'acceptance cases must exercise public business tools and actual rewards.')


def validate_reward_assertion_paths(scenarios, metrics, *, routed=False):
    properties = {'reward': {'type': 'number'}, 'raw_reward': {'type': 'number'},
                  'components': {'type': 'object', 'properties': {
                      m['id']: {'type': 'number'} for m in metrics}}}
    if routed:
        properties.update(execution_route={'type': 'string'},
                          inactive_components={'type': 'array', 'items': {'type': 'string'}})
    schema = {'type': 'object', 'properties': properties}
    for scenario in scenarios:
        rewards = {f"step:{step['step_id']}" for step in scenario.get('steps', [])
                   if step.get('operation') == 'reward' and step.get('step_id')}
        last_is_reward = bool(scenario.get('steps') and scenario['steps'][-1].get('operation') == 'reward')
        for assertion in scenario.get('assertions', []):
            source = assertion.get('source')
            # Named non-reward results and variables have their own schemas.
            if source not in rewards and not (last_is_reward and
                    source != 'variables' and not str(source).startswith('step:')):
                continue
            path = assertion.get('path', '$')
            if not isinstance(path, str) or captured_schema(schema, path) is None:
                raise ValueError(f"REWARD_ASSERTION_PATH: scenario={scenario.get('scenario_id')} "
                    f"path={path!r} is absent from reward output; use $.reward, $.raw_reward "
                    f"or $.components.<metric_id>; metric_ids={sorted(m['id'] for m in metrics)}")
