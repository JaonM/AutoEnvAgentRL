"""Bounded no-model goal sensitivity diagnostic; does not judge business validity."""
from .runtime import Sandbox
from .contract import require
from .expressions import ContractError, evaluate, typed_equal


def temporal_diagnostics(compiled, *, limit=16):
    require(type(limit) is int and 0 <= limit <= 16, "temporal diagnostic limit must be 0..16")
    source = compiled["source"]
    values, visited = [], 0
    def visit(value):
        nonlocal visited
        if visited >= 4096: return
        visited += 1
        if type(value) in (int, float) and 0 < value <= 86400 and value not in values:
            values.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                if visited >= 4096: break
                visit(v)
        elif isinstance(value, list):
            for v in value:
                if visited >= 4096: break
                visit(v)
    visit(source.get('user', {}).get('facts', {}))
    visit(source.get('events', []))
    visit(source.get('rules', []))
    visit(source.get('goals', []))
    candidates = list(dict.fromkeys([*values, 1, 86400]))[:limit]
    baseline = Sandbox(compiled)
    predicates = [g for g in baseline.source.get('goals', []) if 'predicate' in g]
    def states(world):
        result = {}
        for goal in predicates:
            try:
                value = evaluate(goal['predicate'], world.context(private=True))
                if type(value) is bool: result[goal['id']] = value
            except ContractError: pass
        return result
    before = states(baseline)
    findings, changed_world = [], 0
    for seconds in candidates:
        world = Sandbox(compiled)
        world.wait(seconds)
        if not typed_equal(world.data, baseline.data) or world.user.snapshot() != baseline.user.snapshot() or world.version != baseline.version:
            changed_world += 1
            continue
        after = states(world)
        for identity, value in before.items():
            if value is False and after.get(identity) is True:
                findings.append({'goal': identity, 'wait_seconds': seconds, 'before': False,
                                 'after': True, 'same_world_and_user_state': True,
                                 'trace': world.trace, 'full_reward_tested': False})
    return {'scope':'clock_only_goal_sensitivity_not_business_verdict', 'candidate_seconds': candidates, 'scan_limit': 4096, 'variant_limit': limit,
            'changed_world_excluded': changed_world, 'findings': findings,
            'instruction':'Check reference-time obligations against original requirements. A legitimate wait task can have this behavior. Do not automatically reject or replace current time with initial time.'}

