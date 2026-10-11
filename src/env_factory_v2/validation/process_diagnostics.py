"""Cheap branch-local process ablations; hints never waive a requirement."""
import copy

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import refs, typed_equal
from env_factory_v2.validation.verification import authored_cases, run_scenario


def process_diagnostics(compiled, *, max_variants=16):
    require(type(max_variants) is int and 0 <= max_variants <= 16, 'process diagnostic budget must be 0..16')
    source = compiled['source']
    stages = {s['id']: s for s in source['user']['stages']}
    tools = {t['id']: t for t in source['tools']}
    rules = {r['id']: r['predicate'] for r in source['rules']}
    def dependencies(value, seen=()):
        found = set(refs(value))
        if isinstance(value, dict):
            if 'literal' in value:
                return found
            if set(value) == {'rule'}:
                identity = value['rule']['id']
                if identity not in seen:
                    found |= dependencies(rules[identity], (*seen, identity))
            if set(value) == {'transaction_preconditions'}:
                tool = tools[value['transaction_preconditions']['tool']]
                for identity in tool['rule_ids']:
                    found |= dependencies(rules[identity], (*seen, identity))
                for operation in tool['operations']:
                    found |= dependencies(operation.get('for_each', {'literal': []}), seen)
                    found |= dependencies(operation['key'], seen)
                    found |= dependencies(operation.get('when', {'literal': True}), seen)
                    found |= dependencies(operation['set'], seen)
            for item in value.values():
                found |= dependencies(item, seen)
        elif isinstance(value, list):
            for item in value:
                found |= dependencies(item, seen)
        return found
    def evidence(run):
        return run['outcome'].get('workflow', run['outcome'].get('deterministic_grade', {}))
    def otherwise_valid(run):
        workflow = evidence(run)
        components = run['outcome'].get('structured_components', workflow.get('components', {}))
        return (run['error'] is None and workflow.get('safety') is True
                and workflow.get('category_valid') is True
                and all(workflow.get('invariants', []))
                and all(g['passed'] for g in workflow.get('goals', []))
                and all(c['passed'] for c in components.values()))
    findings, tested, baselines = [], 0, 0
    for _, scenario, _ in authored_cases(compiled):
        if tested >= max_variants or baselines >= 8:
            break
        # No-write paths expose irrelevant write-input questions without trying
        # to remove consent, revocation, writes or business reads from a recipe.
        if not scenario['expect_success'] or scenario.get('expect_error') or any(
                step['kind'] in {'call', 'traverse'} and tools[step['tool']]['kind'] == 'transaction'
                for step in scenario['steps']):
            continue
        used_refs = dependencies(scenario['answer']) | dependencies(source['answers'])
        used_refs |= dependencies(source.get('explanations', []))
        candidates = []
        for step in scenario['steps']:
            if step['kind'] != 'user':
                continue
            stage = stages[step['stage']]
            if stage['kind'] != 'fact' or ('stage', stage['id']) in candidates:
                continue
            # Private answer recipes must not fake knowledge after its disclosure
            # is removed. A hint is still not proof of semantic independence.
            fact_ref = 'user.' + stage['fact']
            if any(p == 'user' or p == fact_ref or p.startswith(fact_ref + '.') for p in used_refs):
                continue
            candidates.append(('stage', stage['id']))
        # Remove only read calls; dependency errors suppress a finding. These
        # are counterexample candidates, not automatic permission to waive reads.
        candidates += [('call', step['id']) for step in scenario['steps']
                       if step['kind'] in {'call', 'traverse'} and tools[step['tool']]['kind'] == 'query']
        if not candidates:
            continue
        baselines += 1
        baseline = run_scenario(compiled, scenario, recipes_only=True)
        if not otherwise_valid(baseline) or not any(b['passed'] for b in evidence(baseline).get('process', [])):
            continue
        for kind, target in candidates:
            if tested >= max_variants:
                break
            variant = copy.deepcopy(scenario)
            variant['steps'] = [s for s in variant['steps'] if not (
                (kind == 'stage' and s['kind'] == 'user' and s['stage'] == target)
                or (kind == 'call' and s['kind'] in {'call', 'traverse'} and s['id'] == target))]
            tested += 1
            run = run_scenario(compiled, variant, recipes_only=True)
            if not otherwise_valid(run) or not typed_equal(run['answer'], baseline['answer']) \
                    or not typed_equal(run['final_state'], baseline['final_state']) \
                    or run['time']['elapsed_seconds'] != baseline['time']['elapsed_seconds']:
                continue
            before = {b['id']: b for b in evidence(baseline)['process']}
            after = {b['id']: b for b in evidence(run)['process']}
            if any(b['passed'] for b in after.values()):
                continue
            branches = [b['id'] for b in source['success'] if (kind == 'call' or target in b['required_stages'])
                        and before[b['id']]['passed'] and after[b['id']]['active']
                        and not after[b['id']]['passed']]
            if branches:
                findings.append({'scenario': scenario['id'],
                    **({'omitted_fact_stage': target} if kind == 'stage' else {'omitted_query_call': target}),
                    'rejected_branches': branches, 'steps': variant['steps'],
                    'same_answer_and_world': True, 'safety_and_goals_passed': True,
                    'semantic_communication_checked': False})
    return {'scope': 'author_process_ablation_hint_not_business_verdict',
            'baseline_limit': 8, 'baselines_tested': baselines, 'variant_limit': max_variants,
            'variants_tested': tested, 'findings': findings,
            'instruction': 'Review these no-write paths against original requirements: omitting a fact question or read call '
            'preserves recipe answers, final state, safety and goals but fails an explicit process gate. '
            'A question may still have a real business or communication purpose. Keep necessary questions; '
            'make genuinely write-only requirements branch-applicable using shared business rules. '
            'Do not remove authorization, change expectations, or waive independent review to silence hints.'}
