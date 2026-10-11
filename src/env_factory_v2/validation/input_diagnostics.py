"""Bounded author hints for input-sensitive expressions; never qualification gates."""
import copy

from env_factory_v2.contracts.contract import require, validate_user_overrides, validate_value
from env_factory_v2.contracts.expressions import ContractError, canonical, evaluate
from env_factory_v2.execution.runtime import Sandbox


def input_diagnostics(compiled, *, max_variants=16):
    require(type(max_variants) is int and 0 <= max_variants <= 16, 'diagnostic budget must be 0..16')
    source = compiled['source']
    bindings = {}
    for tool in source['tools']:
        if tool['kind'] == 'query':
            for argument, fact in tool.get('user_bindings', {}).items():
                bindings.setdefault(fact, []).append(tool['parameters']['properties'][argument])
    expressions = [(f'/answers/{i}/expected', answer['expected'], None)
                   for i, answer in enumerate(source['answers'])]
    for i, goal in enumerate(source.get('goals', [])):
        for key in ('transaction_plan', 'transaction_timeline'):
            for j, step in enumerate(goal.get(key, [])):
                expressions.extend((f'/goals/{i}/{key}/{j}/arguments/{name}', expression, step.get('when'))
                                   for name, expression in step['arguments'].items())
    def has_index(value):
        if isinstance(value, dict):
            if 'literal' in value:
                return False
            return 'at' in value or any(has_index(v) for v in value.values())
        return isinstance(value, list) and any(has_index(v) for v in value)
    expressions = [item for item in expressions if has_index(item[1])]
    candidates = len(expressions)
    expressions = expressions[:64] if bindings and max_variants else []
    base = Sandbox(compiled)
    base.reset()
    context = base.context(private=True)
    working = []
    for path, expression, guard in expressions:
        try:
            if guard is None or evaluate(guard, context):
                evaluate(expression, context)
                working.append((path, expression, guard))
        except ContractError:
            # Runtime observations may legitimately be unavailable before execution.
            continue
    marker = '__author_input_probe__'
    while marker in canonical(source):
        marker += '_'
    findings, tested = [], 0
    for fact, schemas in sorted(bindings.items()):
        if tested >= max_variants:
            break
        original = source['user']['facts'].get(fact)
        if isinstance(original, str):
            value = marker
        elif isinstance(original, list) and original and all(isinstance(v, str) for v in original):
            value = [marker + str(i) for i in range(len(original))]
        else:
            continue
        try:
            validate_user_overrides(source, {fact: value})
            for schema in schemas:
                validate_value(value, schema, 'diagnostic query input')
        except ContractError:
            continue
        tested += 1
        varied = copy.deepcopy(context)
        varied['user'][fact] = value
        for path, expression, guard in working:
            try:
                if guard is None or evaluate(guard, varied):
                    evaluate(expression, varied)
            except ContractError as exc:
                findings.append({'path': path, 'user_overrides': {fact: value}, 'error': str(exc)})
    return {'scope': 'author_input_sensitivity_hint_not_business_verdict', 'variants_tested': tested,
            'variant_limit': max_variants, 'expression_limit': 64, 'candidate_expressions': candidates,
            'expressions_tested': len(working), 'findings': findings,
            'instruction': 'Check these input-sensitive expressions against original input guarantees. '
            'If empty lookup is reachable, provide truthful no-result completion and guard dependent reads. '
            'Do not invent records, narrow user inputs or change public obligations to hide a diagnostic. '
            'These probes do not establish business validity or a successful trajectory.'}
