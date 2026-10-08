"""Authoring diagnostics for expressions in literal business-goal selectors.

Kept separate from the evaluator: this validator never rewrites a reward.
"""


def validate_predicate_selectors(source):
    predicates = [
        (f'semantic_goal.row_predicates[{index}]', predicate)
        for index, predicate in enumerate(source.get('semantic_goal', {}).get('row_predicates', []))
    ]
    for index, metric in enumerate(source.get('metric_implementations', [])):
        if metric.get('operator') == 'state_predicates':
            predicates.extend(
                (f'metric_implementations[{index}].expected[{row}]', predicate)
                for row, predicate in enumerate(metric.get('expected', []))
            )
    failures = []
    for path, predicate in predicates:
        if not isinstance(predicate, dict) or not isinstance(predicate.get('where'), dict):
            continue  # The structural schema reports malformed predicates.
        for field, value in predicate['where'].items():
            if not isinstance(value, dict):
                continue
            expression = (set(value) in ({'from_tool'}, {'lookup'}, {'at_tool'}, {'initial'}, {'$ref'}, {'$expr'})
                          or 'op' in value and 'args' in value)
            if expression:
                failures.append(f'{path}.where.{field}')
    if failures:
        raise ValueError(
            'STATE_SELECTOR_EXPRESSION: ' + ', '.join(failures)
            + ' compares literal values; expressions are not evaluated in where. '
              'Use discoverable literal business selectors and put dynamic field targets '
              'in value_expressions (including the selected ID when needed). '
              'Keep the business condition and reward; do not replace a dynamic target with a fixture ID.'
        )
