"""Conservative negative-test proof for normalized rewards and known penalties."""
import math
import json
from env_factory.sandbox_runtime import ContractRewardGate, ContractRewardAggregator


def validate_reward_goal_consistency(source):
    """Reject provably contradictory positive terminal predicates, not partial goals."""
    metrics = {metric['id']: metric for metric in source.get('metrics', [])}
    goals = source.get('semantic_goal', {}).get('row_predicates', [])
    conflicts = []
    for ri, rule in enumerate(source.get('metric_implementations', [])):
        metric = metrics.get(rule.get('metric_id'), {})
        mapping = rule.get('score_mapping', {})
        if (rule.get('operator') != 'state_predicates' or metric.get('weight', 0) <= 0
                or mapping.get('pass') != 1 or mapping.get('fail') != 0):
            continue
        for pi, predicate in enumerate(rule.get('expected', [])):
            for gi, goal in enumerate(goals):
                if (predicate.get('table') != goal.get('table')
                        or predicate.get('where') != goal.get('where')
                        or predicate.get('count', 0) <= 0 or goal.get('count', 0) <= 0):
                    continue
                for field in predicate.get('values', {}).keys() & goal.get('values', {}).keys():
                    if field in predicate.get('value_expressions', {}) or field in goal.get('value_expressions', {}):
                        continue
                    if predicate['values'][field] != goal['values'][field]:
                        conflicts.append({'metric_id': rule.get('metric_id'), 'field': field,
                            'reward_path': f'metric_implementations[{ri}].expected[{pi}].values.{field}',
                            'goal_path': f'semantic_goal.row_predicates[{gi}].values.{field}',
                            'reward_value': predicate['values'][field], 'goal_value': goal['values'][field]})
    if conflicts:
        raise ValueError('REWARD_GOAL_CONFLICT: ' + json.dumps(conflicts, ensure_ascii=False))


def reward_upper_bound(metrics, deterministic_scores, gate, context):
    # Nonnegative weights make unknown [0,1] scores monotone. A known [-1,0]
    # penalty contributes its actual fixed value, including during route weight
    # normalization. Unknown penalties and custom contracts still need judgment.
    if type(gate) is not ContractRewardGate or not metrics:
        return None
    ids = [metric.get('id') for metric in metrics]
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != len(ids):
        return None
    for metric in metrics:
        weight = metric.get('weight')
        penalty = metric.get('category') == 'penalty'
        if (metric.get('category') not in {'process', 'outcome', 'penalty'}
                or metric.get('score_range') != ([-1, 0] if penalty else [0, 1])
                or penalty and metric['id'] not in deterministic_scores
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            return None
    known = {identity: deterministic_scores[identity] for identity in ids if identity in deterministic_scores}
    ranges = {metric['id']: metric['score_range'] for metric in metrics}
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not ranges[identity][0] <= value <= ranges[identity][1]
           for identity, value in known.items()):
        return None
    unknown = [identity for identity in ids if identity not in known]
    optimistic = {**known, **dict.fromkeys(unknown, 1.0)}
    result = ContractRewardAggregator(metrics).aggregate(gate.apply(optimistic, context))
    return {'scope': 'reward_upper_bound_not_observed_reward',
            'reward_upper_bound': result['reward'], 'assumed_full_credit': unknown,
            'deterministic_components': known}
