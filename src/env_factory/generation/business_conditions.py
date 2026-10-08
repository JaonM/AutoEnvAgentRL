"""Compile explicit business eligibility requirements into shared query predicates.

This compiler does not infer requirements from prose. Independent semantic review
must still check completeness and meaning against the original public goal.
"""
from __future__ import annotations

import json
from .report_values import report_value


def _report_value(value):
    """Serialize numeric evidence without changing runtime comparison operands."""
    try:
        return report_value(value)
    except ValueError as exc:
        raise ValueError('BUSINESS_CONDITION_INVALID: non-finite computed operand') from exc


def compile_business_conditions(source: dict) -> list[dict]:
    conditions = source.get('business_conditions', [])
    if not isinstance(conditions, list):
        raise ValueError('BUSINESS_CONDITION_INVALID: expected a list')
    tools = {tool['function']['name']: tool['function'] for tool in source['tools']}
    implementations = {item['tool_name']: item for item in source['tool_implementations']}
    tables = {table['table_name']: table for table in source['tables']}
    public = json.dumps(source['description'].get('public_input', {}), ensure_ascii=False)
    seen, evidence = set(), []
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {
                'id', 'public_requirement', 'tool', 'argument', 'column', 'operator'}:
            raise ValueError('BUSINESS_CONDITION_INVALID: requires id/public_requirement/tool/argument/column/operator')
        if any(not isinstance(value, str) or not value.strip() for value in condition.values()):
            raise ValueError('BUSINESS_CONDITION_INVALID: fields must be nonempty strings')
        identity = condition['id']
        if identity in seen:
            raise ValueError('BUSINESS_CONDITION_INVALID: duplicate id ' + identity)
        seen.add(identity)
        if condition['public_requirement'] not in public:
            raise ValueError('BUSINESS_CONDITION_INVALID: quote an actual public requirement: ' + identity)
        name, argument, column = condition['tool'], condition['argument'], condition['column']
        impl, tool = implementations.get(name), tools.get(name)
        if not impl or not tool or impl.get('operation') != 'select':
            raise ValueError('BUSINESS_CONDITION_INVALID: eligibility currently requires a select tool: ' + identity)
        schema = tool['parameters']
        if argument not in schema.get('properties', {}) or argument not in schema.get('required', []):
            raise ValueError('BUSINESS_CONDITION_INVALID: condition argument must be declared and required: ' + identity)
        table = tables.get(impl.get('table'), {})
        if column not in {item['name'] for item in table.get('columns', [])}:
            raise ValueError('BUSINESS_CONDITION_INVALID: unknown business column: ' + identity)
        if condition['operator'] not in {'eq', 'gte', 'lte', 'in', 'contains'}:
            raise ValueError('BUSINESS_CONDITION_INVALID: unsupported operator: ' + identity)
        predicate = {key: condition[key] for key in ('argument', 'column', 'operator')}
        filters = impl.setdefault('filters', [])
        related = [item for item in filters if item.get('argument') == argument and item.get('column') == column]
        if related and related != [predicate]:
            raise ValueError('BUSINESS_CONDITION_INVALID: conflicting handwritten filter: ' + identity)
        if not related:
            filters.append(predicate)
        evidence.append({**condition, 'table': impl['table']})
    return evidence


def _condition_bindings(condition, bindings, reference_details):
    """Keep different reference queries separate; never choose the easiest one."""
    name = condition['tool']
    if name in bindings:
        return [(bindings[name], None)]
    grouped = {}
    for reference in reference_details or []:
        if name not in reference['called_tools']:
            continue
        binding = reference['bindings'].get(name)
        if binding is None or not reference.get('scenario_id'):
            raise ValueError('BUSINESS_CONDITION_UNBOUND: ambiguous reference binding for ' + condition['id'])
        key = json.dumps(binding, sort_keys=True, allow_nan=False)
        entry = grouped.setdefault(key, (binding, []))
        entry[1].append(reference['scenario_id'])
    if not grouped:
        raise ValueError('BUSINESS_CONDITION_UNBOUND: no executable reference binding for ' + condition['id'])
    return list(grouped.values())


def validate_condition_bindings(conditions: list[dict], bindings: dict, source: dict,
                                *, reference_details: list | None = None) -> None:
    """No declared condition may disappear because the reference omitted its call/argument."""
    for condition in conditions:
        def covered(value, expected):
            if isinstance(value, list):
                return any(covered(child, expected) for child in value)
            if not isinstance(value, dict) or set(value) == {'literal'}:
                return False
            result = value.get('tool_result', {})
            if result.get('name') == condition['tool']:
                import re
                implementation = next((item for item in source['tool_implementations']
                    if item['tool_name'] == condition['tool']), {})
                path = result.get('path', '')
                match = re.fullmatch(r'\$\.' + re.escape(implementation.get('result_field', 'records'))
                                    + r'\[\d+\]\.([^.[\]]+)', path)
                projected = set(implementation.get('projection', [])) | set(implementation.get('projection_aliases', {}))
                if implementation.get('operation') == 'select' and match and (not projected or match.group(1) in projected):
                    return True
            query = value.get('lookup', {})
            predicate = query.get('where', {}).get(condition['column'], {})
            if (query.get('table') == condition['table'] and isinstance(predicate, dict)
                    and condition['operator'] in predicate
                    and predicate[condition['operator']] == expected):
                return True
            return any(covered(child, expected) for child in value.values())
        for binding, _ in _condition_bindings(condition, bindings, reference_details):
            where = binding.get('where', {})
            if condition['operator'] not in where.get(condition['column'], {}):
                raise ValueError('BUSINESS_CONDITION_UNBOUND: no executable reference binding for ' + condition['id'])
            expected = where[condition['column']][condition['operator']]
            if not covered(source['metric_implementations'], expected):
                raise ValueError('BUSINESS_CONDITION_UNSCORED: bind an outcome through from_tool for ' + condition['id'])


def probe_business_conditions(conditions: list[dict], bindings: dict, source: dict,
                              *, reference_details: list | None = None) -> dict:
    """Generate isolated predicate probes using the actual runtime comparator.

    These are query-level counterfactuals, not full episode/reward certification.
    Keep unsupported or coupled predicates visible instead of declaring them passed.
    """
    import copy
    import datetime
    import math
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator
    from env_factory.sandbox_runtime import DeclarativeToolCompiler as Tools

    if reference_details is not None:
        cases = []
        for condition in conditions:
            for binding, references in _condition_bindings(condition, bindings, reference_details):
                report = probe_business_conditions([condition], {condition['tool']: binding}, source)
                for case in report['cases']:
                    if references is not None:
                        case['reference_ids'] = references
                    cases.append(case)
        return {'scope': 'isolated_query_predicates_not_episode_reward', 'cases': cases,
                'verified': sum(case['status'] == 'verified' for case in cases),
                'needs_authored_case': sum(case['status'] != 'verified' for case in cases)}

    state = {table['table_name']: copy.deepcopy(table['rows']) for table in source['tables']}
    cases = []
    for condition in conditions:
        table, column, operator = condition['table'], condition['column'], condition['operator']
        where = Evaluator._numeric_where(bindings[condition['tool']]['where'], state, depth=0)
        expected = where.get(column, {}).get(operator)
        rows = state[table]
        selected = [(index, row) for index, row in enumerate(rows) if Evaluator._where_matches(row, where)]
        # A list query can return several eligible alternatives. Its first row
        # need not be the row selected by the business outcome. Mutate a row
        # actually addressed by the condition-bearing reward lookup first.
        reward_rows = set()
        def find_reward_rows(value):
            if isinstance(value, list):
                for child in value:
                    find_reward_rows(child)
            elif isinstance(value, dict) and set(value) != {'literal'}:
                query = value.get('lookup')
                if isinstance(query, dict) and query.get('table') == table:
                    reward_where = Evaluator._numeric_where(query.get('where', {}), state, depth=0)
                    predicate = reward_where.get(column, {})
                    if isinstance(predicate, dict) and predicate.get(operator) == expected and operator in predicate:
                        reward_rows.update(index for index, row in selected
                            if Evaluator._where_matches(row, reward_where))
                for child in value.values():
                    find_reward_rows(child)
        find_reward_rows(source.get('metric_implementations', []))
        find_reward_rows(source.get('semantic_goal', {}))
        selected.sort(key=lambda item: item[0] not in reward_rows)
        case = {'condition_id': condition['id'], 'tool': condition['tool'], 'table': table,
                'column': column, 'operator': operator, 'resolved_operand': _report_value(expected),
                'status': 'needs_authored_case', 'reason': 'no isolated valid mutation found'}
        remainder = copy.deepcopy(where)
        remainder[column].pop(operator)
        if not remainder[column]: remainder.pop(column)
        mutations = []
        for index, row in selected:
            original = row[column]
            candidates = [item.get(column) for item in rows]
            if isinstance(original, bool):
                candidates.append(not original)
            elif isinstance(original, (int, float)):
                boundary = expected if isinstance(expected, (int, float)) and not isinstance(expected, bool) else original
                candidates.extend([boundary - 1, boundary + 1])
            elif isinstance(original, str):
                candidates.extend(['', '__counterfactual_value__'])
                try:
                    boundary = datetime.date.fromisoformat(expected)
                    candidates.extend([(boundary + datetime.timedelta(days=offset)).isoformat() for offset in (-1, 1)])
                except (ValueError, TypeError):
                    pass
            elif isinstance(original, list):
                candidates.append([])
            for value in candidates:
                if type(value) is not type(original) or value == original:
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                changed = {**row, column: value}
                try:
                    violates = not Tools._matches(value, expected, operator)
                except TypeError:
                    continue
                if not violates or not Evaluator._where_matches(changed, remainder):
                    continue
                if Evaluator._where_matches(changed, where):
                    raise ValueError('BUSINESS_CONDITION_COUNTERFACTUAL_FAILED: ' + condition['id'])
                mutations.append({'row_index': index, 'original_value': original, 'mutated_value': value})
                break
        if selected and len(mutations) == len(selected):
            case.update(status='verified', reason='all eligible alternatives violate only this query predicate',
                        **mutations[0], row_mutations=mutations, query_where=copy.deepcopy(bindings[condition['tool']]['where']),
                        baseline_matches=True, mutant_matches=False, other_predicates_match=True)
        if not selected:
            case['reason'] = 'reference query has no matching business row'
        cases.append(case)
    return {'scope': 'isolated_query_predicates_not_episode_reward', 'cases': cases,
            'verified': sum(case['status'] == 'verified' for case in cases),
            'needs_authored_case': sum(case['status'] != 'verified' for case in cases)}
