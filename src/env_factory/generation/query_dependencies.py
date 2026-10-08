"""Structural proof that captured select inputs actually affect its query."""

def validate_query_dependencies(source):
    implementations = {item['tool_name']:item for item in source['tool_implementations']}
    def dependent(value):
        if isinstance(value, dict):
            return '$ref' in value or any(dependent(child) for child in value.values())
        return isinstance(value,list) and any(dependent(child) for child in value)
    for scenario in source['scenarios']:
        if scenario.get('kind') != 'goal_success':
            continue
        for step in scenario['steps']:
            if step.get('operation') != 'tool_call':
                continue
            implementation = implementations.get(step['tool_name'], {})
            if implementation.get('operation') not in {'select','aggregate_count'}:
                continue
            consumed = {item['argument'] for item in implementation.get('filters', [])}
            if isinstance(implementation.get('pagination'), dict):
                consumed.add(implementation['pagination'].get('cursor_argument'))
            for argument,value in step.get('arguments', {}).items():
                if dependent(value) and argument not in consumed:
                    raise ValueError('QUERY_DEPENDENCY_UNUSED: '+step['tool_name']+'.'+argument+
                        ' references an earlier tool result but has no query filter; declare the intended column and operator')
