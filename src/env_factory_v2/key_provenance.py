"""Static origins for relational keys; never infer joins from matching values."""
def row_table(expression, tables):
    op, value = next(iter(expression.items()))
    if op == 'ref':
        parts = value.split('.')
        return parts[1] if len(parts) == 2 and parts[0] in {'data', 'initial'} and parts[1] in tables else None
    if op in {'filter', 'sort'}:
        return row_table(value['items'], tables)
    return None


def has_rule_selection(expression):
    """Only filtering can establish selection; a sort key cannot do so."""
    op, value = next(iter(expression.items()))
    if op == 'sort':
        return has_rule_selection(value['items'])
    if op == 'filter':
        return predicate_has_rule(value['value']) or has_rule_selection(value['items'])
    return False


def predicate_has_rule(expression):
    op, value = next(iter(expression.items()))
    if op == 'rule':
        return True
    if op == 'let':
        return predicate_has_rule(value['value'])
    if op in {'and', 'or', 'not'}:
        return any(predicate_has_rule(child) for child in value)
    return False


def key_origin(expression, table, bindings, tables):
    """Return (table, column, collection, selected_by_public_rule), or unknown."""
    op, value = next(iter(expression.items()))
    if op == 'ref':
        parts = value.split('.')
        if len(parts) == 2 and parts[0] == 'vars':
            return bindings.get(parts[1])
        if len(parts) == 2 and parts[0] == 'row' and table in tables:
            if parts[1] in tables[table]['schema']['properties']:
                return (table, parts[1], False, False)
    if op == 'map':
        origin = key_origin(value['value'], row_table(value['items'], tables), bindings, tables)
        if origin and not origin[2]:
            # The selected set must be derived from a rule-filtered table;
            # nested predicate validation independently rejects invented clauses.
            return (*origin[:2], True, has_rule_selection(value['items']))
    return None


def linked(left, right, tables):
    if not left or not right:
        return False
    a, column_a, *_ = left
    b, column_b, *_ = right
    def forward(source, column, target, target_column):
        return tables[target]['primary_key'] == target_column and any(
            fk['column'] == column and fk['target_table'] == target
            and fk['target_column'] == target_column for fk in tables[source].get('foreign_keys', []))
    return forward(a, column_a, b, column_b) or forward(b, column_b, a, column_a)


def relational_predicate(expression, table, bindings, tables):
    op, value = next(iter(expression.items()))
    if op not in {'eq', 'in'}:
        return False
    left, right = (key_origin(item, table, bindings, tables) for item in value)
    if not linked(left, right, tables):
        return False
    if op == 'eq':
        return not left[2] and not right[2]
    return not left[2] and right[2] and right[3]
