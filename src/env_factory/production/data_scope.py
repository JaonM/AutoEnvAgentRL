"""Caller-declared customer/record partitions for synthetic world sampling.

Materialize disjoint private tables before task authoring. Shared catalog tables
stay shared; ownership columns and identity fields are caller configuration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random


def partition(tables, ownership, *, seed, heldout_count):
    """Return owner groups plus table rows; no generated task can access the other group's rows."""
    if not isinstance(ownership, dict) or set(ownership) != {'owner_table', 'owner_key', 'tables'}:
        raise ValueError('ownership requires owner_table, owner_key, tables')
    owner_table, owner_key = ownership['owner_table'], ownership['owner_key']
    mappings = ownership['tables']
    if not isinstance(mappings, dict) or mappings.get(owner_table) != owner_key:
        raise ValueError('owner table must be included with its owner key')
    if owner_table not in tables:
        raise ValueError('owner table missing')
    owners = [row.get(owner_key) for row in tables[owner_table]]
    if any(not isinstance(owner, str) or not owner for owner in owners) or len(set(owners)) != len(owners):
        raise ValueError('owners must have unique nonempty string identifiers')
    if type(heldout_count) is not int or not 0 < heldout_count < len(owners):
        raise ValueError('heldout count must leave both partitions nonempty')
    known = set(owners)
    for table, field in mappings.items():
        if table not in tables or not isinstance(field, str):
            raise ValueError('invalid ownership table or column')
        if any(row.get(field) not in known for row in tables[table]):
            raise ValueError(f'{table} has a missing/unknown owner')
        ids = [row.get('id') for row in tables[table]]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{table} needs unique row identifiers')
    shuffled = sorted(owners)
    random.Random(seed).shuffle(shuffled)
    groups = {'train': sorted(shuffled[heldout_count:]), 'dev': sorted(shuffled[:heldout_count])}
    result = {}
    for name, members in groups.items():
        allowed = set(members)
        scoped = {table: [copy.deepcopy(row) for row in rows if table not in mappings or row[mappings[table]] in allowed]
                  for table, rows in tables.items()}
        result[name] = {'owners': members, 'tables': scoped,
                        'private_row_ids': {table: sorted(row['id'] for row in scoped[table]) for table in mappings}}
    for table in mappings:
        if set(result['train']['private_row_ids'][table]) & set(result['dev']['private_row_ids'][table]):
            raise ValueError('private rows overlap')
    return {'seed': seed, 'ownership': copy.deepcopy(ownership), 'partitions': result,
            'source_tables_sha256': hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest(),
            'shared_tables': sorted(set(tables) - set(mappings))}


def check_references(tables, rules):
    """Check caller-declared scalar/list foreign keys after scoping (including nested list columns).

    Rule: {table, field, target_table, target_field}. A list column is a list of scalar keys.
    Missing fields, dangling references or changed owners must never be silently repaired.
    """
    problems = []
    for rule in rules:
        if set(rule) != {'table', 'field', 'target_table', 'target_field'}:
            raise ValueError('invalid reference rule')
        target = {row[rule['target_field']] for row in tables[rule['target_table']]}
        for row in tables[rule['table']]:
            value = row.get(rule['field'])
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, (str, int)) or v not in target for v in values):
                problems.append(f"{rule['table']}/{row.get('id')}.{rule['field']}: dangling reference")
    return problems
