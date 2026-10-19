"""Seeded scale budgets, graph integrity and bounded author views for any world.

Business distributions belong to caller-owned recipes. These helpers know no
industry, public benchmark records, policy outcomes or tool names.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import random


def values(obj, path):
    if not path:
        return [obj]
    head, *tail = path.split('.')
    rest = '.'.join(tail)
    if head == '*':
        children = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
        return [v for child in children for v in values(child, rest)]
    return values(obj[head], rest) if isinstance(obj, dict) and head in obj else []


def allocate(total, count, *, minimum=0, maximum, rng, weights=None):
    """Exact total with variable per-entity cardinality, deterministic for the seed."""
    if type(total) is not int or type(count) is not int or count < 0 or not minimum * count <= total <= maximum * count:
        raise ValueError('infeasible cardinality budget')
    candidates = list(range(minimum, maximum + 1))
    result = rng.choices(candidates, weights=weights, k=count)
    delta = total - sum(result)
    while delta:
        direction = 1 if delta > 0 else -1
        eligible = [i for i, n in enumerate(result) if minimum <= n + direction <= maximum]
        rng.shuffle(eligible)
        for i in eligible[:abs(delta)]:
            result[i] += direction
            delta -= direction
    return result


def remap(obj, mapping):
    """Rewrite exact identity values AND keyed maps, preserving all other content."""
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, list):
        return [remap(v, mapping) for v in obj]
    if isinstance(obj, dict):
        result = {mapping.get(k, k): remap(v, mapping) for k, v in obj.items()}
        if len(result) != len(obj):
            raise ValueError('identity remapping collapsed keys')
        return result
    return copy.deepcopy(obj)


def integrity(tables, contract):
    """Validate declared nested IDs, foreign keys, inverse ownership and scale."""
    errors = []
    for table, count in contract.get('counts', {}).items():
        if len(tables.get(table, [])) != count:
            errors.append(f'{table}: expected {count} rows')
    for table, paths in contract.get('unique', {}).items():
        for path in paths:
            vals = [v for row in tables[table] for v in values(row, path)]
            if any(not isinstance(v, (str, int)) or v == '' for v in vals) or len(vals) != len(set(vals)):
                errors.append(f'{table}.{path}: invalid/duplicate identity')
    for rule in contract.get('references', []):
        target = {v for row in tables[rule['target_table']] for v in values(row, rule['target_path'])}
        for row in tables[rule['table']]:
            vals = values(row, rule['path'])
            if any(v not in target for v in vals):
                errors.append(f"{rule['table']}/{row.get('id')}.{rule['path']}: dangling reference")
    for rule in contract.get('inverse', []):
        owners = {r[rule['owner_key']]: r for r in tables[rule['owner_table']]}
        children = {k: [] for k in owners}
        for row in tables[rule['child_table']]:
            owner = row[rule['child_owner']]
            if owner not in children:
                errors.append('unknown child owner')
            else:
                children[owner].append(row[rule['child_key']])
        for owner, row in owners.items():
            actual = row[rule['owner_list']]
            if len(actual) != len(set(actual)) or set(actual) != set(children[owner]):
                errors.append(f'{owner}: inverse ownership mismatch')
    return errors


def profile(tables, metrics):
    result = {'tables': {name: len(rows) for name, rows in tables.items()}, 'metrics': {}}
    for metric in metrics:
        vals = [v for row in tables[metric['table']] for v in values(row, metric['path'])]
        if metric.get('measure') == 'length':
            vals = [len(v) for v in vals]
        result['metrics'][metric['name']] = {'observations': len(vals), 'distinct': len(set(map(str, vals))),
                                           'histogram': dict(sorted(Counter(map(str, vals)).items()))}
    return result


def author_context(request, index):
    """Expose a deterministic owner shard and shared-record closure to task authors.

    Entire rows are retained, including nested inventory. Runtime composition and
    grading still use the full world. A small author view never removes distractors
    from the actual sandbox or silently truncates a record.
    """
    shown = {k: copy.deepcopy(v) for k, v in request.items() if k not in (
        'user_facts', 'user_scene', 'user_request', 'user_script', 'agent_policy', 'source',
        'behavior_contract', 'behavior_contract_sha256', 'sop_contract', 'sop_contract_sha256', 'sop_legacy', 'sop_method')}
    config = (request.get('source') or {}).get('authoring_scope')
    if not config:
        return shown
    ownership = config['ownership']
    owner_table, key = ownership['owner_table'], ownership['owner_key']
    owners = sorted(row[key] for row in request['tables'][owner_table])
    rng = random.Random(f"{config.get('seed', 0)}:{index}")
    rng.shuffle(owners)
    for budget in range(min(config['owners_per_task'], len(owners)), 0, -1):
        allowed = set(owners[:budget])
        selected = {}
        for table, rows in request['tables'].items():
            field = ownership['tables'].get(table)
            if field:
                selected[table] = [copy.deepcopy(r) for r in rows if r[field] in allowed]
            else:
                shuffled = list(rows)
                random.Random(f'{index}:{table}').shuffle(shuffled)
                selected[table] = copy.deepcopy(shuffled[:config.get('catalog_rows', 12)])
        # Foreign-key closure only expands shared tables. Pulling another private
        # owner into the view would violate its ownership boundary.
        changed = True
        while changed:
            changed = False
            for rule in config.get('references', []):
                target_table = rule['target_table']
                if target_table in ownership['tables']:
                    continue
                wanted = {v for row in selected[rule['table']] for v in values(row, rule['path'])}
                ids = {r['id'] for r in selected[target_table]}
                for row in request['tables'][target_table]:
                    if row['id'] not in ids and wanted.intersection(values(row, rule['target_path'])):
                        selected[target_table].append(copy.deepcopy(row))
                        ids.add(row['id'])
                        changed = True
        shown['tables'] = selected
        if len(json.dumps(shown, ensure_ascii=False).encode()) <= config.get('max_bytes', 600_000):
            shown['author_view'] = {'owners': sorted(allowed), 'full_world_row_counts':
                                    {t: len(r) for t, r in request['tables'].items()},
                                    'selection_seed': index, 'runtime_uses_full_world': True}
            return shown
    raise ValueError('one complete owner graph exceeds the author context budget')
