"""Bounded, reproducible synthetic rows without generated executable code."""
from __future__ import annotations

import copy
import hashlib
import json
import math


def expand_data_recipes(source, *, seed=0):
    if '_data_recipe_provenance' in source:
        raise ValueError('DATA_RECIPE_INVALID: provenance is compiler-owned')
    recipes = [table for table in source.get('tables', []) if 'row_recipe' in table]
    if not recipes:
        return source
    if type(seed) is not int:
        raise ValueError('DATA_RECIPE_INVALID: runner seed must be an integer')
    total, cells = 0, 0
    generators = {}

    def scalar(value):
        return (value is None or type(value) in {bool, int}
                or isinstance(value, str) and len(value) <= 1024
                or type(value) is float and math.isfinite(value))

    def generator(definition):
        if not isinstance(definition, dict) or len(definition) != 1:
            raise ValueError('DATA_RECIPE_INVALID: each column requires one generator')
        kind, body = next(iter(definition.items()))
        if kind == 'literal' and scalar(body):
            return lambda index, entropy: body
        if kind in {'cycle', 'seeded_choice'} and isinstance(body, list) and 1 <= len(body) <= 100 and all(map(scalar, body)):
            return lambda index, entropy: body[(index if kind == 'cycle' else entropy) % len(body)]
        if (kind == 'integer' and isinstance(body, dict) and set(body) == {'min', 'max'}
                and type(body['min']) is int and type(body['max']) is int
                and -10**12 <= body['min'] <= body['max'] <= 10**12):
            return lambda index, entropy: body['min'] + entropy % (body['max'] - body['min'] + 1)
        keys = {'start', 'step'} | ({'prefix', 'suffix'} if kind == 'string_sequence' else set())
        if (kind in {'sequence', 'string_sequence'} and isinstance(body, dict) and set(body) == keys
                and type(body['start']) is int and type(body['step']) is int
                and abs(body['start']) <= 10**12 and 0 < abs(body['step']) <= 10**9):
            if kind == 'sequence':
                return lambda index, entropy: body['start'] + index * body['step']
            if all(isinstance(body[key], str) and len(body[key]) <= 128 for key in ('prefix', 'suffix')):
                return lambda index, entropy: body['prefix'] + str(body['start'] + index * body['step']) + body['suffix']
        raise ValueError('DATA_RECIPE_INVALID: invalid bounded scalar generator: ' + str(kind))

    # Check every declaration and the total cap before allocating any rows.
    for table in source['tables']:
        rows = table.get('rows', [])
        if not isinstance(rows, list):
            raise ValueError('DATA_RECIPE_INVALID: explicit rows must be a list')
        total += len(rows)
        if 'row_recipe' not in table:
            continue
        recipe = table['row_recipe']
        if (not isinstance(recipe, dict) or set(recipe) != {'count', 'columns'}
                or type(recipe['count']) is not int or not 1 <= recipe['count'] <= 100000
                or not isinstance(recipe['columns'], dict)
                or not 1 <= len(recipe['columns']) <= 32
                or set(recipe['columns']) != {column['name'] for column in table['columns']}):
            raise ValueError('DATA_RECIPE_INVALID: recipe requires count 1..100000 and exactly the schema columns')
        total += recipe['count']
        cells += recipe['count'] * len(recipe['columns'])
        if total > 100000 or cells > 2000000:
            raise ValueError('DATA_RECIPE_INVALID: expanded dataset exceeds 100000 rows or 2000000 generated cells')
        name = table['table_name']
        if name in generators:
            raise ValueError('DATA_RECIPE_INVALID: duplicate table name')
        generators[name] = {key: generator(value) for key, value in recipe['columns'].items()}
    if total > 100000:
        raise ValueError('DATA_RECIPE_INVALID: total expanded dataset exceeds 100000 rows')
    result = copy.deepcopy(source)
    provenance = {}
    serialized_bytes = 0
    for table in result['tables']:
        if 'row_recipe' not in table:
            continue
        recipe = table.pop('row_recipe')
        name = table['table_name']
        def cell(key, index):
            payload = json.dumps([seed, name, key, index], ensure_ascii=False, separators=(',', ':'))
            entropy = int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')
            return generators[name][key](index, entropy)
        generated = []
        for index in range(recipe['count']):
            row = {key: cell(key, index) for key in recipe['columns']}
            serialized_bytes += len(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
            if serialized_bytes > 64 * 1024 * 1024:
                raise ValueError('DATA_RECIPE_INVALID: generated rows exceed 64 MiB')
            generated.append(row)
        table['rows'] = generated + table.get('rows', [])
        identity = hashlib.sha256(json.dumps(table['rows'], ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        provenance[name] = {'algorithm': 'sha256-index-v1', 'seed': seed, 'recipe': recipe,
                            'expanded_row_count': len(table['rows']), 'expanded_rows_sha256': identity}
    result['_data_recipe_provenance'] = provenance
    return result
