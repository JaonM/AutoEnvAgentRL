"""Bounded deterministic fixture materialization, never model-written bulk rows."""
from __future__ import annotations

import copy

from .contract import fields, require
from .expressions import canonical, ContractError


def materialize(tables):
    previous = {}
    for table in tables:
        if 'generator' not in table:
            previous[table['id']] = table
            continue
        generator = table['generator']
        fields(generator, ['count', 'seed', 'columns'])
        require(type(generator['count']) is int and 0 <= generator['count'] <= 10000, 'fixture count limit')
        require(type(generator['seed']) is int and 0 <= generator['seed'] < 2**32, 'invalid fixture seed')
        require(isinstance(generator['columns'], dict), 'fixture columns required')
        require(set(table['schema']['required']) <= set(generator['columns']) <= set(table['schema']['properties']), 'fixture columns mismatch')
        for spec in generator['columns'].values():
            require(isinstance(spec, dict) and spec.get('kind') in {'constant', 'sequence', 'cycle', 'foreign_key'},
                    'unsupported fixture column generator')
        rows = []
        for index in range(generator['count']):
            row = {}
            for column, spec in generator['columns'].items():
                require(isinstance(spec, dict), 'column generator required')
                kind = spec.get('kind')
                if kind == 'constant':
                    fields(spec, ['kind', 'value'])
                    value = copy.deepcopy(spec['value'])
                elif kind == 'sequence':
                    fields(spec, ['kind', 'start', 'step'], ['prefix', 'width'])
                    require(type(spec['start']) in {int, float} and type(spec['step']) in {int, float}, 'numeric sequence required')
                    value = spec['start'] + index * spec['step']
                    if 'prefix' in spec:
                        require(isinstance(spec['prefix'], str) and type(value) is int, 'string sequence requires integer indices')
                        require(type(spec.get('width', 0)) is int and 0 <= spec.get('width', 0) <= 12, 'sequence width limit')
                        value = spec['prefix'] + str(value).zfill(spec.get('width', 0))
                elif kind == 'cycle':
                    fields(spec, ['kind', 'values'], ['offset'])
                    require(isinstance(spec['values'], list) and 1 <= len(spec['values']) <= 128, 'cycle values limit')
                    require(type(spec.get('offset', 0)) is int, 'integer cycle offset required')
                    value = copy.deepcopy(spec['values'][(index + generator['seed'] + spec.get('offset', 0)) % len(spec['values'])])
                elif kind == 'foreign_key':
                    fields(spec, ['kind', 'table', 'column'], ['offset'])
                    require(spec['table'] in previous and previous[spec['table']]['rows'], 'fixture FK must use an earlier nonempty table')
                    require(type(spec.get('offset', 0)) is int, 'integer FK offset required')
                    target = previous[spec['table']]['rows'][(index + generator['seed'] + spec.get('offset', 0)) % len(previous[spec['table']]['rows'])]
                    require(spec['column'] in target, 'unknown fixture FK column')
                    value = copy.deepcopy(target[spec['column']])
                else:
                    raise ContractError('unsupported fixture column generator')
                row[column] = value
            rows.append(row)
        if 'rows' in table:
            require(canonical(table['rows']) == canonical(rows), 'materialized fixture differs from its declared generator')
        table['rows'] = rows
        previous[table['id']] = table


def review_contract(source):
    """Lossless generator plans replace derivable bulk rows, with concrete previews."""
    source = copy.deepcopy(source)
    evidence = {}
    for table in source['tables']:
        if 'generator' in table:
            rows = table.pop('rows')
            evidence[table['id']] = {'row_count': len(rows), 'first_rows': rows[:4], 'last_rows': rows[-4:]}
    return source, evidence
