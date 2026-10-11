"""World variants shared by the author/oracle self-checks and the evaluator."""
from __future__ import annotations

import copy
import random

from v3_runtime import canonical


def approval_flips(facts):
    """Each boolean approve_* fact flipped on its own."""
    variants = []
    for key, value in facts.items():
        if key.startswith('approve') and isinstance(value, bool):
            changed = copy.deepcopy(facts)
            changed[key] = not value
            variants.append((f'flip:{key}', changed))
    return variants


def _mutable_cells(data):
    cells = []
    for table, rows in data.items():
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for column, value in row.items():
                if column == 'id' or column.endswith('_id') or column.endswith('_ids'):
                    continue
                if isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)):
                    cells.append((table, index, column))
    return cells


def perturb(data, seed):
    """Small generic edits to integer/boolean business fields; identities untouched."""
    rng = random.Random(seed)
    world = copy.deepcopy(data)
    cells = _mutable_cells(world)
    if not cells:
        return world
    for table, index, column in rng.sample(cells, k=min(len(cells), rng.randint(1, 3))):
        value = world[table][index][column]
        if isinstance(value, bool):
            world[table][index][column] = not value
        else:
            peers = sorted({row[column] for row in world[table]
                            if isinstance(row, dict) and isinstance(row.get(column), int)
                            and not isinstance(row.get(column), bool)})
            options = sorted({max(0, value + d) for d in (-3, -2, -1, 1, 2, 3)} | set(peers) | {0})
            options = [v for v in options if v != value] or [value + 1]
            world[table][index][column] = rng.choice(options)
    return world


def variants(data, facts, perturbed=10):
    worlds = [('original', data, facts)]
    worlds += [(name, data, changed) for name, changed in approval_flips(facts)]
    worlds += [(f'perturb:{seed}', perturb(data, seed), facts) for seed in range(1, perturbed + 1)]
    return worlds


def state_diff(expected_state, actual_state, tables):
    """Tables whose rows differ (order-insensitive, exact values)."""
    differences = {}
    for table in tables:
        want = sorted((canonical(row) for row in expected_state.get(table, [])))
        got = sorted((canonical(row) for row in actual_state.get(table, [])))
        if want != got:
            differences[table] = {'missing': sorted(set(want) - set(got))[:5],
                                  'unexpected': sorted(set(got) - set(want))[:5]}
    return differences


def flatten_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_values(item)
    else:
        yield value


def answer_recall(expected_answer, actual_answer):
    """Share of the truth's non-boolean leaf values that appear in the actual answer."""
    want = [v for v in flatten_values(expected_answer) if v is not None and not isinstance(v, bool)]
    got = [canonical(v) for v in flatten_values(actual_answer)]
    if not want:
        return None
    pool = list(got)
    hits = 0
    for value in want:
        key = canonical(value)
        alternatives = [key]
        if isinstance(value, (int, float)):
            alternatives += [canonical(float(value)), canonical(int(value))] if float(value).is_integer() else []
        match = next((a for a in alternatives if a in pool), None)
        if match is not None:
            pool.remove(match)
            hits += 1
    return hits / len(want)
