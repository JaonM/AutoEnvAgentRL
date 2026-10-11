"""World variants: original data, flipped approvals and generic perturbations."""
from __future__ import annotations

import copy
import random

from env_factory_v3.runtime import canonical


def approval_flips(facts):
    """Each boolean approve* fact flipped on its own."""
    variants = []
    for key, value in sorted(facts.items()):
        if key.startswith('approve') and isinstance(value, bool):
            changed = copy.deepcopy(facts)
            changed[key] = not value
            variants.append((f'flip:{key}', changed))
    return variants


def _mutable_cells(data):
    cells = []
    for table in sorted(data):
        for index, row in enumerate(data[table]):
            if not isinstance(row, dict):
                continue
            for column in sorted(row):
                value = row[column]
                if column == 'id' or column.endswith('_id') or column.endswith('_ids'):
                    continue
                if isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)):
                    cells.append((table, index, column))
    return cells


def perturb(data, seed):
    """Small edits to integer/boolean business fields. Identities and relations stay intact."""
    rng = random.Random(seed)
    world = copy.deepcopy(data)
    cells = _mutable_cells(world)
    if not cells:
        return world
    for table, index, column in rng.sample(cells, k=min(len(cells), rng.randint(1, 3))):
        value = world[table][index][column]
        if isinstance(value, bool):
            world[table][index][column] = not value
            continue
        peers = sorted({row[column] for row in world[table] if isinstance(row, dict)
                        and isinstance(row.get(column), int) and not isinstance(row.get(column), bool)})
        options = sorted({max(0, value + d) for d in (-3, -2, -1, 1, 2, 3)} | set(peers) | {0})
        options = [v for v in options if v != value] or [value + 1]
        world[table][index][column] = rng.choice(options)
    return world


def variants(data, facts, *, perturbed=20):
    """[(name, data, facts)] in a fixed order: original, approval flips, perturbed worlds."""
    worlds = [('original', copy.deepcopy(data), copy.deepcopy(facts))]
    worlds += [(name, copy.deepcopy(data), changed) for name, changed in approval_flips(facts)]
    worlds += [(f'perturb:{seed}', perturb(data, seed), copy.deepcopy(facts)) for seed in range(1, perturbed + 1)]
    return worlds


def state_diff(expected_state, actual_state, tables=None):
    """Tables whose rows differ, order-insensitive and exact. Empty dict means equal."""
    tables = sorted(set(expected_state) | set(actual_state)) if tables is None else tables
    differences = {}
    for table in tables:
        want = sorted(canonical(row) for row in expected_state.get(table, []))
        got = sorted(canonical(row) for row in actual_state.get(table, []))
        if want != got:
            differences[table] = {'missing': sorted(set(want) - set(got))[:5],
                                  'unexpected': sorted(set(got) - set(want))[:5]}
    return differences


def state_changed(initial, final):
    return bool(state_diff(initial, final))
