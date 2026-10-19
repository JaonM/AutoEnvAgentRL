"""World variants: original data, flipped approvals and generic perturbations."""
from __future__ import annotations

import copy
import random
import re

from env_factory.runtime import canonical


def approval_flips(facts):
    """Each boolean approve* fact flipped on its own."""
    variants = []
    for key, value in sorted(facts.items()):
        if key.startswith('approve') and isinstance(value, bool):
            changed = copy.deepcopy(facts)
            changed[key] = not value
            variants.append((f'flip:{key}', changed))
    return variants


CALENDAR_SHIFTS = (-3 * 1440, -1440, -120, -60, 60, 120, 1440, 3 * 1440)  # minutes: hours and days around a boundary


def _shift(value, minutes):
    import datetime
    moment = datetime.datetime.fromisoformat(value) + datetime.timedelta(minutes=minutes)
    return moment.strftime('%Y-%m-%dT%H:%M') if 'T' in value else moment.strftime('%Y-%m-%d')


def _calendar_cell(value):
    import datetime
    if not (isinstance(value, str) and re.match(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$', value)):
        return False
    try:
        datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


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
                if isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)) \
                        or _calendar_cell(value):
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
        if isinstance(value, str):  # a calendar date or time: moved by hours or days, across deadlines
            minutes = rng.choice(CALENDAR_SHIFTS if 'T' in value else [m for m in CALENDAR_SHIFTS if m % 1440 == 0])
            world[table][index][column] = _shift(value, minutes)
            continue
        peers = sorted({row[column] for row in world[table] if isinstance(row, dict)
                        and isinstance(row.get(column), int) and not isinstance(row.get(column), bool)})
        options = sorted({max(0, value + d) for d in (-3, -2, -1, 1, 2, 3)} | set(peers) | {0})
        options = [v for v in options if v != value] or [value + 1]
        world[table][index][column] = rng.choice(options)
    return world


def variants(data, facts, *, perturbed=20, version=1, diagnostics=None):
    """[(name, data, facts)] in a fixed order: original, approval flips, perturbed worlds."""
    stats = {'attempted': 0, 'duplicates_removed': 0}
    worlds = [('original', copy.deepcopy(data), copy.deepcopy(facts))]
    worlds += [(name, copy.deepcopy(data), changed) for name, changed in approval_flips(facts)]
    if version == 1:
        stats['attempted'] = perturbed
        worlds += [(f'perturb:{seed}', perturb(data, seed), copy.deepcopy(facts)) for seed in range(1, perturbed + 1)]
    elif version == 2:
        seen = {canonical(data)}
        accepted = 0
        cells = _v2_cells(data)
        for seed in range(1, max(0, perturbed) * 20 + 1):
            stats['attempted'] += 1
            changed = perturb_v2(data, seed, cells=cells)
            signature = canonical(changed)
            if signature in seen:
                stats['duplicates_removed'] += 1
                continue
            seen.add(signature)
            worlds.append((f'perturb:{seed}', changed, copy.deepcopy(facts)))
            accepted += 1
            if accepted >= perturbed:
                break
    else:
        raise ValueError('unsupported perturbation version')
    if diagnostics is not None:
        diagnostics.update(stats)
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


ENUM_FIELDS = frozenset({'status', 'state', 'tier', 'type', 'category', 'mode'})


def _leaves(value, prefix=()):
    if isinstance(value, dict):
        for key in sorted(value):
            if key == 'id' or key.endswith('_id') or key.endswith('_ids'):
                continue
            yield from _leaves(value[key], prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, prefix + (index,))
    else:
        yield prefix, value


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def _put(value, path, replacement):
    _at(value, path[:-1])[path[-1]] = replacement


def _options(data, table, path, value, enums=None):
    from decimal import Decimal
    import math
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)) and math.isfinite(value):
        if isinstance(value, int):
            return sorted({max(0, value + d) for d in (-3, -2, -1, 1, 2, 3)} | {0})
        number = Decimal(str(value))
        return sorted({float(max(Decimal(0), number + Decimal(d))) for d in ('-1', '-.01', '.01', '1')} | {0.})
    if _calendar_cell(value):
        shifts = CALENDAR_SHIFTS if 'T' in value else [m for m in CALENDAR_SHIFTS if m % 1440 == 0]
        return sorted({_shift(value, m) for m in shifts})
    if isinstance(value, str) and path[-1] in ENUM_FIELDS:
        if enums is not None:
            return sorted(enums.get((table, path), set()))
        peers = set()
        for row in data[table]:
            try:
                other = _at(row, path)
                if isinstance(other, str):
                    peers.add(other)
            except (KeyError, IndexError, TypeError):
                pass
        return sorted(peers)
    return []


def _v2_cells(data):
    cells, enums = [], {}
    for table in sorted(data):
        for row in data[table]:
            for path, value in _leaves(row):
                if isinstance(value, str) and path and path[-1] in ENUM_FIELDS:
                    enums.setdefault((table, path), set()).add(value)
    for table in sorted(data):
        for index, row in enumerate(data[table]):
            for path, value in _leaves(row):
                options = [v for v in _options(data, table, path, value, enums) if v != value]
                if options:
                    cells.append((table, index, path, options))
    return cells


def perturb_v2(data, seed, *, cells=None):
    rng, world = random.Random(seed), copy.deepcopy(data)
    cells = _v2_cells(data) if cells is None else cells
    for table, index, path, options in rng.sample(cells, min(len(cells), rng.randint(1, 3))):
        _put(world[table][index], path, rng.choice(options))
    return world


def request_variants(request, *, perturbed=20, diagnostics=None):
    """Versioned generation keeps existing bundles' world names and contents stable."""
    worlds = variants(request['tables'], request['user_facts'], perturbed=perturbed,
                      version=request.get('perturbation_version', 1), diagnostics=diagnostics)
    if perturbed:
        for case in request.get('perturbation_cases', []):
            data = copy.deepcopy(request['tables'])
            for change in case['changes']:
                row = next(row for row in data[change['table']] if row.get('id') == change['id'])
                path = change['path']
                if not path or any(isinstance(k, str) and (k == 'id' or k.endswith('_id') or k.endswith('_ids')) for k in path):
                    raise ValueError('perturbation cases cannot change identities or relations')
                if isinstance(_at(row, path), (dict, list)) or isinstance(change['value'], (dict, list)):
                    raise ValueError('perturbation changes must target scalar fields')
                _put(row, path, copy.deepcopy(change['value']))
            worlds.append(('perturb:case:' + case['id'], data, copy.deepcopy(request['user_facts'])))
    return worlds


def perturbation_coverage(request, worlds, *, requested):
    stats = {}
    candidates = request_variants(request, perturbed=requested, diagnostics=stats)
    original = canonical([request['tables'], request['user_facts']])
    seen, paths = {original}, set()
    duplicates = unchanged = generated = 0
    for name, data, facts in candidates:
        if not name.startswith('perturb'):
            continue
        generated += 1
        signature = canonical([data, facts])
        unchanged += signature == original
        duplicates += signature in seen
        seen.add(signature)
        for table, rows in request['tables'].items():
            for index, row in enumerate(rows):
                for path, value in _leaves(row):
                    if _at(data[table][index], path) != value:
                        paths.add(table + '.' + '.'.join(map(str, path)))
    return {**stats, 'version': request.get('perturbation_version', 1), 'requested': requested,
            'generated': generated, 'unique_changed': generated - duplicates,
            'duplicates': duplicates, 'unchanged': unchanged, 'changed_paths': sorted(paths),
            'accepted': sum(w['world'].startswith('perturb') and w['status'] == 'agree' for w in worlds),
            'rejected': sum(w['world'].startswith('perturb') and w['status'] != 'agree' for w in worlds),
            'declared_rule_ids': sorted({r for case in request.get('perturbation_cases', []) for r in case.get('rule_ids', [])})}
