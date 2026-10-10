"""Held-out synthetic instances of established business families, not unseen domains."""
import copy
from prepare_v2_natural_pilot import cases
from v2_natural_oracle import expand

def heldout_cases():
    result = copy.deepcopy(cases())
    for index, case in enumerate(result, 1):
        prefix = f"H{index:02d}_"
        identities = {row['id']: prefix + row['id'] for rows in expand(case['business']).values()
                      for row in rows if row['id'] not in {'F1', 'F2'}}
        def rewrite(value):
            if isinstance(value, str): return identities.get(value, value)
            if isinstance(value, list): return [rewrite(v) for v in value]
            if isinstance(value, dict): return {k: rewrite(v) for k,v in value.items()}
            return value
        case.update(rewrite(case))
        for table in case['business']['tables']:
            if 'generator' in table:
                plan = table['generator']
                plan['seed'] = index + 7
                plan['columns']['id']['prefix'] = prefix + plan['columns']['id']['prefix']
            for row in table.get('rows', []):
                for key in ['points', 'amount', 'duration', 'capacity', 'stock', 'remaining', 'inspection_fee']:
                    if type(row.get(key)) is int:
                        row[key] += index % 5 + 1
        p = case['acceptance']['params']
        kind = case['acceptance']['kind']
        if kind == 'energy': p.update(watts=[920, 85], hours=[4, 6], price=0.82)
        elif kind == 'discount': p['prices'] = [147, 109]
        elif kind == 'duration': p.update(start_minutes=615, end_minutes=790)
        elif kind == 'ranking':
            for i, candidate in enumerate(p['candidates']):
                candidate['price'] += (i + 1) * 31
        case['business']['provenance'] = 'held-out synthetic instance; frozen before contract generation'
    return result
